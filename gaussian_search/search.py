# Copyright (C) 2021  Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Gaussian process grid search class."""

import os
import sys
import warnings
from datetime import datetime
from copy import deepcopy

import numpy
from scipy.stats import uniform

from sklearn.gaussian_process import (GaussianProcessRegressor, kernels)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
import joblib

from dynesty import NestedSampler
from dynesty import utils as dyfunc


class GaussianProcessSearch:
    r"""
    GaussianProcessSearch is an adaptive grid search that builds a surrogate
    Gaussian process model on top of the expensive likelihood, which serves
    to decide which points to sample next to maximise the information gain and
    to sample the surrogate model.

    New points from either the surrogate model or the acquisition function are
    sampled with `dynesty.NestedSampler`. The acquisition function is defined
    as

        . math::
            y = \mu(x) + \kappa * \sigma(x),

    where :math:`\mu` and :math:`\sigma` are the mean and standard devitation
    of the Gaussian process at :math:`x`. :math:`kappa` controls exploration,
    however new points are sampled from the acqusition function, not from the
    peaks of the acquisition function, and therefore the model is capable of
    exploring the parameter space well.

    To achieve paralellisation new points are sampled in batches. Typically
    this means that first :math:`N` points are sampled from the acquisition
    function, then evaluated in parallel, and at the end the Gaussian process
    is retrained before moving on to the next batch. First few batches are
    distributed uniformly over the prior boundaries.

    The grid search terminates after a given number of batches was explored or,
    alternatively, after the information gain between batches has stagnated
    (as measured by the KL divergence).

    The grid search automatically checkpoints and can be easily restarted
    from the checkpoints, at which point it can be instructed to also sample
    user-specified points. Output is automatically dumped in `./out.`

    Parameters
    ----------
    name : str
        A unique name to distinguish this grid search run.
    params : list of str
        Target function's parameters.
    logmodel : py:func
        Logarithm of the target function.
    bounds : dict
        Dictionary of prior boundaries.
    nthreads : int, optional
        Number of threads. By default 1.
    kappa : float, optional
        Parameter controlling the acquisition function. Typically values that
        are larger than 1 are sufficient. By default 5.
    gp : `sklearn.gaussian_process.GaussianProcessRegressor`
        The underlying Gaussian process and kernel. By default Matern kernel.
    hyper_grid : dict, optional
        Hyperparameter grid to be explored when fitting the Gaussian process.
        By default `None`, the default hyperparameters are used.
    stopping_tolerance : float, optional
        Relative information gain between batches stopping tolerance. Typically
        :math:`1e-3`. If the information gain is less than this threshold for
        `self.patience` batches the grid search will terminate. By default
        `None`, the grid search does not employ this termination strategy.
    patience : int, optional
        Number of batches to wait whether to terminate the grid search.
    random_state : int, optional
        Initial random state.
    verbose : bool, optional
        Verbosity flag, by default `True`.
    sampler_kwargs : **kwargs
        Keyword arguments passed into `dynesty.NestedSampler`. Includes
        multiprocessing pool.
    """
    def __init__(self, name, params, logmodel, bounds, nthreads=1, kappa=5.,
                 gp=None, hyper_grid=None, stopping_tolerance=None,
                 patience=None, random_state=None, verbose=True,
                 sampler_kwargs=None):
        self._name = None
        self._params = None
        self._logmodel = logmodel
        self._bounds = None
        self._nthreads = None
        self._kappa = None
        self._pdist = None
        self._stopping_tolerance = None
        self._patience = None
        self._verbose = None
        self._X = None
        self._y = None
        self._blobs = None
        self._surrogate_model = None
        self._prior_min = None
        self._prior_width = None
        if sampler_kwargs is None:
            self._sampler_kwargs = {}
        else:
            self._sampler_kwargs = sampler_kwargs

        # Set the random state
        if isinstance(random_state, numpy.random.RandomState):
            self.generator = random_state
        else:
            self.generator = numpy.random.RandomState(random_state)

        self.name = name
        self.params = params
        self.bounds = bounds
        self.nthreads = nthreads
        self.kappa = kappa
        self._initialise_gp(gp, hyper_grid)
        self.verbose = verbose
        self._set_stopping(stopping_tolerance, patience)
        # Check if the temporary results folder exists
        if not os.path.exists('./temp/'):
            os.mkdir('./temp/')
        # Warn if the checkpoint file exists
        self._checkpoint_path = './temp/checkpoint_{}.z'.format(self.name)
        if os.path.isfile(self._checkpoint_path):
            warnings.warn("Temporal checkpoint at {} exists, will be "
                          "overwritten.".format(self._checkpoint_path),
                          UserWarning)
        # Will be used to checkpoint the grid search
        self._state = {'name': self.name,
                       'params': self.params,
                       'bounds': self.bounds,
                       'nthreads': self.nthreads,
                       'gp': gp,
                       'hyper_grid': hyper_grid,
                       'verbose': self.verbose}
        # Will store the previous batch's fitted Gaussian process
        self._previous_gp = None
        # Count batches
        self._batch_iter = 0
        self._batch_entropies = []

    @property
    def name(self):
        """
        Name of this grid search run. Used to store outputs.

        Returns
        -------
        name : str
            Run name.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets `name`.
        """
        if not isinstance(name, str):
            raise TypeError("`name` '{}' must be of str type.".format(name))
        self._name = name

    @property
    def params(self):
        """
        Grid search's parameters passed into `self.logmodel`.

        Returns
        -------
        params : list of str
            Parameter names.
        """
        return self._params

    @params.setter
    def params(self, params):
        """
        Sets `params`. Ensures it is a list of strings.
        """
        if not isinstance(params, (list, tuple)):
            raise TypeError("`params` must be a list.")
        params = list(params)
        for par in params:
            if not isinstance(par, str):
                raise TypeError("Parameter '{}' must be a string".format(par))
        self._params = params

    @property
    def logmodel(self):
        """
        Logarithm (natural) of the target to be approximated by the surrogate
        Gaussian process model.

        Returns
        -------
        logmodel : py:function
            Target model.
        """
        return self._logmodel

    @property
    def bounds(self):
        """
        The prior boundaries.

        Returns
        -------
        bounds : dict
            Prior boundaries {`parameter`: (min, max)}
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """Sets `bounds` and stores the uniform distributions."""
        if not isinstance(bounds, dict):
            raise TypeError("`bounds` must be a dict.")
        # Check each parameter has a boundary
        for par in self.params:
            if par not in bounds.keys():
                raise ValueError("Parameter '{}' is missing a boundary."
                                 .format(par))
        # Check each entry is a len-2 tuple
        for par, bound in bounds.items():
            if not isinstance(bound, (list, tuple)) or len(bounds) != 2:
                raise TypeError("Parameter's '{}' boundary '{}' must be a "
                                "len-2 tuple.".format(par, bound))
            # Ensure correct ordering
            if bound[0] > bound[1]:
                bounds.update({par: bound[::-1]})

        self._bounds = bounds
        # Prior uniform distribution for the target
        self._prior_min = numpy.array([bnd[0] for bnd in self.bounds.values()])
        self._prior_max = numpy.array([bnd[1] for bnd in self.bounds.values()])
        self._pdist = uniform(loc=self._prior_min,
                              scale=self._prior_max - self._prior_min)

    @property
    def nthreads(self):
        """
        Number of threads used to evaluate the target model. The acquisition
        function is sampled using a single thread.

        Returns
        -------
        nthreads : int
            Number of parallel threads.
        """
        return self._nthreads

    @nthreads.setter
    def nthreads(self, nthreads):
        """
        Sets `nthreads`. Ensures it is an integer.
        """
        if nthreads is None:
            self._nthreads = 1
        else:
            if not isinstance(nthreads, int):
                raise TypeError("`nthreads` must be of int type.")
            self._nthreads = nthreads

    @property
    def kappa(self):
        """
        Acquisition function parameter, controls the contribution of the
        Gaussian process's standard deviation. See class documentation for
        more details.

        Returns
        -------
        kappa : float
            Acquisition function parameter.
        """
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        """Sets `kappa`. Ensures it is a scalar."""
        if not isinstance(kappa, (int, float)):
            raise TypeError("`kappa` must be a positive scalar.")
        self._kappa = kappa

    def _initialise_gp(self, gp, hyper_grid):
        """
        Initialises the Gaussian process surrogate model. If `gp` is `None`
        uses the default kernel and Gaussian process:

            `kernel = sklearn.gaussian_process.kernels.Matern(nu=2.5)`
            `gp = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5,
                random_state=self.generator)`,

        such that `random_state` is always set to the class generator. The
        data is always scaled using `sklearn.preprocessing.StandardScaler`.

        If `hyper_grid` is not `None` the best fit combination will be used
        as a surrogate model (calls `sklearn.model_selection.GridSearchCV`)
        with 5-fold cross-validation.

        Parameters
        ----------
        gp : None or `sklearn.gaussian_process.GaussianProcessRegressor`
            Surrogate model Gaussian process.
        hyper_grid : None or dict of dictionaries
            Hyperparameter grid to be explored when fitting the Gaussian
            process.
        """
        # Set up the Gaussian process, pipeline and grid search
        if gp is None:
            kernel = kernels.Matern(nu=2.5)
            gp = GaussianProcessRegressor(kernel, alpha=1e-6, normalize_y=True,
                                          n_restarts_optimizer=5,
                                          random_state=self.generator)
        elif not isinstance(gp, GaussianProcessRegressor):
            raise TypeError("`gp` must be of {} type."
                            .format(GaussianProcessRegressor))
        else:
            # Always overwrite the random state
            gp.random_state = self.generator
        # Set up the pipeline to scale the data
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('gp', gp)])
        # Optionally set the hyperparameter grid
        if hyper_grid is None:
            self._surrogate_model = pipe
        else:
            self._surrogate_model = GridSearchCV(pipe, hyper_grid,
                                                 n_jobs=self.nthreads)

    def _set_stopping(self, stopping_tolerance, patience):
        """
        Sets the relative entropy termination parameters. Ensures either
        both are `None` or both set.
        """
        if stopping_tolerance is None and patience is None:
            return
        err = ("`stopping_tolerance` and `patience` must either be both `None`"
               " or both assigned values")
        if stopping_tolerance is None and patience is not None:
            raise ValueError(err)
        elif stopping_tolerance is not None and patience is None:
            raise ValueError(err)
        if not isinstance(stopping_tolerance, float):
            raise TypeError("`stopping_tolerance` must be of float type.")
        if not isinstance(patience, int):
            raise TypeError("`patience` must be of int type.")
        self._stopping_tolerance = stopping_tolerance
        self._patience = patience

    @property
    def stopping_tolerance(self):
        """
        Relative information gain fluctuation tolerance between batches.

        Returns
        -------
        stopping_tolerance : float
            Information gain tolerance.
        """
        return self._stopping_tolerance

    @property
    def patience(self):
        """
        Number of consecutive batch iterations that must satisfy stopping
        tolerance to terminate the grid seach.

        Returns
        -------
        patience : int
            Number of patience batch iterations.
        """
        return self._patience

    @property
    def verbose(self):
        """
        Whether to print checkpoint messages.

        Returns
        -------
        verbose : bool
            Verbosity flag.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """
        Sets `verbose`. Checks it is a bool.
        """
        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be of bool type")
        self._verbose = verbose

    def run_points(self, X, to_save=False, kwargs=None, to_refit=True):
        """
        Samples points specified by `X`, runs in parallel. After points are
        sampled retrains the Gaussian process.

        Parameters
        ----------
        X : numpy.ndarray (npoints, nfeatures)
            Array of points to be sampled.
        to_save : bool, optional
            Whether to save the points upon evaluation. By default `False` and
            newly sampled points are only saved upon termination. However a
            checkpoint is always stored.
        kwargs : dict, optional
            Keyword arguments passed into `self.logmodel` that are not the
            sampled positions.
        to_refit : bool, optional
            Whether to refit the Gaussian process. By default `True`.
        """
        # Unpack X into a list of dicts
        points = [{attr: X[i, j] for j, attr in enumerate(self.params)}
                  for i in range(X.shape[0])]
        # Process the points in parallel
        if self.verbose:
            print("{}: Evaluating {} samples.".format(datetime.now(),
                                                      len(points)))
            sys.stdout.flush()

        with joblib.Parallel(n_jobs=self.nthreads) as par:
            if kwargs is None:
                res = par(joblib.delayed(self.logmodel)(point)
                          for point in points)
            else:
                res = par(joblib.delayed(self.logmodel)(point, **kwargs)
                          for point in points)
        # Figure out whether we have any blobs
        if isinstance(res[0], tuple):
            targets = [out[0] for out in res]
            blobs = [out[1] for out in res]
        else:
            targets = res

        # Append the results
        if self._X is None:
            self._X = X
            self._y = numpy.array(targets)
            # If we have any blobs store those
            try:
                self._blobs = blobs
            except NameError:
                pass
        else:
            self._X = numpy.vstack([self._X, X])
            self._y = numpy.hstack([self._y, targets])
            try:
                for blob in blobs:
                    self._blobs.append(blob)
            except NameError:
                pass

        if to_refit:
            self._refit_gp()
        self.save_checkpoint()

        if to_save:
            self.save_grid()

        # Bump up the batch iteration counter
        self._batch_iter += 1

    def _refit_gp(self):
        """
        Refits the Gaussian process with internally stored points `self._X`
        and `self._y`
        """
        if self.verbose:
            print("{}: Refitting the Gaussian process.".format(datetime.now()))
            sys.stdout.flush()
        self._previous_gp = deepcopy(self._surrogate_model)
        # Silence convergence warning of the GaussianRegressor's optimiser
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            self._surrogate_model.fit(self._X, self._y)

    def run_batches(self, Ninit=0, Nmcmc=0, batch_size=5, to_save=True,
                    beta=1, kwargs=None):
        r"""
        Samples `Ninit` and `Nmcmc` in batches of size `batch_size`. If any
        `kwargs` are passed, these will be memory-mapped to the disk for
        faster parallel computation.

        Parameters
        ----------
        Ninit : int, optional
            Number of initially uniformly sampled batches. By default 0.
        Nmcmc : int, optional
            Number of batches from the acquisition function. By default 0.
        batch_size : int, optional
            Batch size, determines how many points are sampled without
            without updating the surrogate Gaussian Process. Typically
            are evaluated in parallel.
        to_save : bool
            Whether to save the points upon evaluation. By default `True` and
            newly sampled points are only saved upon termination.
        beta : float, optional
            Parallel tempering parameter :math:`L^{\beta}`, where :math:`L`
            is the target function.
        kwargs : dict
            Keyword arguments passed into `self.logmodel` that are not the
            sampled positions.
        """
        memmap_path = None
        # If any kwargs passed in create a memory mapping
        if kwargs is not None:
            memmap_path = "./temp/memmap_{}".format(self.name)
            if os.path.isfile(memmap_path):
                warnings.warn("Temporal memory map at {} exists, will be "
                              "overwritten.".format(memmap_path), UserWarning)

            joblib.dump(kwargs, memmap_path)
            kwargs = joblib.load(memmap_path, mmap_mode='r')

        if Ninit == 0 and Nmcmc == 0:
            raise ValueError("Both `Ninit` and `Nmcmc` are 0, exiting.")

        # Initial batches sampled from the prior
        for i in range(Ninit):
            X = self._uniform_samples(batch_size)
            if i != Ninit - 1:
                self.run_points(X, kwargs=kwargs, to_refit=False)
            else:
                self.run_points(X, kwargs=kwargs, to_refit=True)
            if self.verbose:
                print("{}: Completed {}/{} uniform iterations."
                      .format(datetime.now(), i+1, Ninit))
                sys.stdout.flush()

        # Batches sampled from the acquisition function
        for i in range(Nmcmc):
            X = self._acquisition_samples(beta=beta)

            if batch_size > X.shape[0]:
                raise ValueError("Cannot ask for larger batch size ({}) than "
                                 "the number of sampled points {}"
                                 .format(batch_size, X.shape[0]))
            # Down-sample the samples we will evaluate
            mask = self.generator.choice(X.shape[0], batch_size, replace=False)
            self.run_points(X[mask, :], kwargs=kwargs, to_refit=True)
            # Calculate the relative entropy with acqusition function samples
            if self.stopping_tolerance is not None and self._batch_iter > 0:
                entropy = self.relative_entropy(X)
                self._batch_entropies.append([self._batch_iter, entropy])
            if self._to_terminate():
                if self.verbose:
                    print("{}: Terminating, entropy condition met."
                          .format(datetime.now()))
                    sys.stdout.flush()
                break
            if self.verbose:
                print("{}: Completed {}/{} MCMC iterations."
                      .format(datetime.now(), i+1, Nmcmc))
                sys.stdout.flush()

        if self.verbose and not self._to_terminate():
            print("{}: Terminating, number of requested iterations reached."
                  .format(datetime.now()))
            sys.stdout.flush()
        # Clean up the memory mapping
        if memmap_path is not None:
            os.remove(memmap_path)

        if to_save:
            self.save_grid()

    def surrogate_predict(self, X, kappa=0, beta=1):
        r"""
        Evaluates the surrogate model at positions `X`, such that

            .. math::
                y = \mu + \kappa * \sigma,

        where :math:`\mu` and :math:`\sigma` is the mean and standard
        devitation of the fitted Gaussian process regressor at `X`.
        Non-zero values of :math:`\kappa` are used in the acquisition function
        to reward exploring unknown areas of the prior space.

        Parameters
        ----------
        X : numpy.ndarray (Npoints, Nfeatures)
            Array of positions to be evaluated by the surrogate model.
        kappa : int, optional
            Parameter controlling the contribution of the Gaussian process's
            standar deviation. By default 0.
        beta : float, optional
            Parallel tempering parameter :math:`L^{\beta}`, where :math:`L`
            is the target function.

        Returns
        -------
        ypred : numpy.ndarray
            Array of predicted values.
        """
        # Needs to be reshaped for the predictor if 1D array
        ndim = X.ndim
        if ndim == 1:
            X = X.reshape(1, -1)
        if kappa != 0:
            # GaussianRegressor raises warning when std=0. Silence it
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                if isinstance(self._surrogate_model, GridSearchCV):
                    mu, std = self._surrogate_model.best_estimator_.predict(
                            X, return_std=True)
                else:
                    mu, std = self._surrogate_model.predict(X, return_std=True)
            ypred = mu + kappa * std
        else:
            ypred = self._surrogate_model.predict(X)

        # Again if 1D input return just a float
        if ndim == 1:
            ypred = ypred[0]
        # Assuming that ypred is logarithm of the target function
        return beta * ypred

    def relative_entropy(self, samples):
        """
        Approximates the relative entropy (Kullback???Leibler divergence, i.e.
        the information gain) between the currently fitted Gaussian process
        and the previous Gaussian process.

        Parameters
        ----------
        samples : numpy.ndarray
            Samples from the target distribution.

        Returns
        -------
        relative_entropy : float
            The relative entropy between the current and previous Gaussian
            process.
        """
        logP = self.surrogate_predict(samples)
        logQ = self._previous_gp.predict(samples)
        return numpy.sum(numpy.exp(logP) * (logP - logQ))

    def _prior_transform(self, u):
        """
        Inverse uniform cdf over the prior range for the nested sampler.

        Parameters
        ----------
        u : numpy.ndarray
            Array of random numbers.

        Returns
        -------
        sampled_points : numpy.ndarray
            Array of points from the prior corresponding to `u`.
        """
        return self._pdist.ppf(u)

    def surrogate_posterior_samples(self):
        """
        Draws samples from the surrogate model defined by the Gaussian process,
        calls the nested sampler (Dynesty) and also returns the log evidence.

        Returns
        -------
        X : numpy.recarray
            Samples drawn from the surrogate model.
        logz : int
            Log evidence as returned by the nested sampler.
        """
        if self.verbose:
            print("{}: Sampling the surrogate model.".format(datetime.now()))
            sys.stdout.flush()
        samples, target, logz = self._samples(kappa=0, return_full=True)
        X = numpy.hstack([samples, target.reshape(-1, 1)])
        X = numpy.core.records.fromarrays(X.T, names=self.params + ['target'])
        return X, logz

    def _acquisition_samples(self, beta=1):
        r"""
        Draws samples from the acquisition function, calls the nested sampler
        (Dynesty).

        Parameters
        ----------
        beta : float, optional
            Parallel tempering parameter :math:`L^{\beta}`, where :math:`L`
            is the target function.

        Returns
        -------
        X : np.ndarray
            Samples drawn from the acquisition function.
        """
        if self.verbose:
            print("{}: Sampling the acquisition function."
                  .format(datetime.now()))
            sys.stdout.flush()
        return self._samples(kappa=self.kappa, beta=beta)

    def _samples(self, kappa, beta=1, return_full=False, print_progress=False):
        r"""
        Draws samples from the surrogate model (optionally the acquisition
        function). Calls `dynesty.NestedSampler`. Runs on a single thread even
        if `self.nthreads > 1`.

        Parameters
        ----------
        kappa : int
            Acquisition function parameter. See class documentation for more
            information.
        beta : float, optional
            Parallel tempering parameter :math:`L^{\beta}`, where :math:`L`
            is the target function. By default 1.
        return_full : bool, optional
            Whether to also return the sampled log-target values and the
            target evidence.
        print_progress : bool, optional
            Whether to print the sampler's progress bar.

        Returns
        -------
        samples : np.ndarray
            Sampled points from the surrogate model.
        logtarget : np.ndarray
            Optionally returned if `return_full=True`, the surrogate model
            target values.
        logz : int
            Optionally returned if `return_full=True`, the surrogate model
            evidence.
        """
        sampler = NestedSampler(
                self.surrogate_predict, self._prior_transform,
                ndim=len(self.params),
                logl_kwargs={'kappa': kappa, 'beta': beta},
                rstate=self.generator, **self._sampler_kwargs)
        sampler.run_nested(print_progress=print_progress)

        results = sampler.results
        logz = results.logz[-1]
        weights = numpy.exp(results.logwt - logz)
        # Resample from the posterior
        samples = dyfunc.resample_equal(results.samples, weights)
        if return_full:
            logtarget = self.surrogate_predict(samples)
            # We're only interested in the contribution to the evidence the
            # likelihood (our target function). The uniform prior is used only
            # to provide samples, hence undo its contribution.
            logprior = -numpy.log(self._prior_max - self._prior_min).sum()
            logz -= logprior
            return samples, logtarget, logz
        return samples

    def _uniform_samples(self, N):
        """
        Samples `N` points from a uniform distribution within the boundaries.

        Parameters
        ----------
        N : int
            Number of points.

        Returns
        -------
        points : numpy.ndarray (N, 2)
            Randomly sampled points.
        """
        return self.generator.uniform(low=self._prior_min,
                                      high=self._prior_max, size=(N, 2))

    def _to_terminate(self):
        """
        Whether to terminate the search, according to the information gain
        criterion. For more information see class documentation.

        Returns
        -------
        to_terminate : bool
            Whether to terminate the grid search.
        """
        if self.stopping_tolerance is None:
            return False
        entrs = self.batch_entropies
        entrs[:, 1] = numpy.abs(entropies[:, 1])
        if entrs.shape[0] < self.patience:
            return False
        elif numpy.all(entrs[-self.patience:, 1] < self.stopping_tolerance):
            return True
        return False

    @property
    def batch_entropies(self):
        """
        Relative entropies between batches.

        Returns
        -------
        batch_entropies : numpy.ndarray (Nbatches, 2)
            Batches' relative entropy. Fist column is the batch iteration,
            second column gives the relative entropy.
        """
        return numpy.array(self._batch_entropies)

    @property
    def positions(self):
        """
        Positions stepped by the grid search.

        Returns
        -------
        positions : numpy.recarray
            Structured array with sampled positions.
        """
        return numpy.core.records.fromarrays(numpy.copy(self._X).T,
                                             names=self.params)

    @property
    def stats(self):
        """
        Target at the stepped positions.

        Returns
        -------
        target : numpy.ndarray
            Target values.
        """
        return numpy.copy(self._y)

    @property
    def blobs(self):
        """
        Blobs returned by the target model.

        Returns
        -------
        blobs : list
            List of blobs
        """
        return self._blobs

    @property
    def current_state(self):
        """
        Current state of the grid search.

        Returns
        -------
        checkpoint : dict
            Current state of the grid search
        """
        self._state.update({'X': self._X,
                            'y': self._y,
                            'blobs': self.blobs,
                            'surrogate_model': self._surrogate_model,
                            'random_state': self.generator})
        return self._state

    def save_checkpoint(self):
        """
        Checkpoints the grid.

        Stores the sampled point along with the random state.
        """
        checkpoint = self.current_state
        if self.verbose:
            print("{}: Checkpoint saved at {}".format(datetime.now(),
                                                      self._checkpoint_path))
            sys.stdout.flush()
        joblib.dump(checkpoint, self._checkpoint_path)

    def save_grid(self):
        """
        Saves the grid results upon termination (grid checkpoint, surrogate
        model samples and surrogate model evidence).
        """
        fpath = './out/{}/'.format(self.name)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        # Sample the surrogate model
        samples, logz = self.surrogate_posterior_samples()
        # Save the samples
        numpy.save(fpath + "surrogate_samples.npy", samples)
        # Save the evidence
        with open(fpath + "logz.txt", 'w') as f:
            f.write(str(logz))
        # Save the sampled points with blobs, if any
        out = {'params': self.positions,
               'stats': self.stats}
        if self.blobs is not None:
            out.update({'blobs': self.blobs})
        joblib.dump(out, fpath + 'samples.z')
        # Dump the checkpoint
        checkpoint = self.current_state
        joblib.dump(checkpoint, fpath + 'checkpoint.z')
        if self.verbose:
            print("{}: Output saved at {}.".format(datetime.now(), fpath))
            sys.stdout.flush()
        # Remove the temporary checkpoint
        os.remove(self._checkpoint_path)

    @classmethod
    def from_checkpoint(cls, logmodel, checkpoint):
        """
        Loads the grid search from a checkpoint.

        Parameters
        ----------
        logmodel : py:func
            Logarithmic target model. Must match the model used to get this
            checkpoint.
        checkpoint : dict
            Checkpoint returned from `self.current_state`.

        Returns
        -------
        grid : `BayesianGridSearch` object
            Grid search object initialised from the checkpoint.
        """
        X = checkpoint.pop('X', None)
        y = checkpoint.pop('y', None)
        blobs = checkpoint.pop('blobs', None)
        surrogate_model = checkpoint.pop('surrogate_model', None)
        grid = cls(logmodel=logmodel, **checkpoint)
        grid._X = X
        grid._y = y
        grid._blobs = blobs
        if surrogate_model is None:
            grid._refit_gp()
        else:
            grid._surrogate_model = surrogate_model
        return grid
