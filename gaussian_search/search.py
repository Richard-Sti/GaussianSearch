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
    """
    TO DO:
        - Make possible to restart from a state (TEST)

        - joblib parallel dumping may reduce the overhead, check this out

        - determine KL divergence between current and last posterior whether
        any more information is being gained

        - correct switching between grid and a single model (TEST)

        - Refitting the GP becomes more and more expensive - increase batchsize
    """


    def __init__(self, name, params, model, bounds, nthreads=1, kappa=2.5,
                 gp=None, hyper_grid=None, random_state=None, verbose=True,
                 sampler_kwargs=None):
        self._name = None
        self._params = None
        self._model = model
        self._bounds = None
        self._nthreads = None
        self._kappa = None
        self._pdist = None
        self._verbose = None
        self._X = None
        self._y = None
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
                       'model': self.model,
                       'bounds': self.bounds,
                       'nthreads': self.nthreads,
                       'gp': gp,
                       'hyper_grid': hyper_grid,
                       'verbose': self.verbose}

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
        Grid search's parameters passed into `self.model`.

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
    def model(self):
        """
        Target model approximated by the surrogate Gaussian process model.

        Returns
        -------
        model : py:function
            Target model.
        """
        return self._model

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
            if not par in bounds.keys():
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
            gp = GaussianProcessRegressor(kernels.Matern(nu=2.5),
                    alpha=1e-6, normalize_y=True, n_restarts_optimizer=5,
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

    def run_points(self, X, to_save=False, kwargs=None):
        """
        Samples points specified by `X`, runs in parallel. After points are
        sampled retrains the Gaussian process.

        Parameters
        ----------
        X : numpy.ndarray (npoints, nfeatures)
            Array of points to be sampled.
        nthreads : int
            Number of jobs to be run in parallel.
        kwargs : dict
            Keyword arguments passed into `self.model` that are not the sampled
            positions.
        """
        if kwargs is None:
            kwargs = {}
        # Unpack X into a list of dicts
        points = [{attr: X[i, j] for j, attr in enumerate(self.params)}
                  for i in range(X.shape[0])]
        # Process the points in parallel
        if self.verbose:
            print("{}: evaluating {} samples."
                  .format(datetime.now(), len(points)))

        with joblib.Parallel(n_jobs=self.nthreads) as par:
            targets = par(joblib.delayed(self.model)(
                **self._merge_dicts(point, kwargs)) for point in points)
        # Append the results
        if self._X is None:
            self._X = X
            self._y = numpy.array(targets)
        else:
            self._X = numpy.vstack([self._X, X])
            self._y = numpy.hstack([self._y, targets])
        if self.verbose:
            print("{}: refitting the Gaussian process.".format(datetime.now()))
        self._refit_gp()
        self.save_checkpoint()

        if to_save:
            self.save_grid()

    @staticmethod
    def _merge_dicts(dict1, dict2):
        """
        A quick method to merge two dictionaries.

        Parameters
        ----------
        dict1 : dict
            First dictionary.
        dict2 : dict
            Second dictionary.

        Returns
        -------
        merged_dict : dict
            Merge of `dict1` and `dict2`.
        """
        dict1.update(dict2)
        return dict1

    def _refit_gp(self):
        """
        Refits the Gaussian process with internally stored points `self._X`
        and `self._y`
        """
        # Silence convergence warning of the GaussianRegressor's optimiser
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            self._surrogate_model.fit(self._X, self._y)

    def run_batches(self, Ninit=0, Nmcmc=0, batch_size=5, to_save=True,
                    kwargs=None):
        """
        Samples `Ninit` and `Nmcmc` in batches of size `batch_size`. If any
        `kwargs` are passed, these will be memory-mapped to the disk for faster
        parallel computation.


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
        kwargs : dict
            Keyword arguments passed into `self.model` that are not the sampled
            positions.
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
            warnings.warn("Both `Ninit` and `Nmcmc` are 0, exiting.",
                           UserWarning)
        # Initial batches sampled from the prior
        for __ in range(Ninit):
            X = self._uniform_samples(batch_size)
            self.run_points(X, kwargs=kwargs)
        # Batches sampled from the acquisition function
        for __ in range(Nmcmc):
            X = self._acquisition_samples(Nsamples=batch_size)
            self.run_points(X, kwargs=kwargs)
        # Clean up the memory mapping
        if memmap_path is not None:
            os.remove(memmap_path)

        if to_save:
            self.save_grid()

    def surrogate_predict(self, X, kappa=0):
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
        return ypred

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
            print("{}: sampling the surrogate model."
                  .format(datetime.now()))
        samples, target, logz = self._samples(kappa=0, return_full=True)
        X = numpy.hstack([samples, target.reshape(-1, 1)])
        X = numpy.core.records.fromarrays(X.T, names=self.params + ['target'])
        return X, logz

    def _acquisition_samples(self, Nsamples):
        """
        Draws `Nsamples` samples from the acquisition function, calls the
        nested sampler (Dynesty).

        Parameters
        ----------
        Nsamples : int
            Number of samples to be drawn from the acqusition function.

        Returns
        -------
        X : np.ndarray
            Samples drawn from the acquisition function.
        """
        if self.verbose:
            print("{}: sampling the acquisition function."
                  .format(datetime.now()))

        X = self._samples(kappa=self.kappa)
        if Nsamples > X.shape[0]:
            raise ValueError("Cannot ask for more samples `Nsamples = {}` than "
                             "the number of sampled points {}"
                             .format(Nsamples, X.shape[0]))
        return X[self.generator.choice(X.shape[0], Nsamples, replace=False), :]

    def _samples(self, kappa, return_full=False):
        """
        Draws samples from the surrogate model (optionally the acquisition
        function). Calls `dynesty.NestedSampler`. Runs on a single thread even
        if `self.nthreads > 1`.

        Parameters
        ----------
        kappa : int
            Acquisition function parameter. See class documentation for more
            information.
        return_full : bool, optional
            Whether to also return the sampled log-target values and the
            target evidence.

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
                ndim=len(self.params), logl_kwargs={'kappa': kappa},
                rstate=self.generator, **self._sampler_kwargs)
        sampler.run_nested()

        results = sampler.results
        logz = results.logz[-1]
        weights = numpy.exp(results.logwt - logz)
        # Resample from the posterior
        samples = dyfunc.resample_equal(results.samples, weights)
        if return_full:
            logl = dyfunc.resample_equal(results.logl, weights)
            # We're only interested in the contribution to the evidence the
            # likelihood (our target function). The uniform prior is used only
            # to provide samples, hence undo its contribution.
            logprior = -numpy.log(self._prior_max - self._prior_min).sum()
            logz -= logprior
            return samples, logl, logz
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
                            'random_state': deepcopy(self.generator)})
        return self._state

    def save_checkpoint(self):
        """
        Checkpoints the grid.

        Stores its parameters, model (`self.current_state`), sampled
        points (`self._X`, `self._y`), and the current random state.
        """
        checkpoint = self.current_state
        if self.verbose:
            print("Checkpoint saved at {}".format(self._checkpoint_path))
        joblib.dump(checkpoint, self._checkpoint_path)

    def save_grid(self):
        """
        Saves the grid results upon termination (grid checkpoint, surrogate
        model samples and surrogate model evidence).
        """
        samples, logz = self.surrogate_posterior_samples()
        # Optionally create the output folder
        fpath = './out/{}/'.format(self.name)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        # Save the evidence
        with open(fpath + 'logz.txt', 'w') as f:
            f.write(str(logz))
        # Save the checkpoint
        checkpoint = self.current_state
        joblib.dump(checkpoint, fpath + 'checkpoint.z')
        # Save the samples
        numpy.save(fpath + 'surrogate_samples.npy', samples)
        if self.verbose:
            print("Output saved at {}.".format(fpath))
        # Remove the temporary checkpoint
        os.remove(self._checkpoint_path)

    @classmethod
    def from_checkpoint(cls, checkpoint, Xnew=None):
        """
        Loads the grid search from a checkpoint. May manually assign points
        to sample.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint returned from `self.current_state`.
        Xnew : numpy.ndarray, optional
            Optional points to be manually sampled.

        Returns
        -------
        grid : `BayesianGridSearch` object
            Grid search object initialised from the checkpoint.
        """
        X = checkpoint.pop('X', None)
        y = checkpoint.pop('y', None)
        grid = cls(**checkpoint)
        grid._X = X
        grid._y = y
        # Save the random generator state and roll it back after fitting
        generator0 = grid.generator
        grid._refit_gp()
        grid.generator = generator0
        # If any new points sample those
        if Xnew is not None:
            grid.run_points(Xnew)
        return grid
