# GaussianSearch: An Adaptive Grid Search

GaussianSearch serves to sample computationally expensive functions, for which traditional methods such as Markov chain Monte Carlo or nested sampling become infeasible. GaussianSearch builds a surrogate Gaussian process model on top of the expensive function, with parallelisation achieved by evaluating the target function in batches, in between which the underlying Gaussian process is refitted.

Suggested points are sampled with a nested sampler (Dynesty) from the acquisition function, which is a sum of the Gaussian process mean and standard deviation, thus rewarding exploration.

## TO DO
[ ] Code documentation
[ ] Project description on Github
[ ] Pip install
[ ] Example notebooks
[ ] Unit testing


## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
