#!/usr/bin/env python

import numpy as np
import numbers
import warnings


def ci(a, bounds=95, axis=None):
    """
    Taken from seaborn utils.py
    Return a percentile range from an array of values

    Args:
        a (np.ndarray): array to get CIs from
        bounds: percentile limits
        axis: if multidimensional a

    Returns:
        Percentile range

    """
    p = 50 - bounds / 2, 50 + bounds / 2
    return np.percentile(a, p, axis)


def linear_fit(x, y):
    """
    Taken from seaborn regression.py
    Low-level regression and prediction using linear algebra
    Args:
        x (array): data x values
        y (array): data y values

    Returns:
        grid (array): x values for fit; 100 points spread linearly through data range
        yhat (array): y values for fit; 100 points
        yhat_boots (array): 1000x bootstrapped y values
        rsq (float): r squared value for linear fit

    """

    def reg_func(_x, _y):
        return np.linalg.pinv(_x).dot(_y)

    grid = np.linspace(np.min(x), np.max(x), 100)

    X, Y = np.c_[np.ones(len(x)), x], y
    grid_c = np.c_[np.ones(len(grid)), grid]
    yhat = grid_c.dot(reg_func(X, Y))

    beta_boots = bootstrap(X, Y, func=reg_func, n_boot=1000).T
    yhat_boots = grid_c.dot(beta_boots).T

    rsq = get_rsquared(x, y, grid, yhat)

    return grid, yhat, yhat_boots, rsq


def get_rsquared(x, y, fit_x, fit_y):
    """
    Get r squared value given original arrays and fitted arrays (from lin_fit)
    Args:
        x: original x data
        y: original y data
        fit_x: fitted x points (linear)
        fit_y: fitted y points (linear)

    Returns:
        rsq: r squared value

    """
    assert len(fit_x) == len(fit_y), 'Fitted line must have x and y of same length'

    p1 = np.r_[fit_x[0], fit_y[0]]
    p2 = np.r_[fit_x[-1], fit_y[-1]]

    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]

    fit_y_vals = np.asarray([m * i + b for i in x])
    y_mean = np.full(len(y), np.mean(y))

    ymean_diffs = y_mean - y
    y_diffs = fit_y_vals - y

    ss_meandiffs = sum(i * i for i in ymean_diffs)
    ss_diffs = sum(i * i for i in y_diffs)

    rsq = (ss_meandiffs - ss_diffs) / ss_meandiffs

    return rsq


def bootstrap(*args, **kwargs):
    """
    Taken from seaborn algorithms.py
    Resample one or more arrays with replacement and store aggregate values.
    Positional arguments are a sequence of arrays to bootstrap along the first
    axis and pass to a summary function.
    Args:
        *args:
        **kwargs:
            n_boot (int): default 10000
                Number of iterations
            axis (int): default None
                Will pass axis to ``func`` as kwarg
            units (np.ndarray): default None
                Array of sampling unit IDs. When used the bootstrap resamples units
                and then observations within units instead of individual
                datapoints.
            func (str or callable): default np.mean
                Function to call on the args that are passed in. If string, tries
                to use as named method on numpy array.
            seed (generator, SeedSequence, RandomState, int, None):
                Seed for the random number generator, useful if you want
                reproducible samples

    Returns:
    boot_dist: array
        array of bootstrapped statistic values
    """
    # Ensure list of arrays are same length
    if len(np.unique(list(map(len, args)))) > 1:
        raise ValueError("All input arrays must have the same length")
    n = len(args[0])

    # Default keyword arguments
    n_boot = kwargs.get("n_boot", 10000)
    func = kwargs.get("func", np.mean)
    axis = kwargs.get("axis", None)
    units = kwargs.get("units", None)
    random_seed = kwargs.get("random_seed", None)
    if random_seed is not None:
        msg = "`random_seed` has been renamed to `seed` and will be removed"
        warnings.warn(msg)
    seed = kwargs.get("seed", random_seed)
    if axis is None:
        func_kwargs = dict()
    else:
        func_kwargs = dict(axis=axis)

    # Initialize the resampler
    rng = _handle_random_seed(seed)

    # Coerce to arrays
    args = list(map(np.asarray, args))
    if units is not None:
        units = np.asarray(units)

    # Allow for a function that is the name of a method on an array
    if isinstance(func, str):
        def f(x):
            return getattr(x, func)()
    else:
        f = func

    # Handle numpy changes
    try:
        integers = rng.integers
    except AttributeError:
        integers = rng.randint

    # Do the bootstrap
    if units is not None:
        return _structured_bootstrap(args, n_boot, units, f,
                                     func_kwargs, integers)

    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(f(*sample, **func_kwargs))
    return np.array(boot_dist)


def _structured_bootstrap(args, n_boot, units, func, func_kwargs, integers):
    """Resample units instead of datapoints."""
    unique_units = np.unique(units)
    n_units = len(unique_units)

    args = [[a[units == unit] for unit in unique_units] for a in args]

    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n_units, n_units, dtype=np.intp)
        sample = [np.take(a, resampler, axis=0) for a in args]
        lengths = map(len, sample[0])
        resampler = [integers(0, n, n, dtype=np.intp) for n in lengths]
        sample = [[c.take(r, axis=0) for c, r in zip(a, resampler)]
                  for a in sample]
        sample = list(map(np.concatenate, sample))
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)


def _handle_random_seed(seed=None):
    """Given a seed in one of many formats, return a random number generator.
    Generalizes across the numpy 1.17 changes, preferring newer functionality.
    """
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        try:
            # General interface for seeding on numpy >= 1.17
            rng = np.random.default_rng(seed)
        except AttributeError:
            # We are on numpy < 1.17, handle options ourselves
            if isinstance(seed, (numbers.Integral, np.integer)):
                rng = np.random.RandomState(seed)
            elif seed is None:
                rng = np.random.RandomState()
            else:
                err = "{} cannot be used to seed the random number generator"
                raise ValueError(err.format(seed))
    return rng
