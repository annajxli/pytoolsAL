#!/usr/bin/env python3

import numpy as np
import numbers
import warnings
import sklearn


def norm(r):
    """
    Normalizes array in (array - mean)/(std)
    Args:
        r: input array to normalize

    Returns:
        normed: output array

    """
    normed = np.divide((r - np.mean(r)), np.std(r))
    return normed


class ReducedRankRegressor(object):
    """
    Reduced rank regressor
    Args:
        - X: n-by-d matrix of features.
        - Y: n-by-D matrix of targets
        - rank: either 'max' or int; rank constraint
        - reg: regularization parameter (optional)

    """
    def __init__(self, X, Y, rank=None, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))

        max_rank = np.min(X.shape + Y.shape)
        if rank is 'max':
            rank = max_rank
        else:
            if rank < 0 or rank > max_rank:
                raise ValueError(f'rank cannot be negative nor greater than minimum input dimension: {max_rank}')
        self.rank = rank

        if reg is None:
            reg = 0

        self.X = X
        self.Y = Y
        self.reg = reg

    def fit(self):
        """
        Fits reduced rank matrix

        A is shape [rank x Ydim2]
        B is shape [Xdim2 x rank]

        """
        X = self.X
        Y = self.Y
        rank = self.rank
        reg = self.reg
        reg_eye = reg * np.eye(np.size(X, 1))

        # X = np.vstack((X, reg_eye))
        # Y = np.vstack((Y, np.zeros((X.shape[1], Y.shape[1]))))

        CXX = np.dot(X.T, X) + reg_eye
        CXY = np.dot(X.T, Y)

        self.CXX = CXX
        self.CXY = CXY

        _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))

        self.A = V[0:rank, :].T
        self.B = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.A)).T

    def predict(self, X):
        """
        Predict Y from X based on fit A and B above
        """
        A = self.A
        B = self.B

        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        pred = np.dot(X, np.dot(A, B).T)
        # convert from matrix to array and transpose - easier for later
        pred = np.asarray(pred.T)
        return pred


def score_mse(y_true, y_pred):
    score = sklearn.metrics.mean_squared_error(y_true, y_pred)
    return score


def score_var_explained(y_true, y_pred, multioutput=None):
    score = sklearn.metrics.explained_variance_score(y_true, y_pred, multioutput=multioutput)
    return score


def get_int_ceil_sqrt(n):
    """
    Mostly for plotting lots of frames in a square
    Get ceiling of sqrt in integer form
    Args:
        n: input number

    Returns:
        sqrt: integer of ceiling of sqrt of input

    """
    sqrt = int(np.ceil(np.sqrt(n)))
    return sqrt


def bootstrap(x, n_reps):
    """
    Returns x, resampled with replacement n_reps times
    Args:
        x:
        n_reps:

    Returns:
        boot_n
    """
    boot_n = np.random.choice(x, (len(x), n_reps))
    return boot_n


def ci(a, bounds=95, axis=None):
    """
    Return a percentile range

    Args:
        a (np.ndarray): array to get CIs from
        bounds: percentile limits
        axis: if multidimensional a

    Returns:
        Percentile range

    """
    p = 50-bounds/2, 50+bounds/2
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

    beta_boots = bootstrap_sb(X, Y, func=reg_func, n_boot=1000).T
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


def bootstrap_sb(*args, **kwargs):
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
