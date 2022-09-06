#!/usr/bin/env python3

import datetime
import itertools
from IPython.display import clear_output, display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numbers
import pytoolsAL as ptAL
import sklearn
import warnings


def norm(r, c=0):
    """
    Normalizes array in (array - mean)/(std)
    Args:
        r: input array to normalize, shape (samples, features)
        c: small constant to add to std if desired

    Returns:
        normed: output array

    """
    normed = np.divide((r - np.mean(r, axis=0)), np.std(r, axis=0)+c)
    return normed


def find_nearest_value_index(array, value):
    """
    returns floor
    e.g. if array is 1, 2, 3 and value is 1.7,
    returns 0 (for value 1)
    """
    array = np.asarray(array)
    array_diff = array - value
    # cut off positive values (i.e. don't round up)
    array_diff = array_diff[array_diff <= 0]

    ix = (np.abs(array_diff)).argmin()
    return ix


def find_nearest_values_indices(array, values):
    """
    a version of the prev fx to do for array
    """
    array = np.asarray(array)
    ixs = []
    for v in values:
        array_diff = array - v
        # cut off positive values (i.e. don't round up)
        array_diff = array_diff[array_diff <= 0]
        ix = (np.abs(array_diff)).argmin()
        ixs.append(ix)
    return ixs


def smooth_lowess(y, x=None, span=10, robust=False, iter=None, axis=-1):
    """
    Stolen from Mark Histed (thanks!)
    Uses statsmodels.  As of around 2013, this is faster than Bio.statistics.
    Args:
        y: ndarray, can be 2d  (or maybe N-d - needs testing)
        x: None (default) or 1d ndarray, same length as specified axis of y.  If None, use 1:len(y)
        robust: bool, default False.  Whether to reweight to reduce influence of outliers, see docs
        span: number of pts, or percent of total number of points, as in matlab smooth.m
        axis: the axis to smooth over.
    Notes:
        Has MATLAB's nan behavior: drops nan's before smoothing, then puts the nans
        back in the same places in the smoothed array and returns.
    """

    import statsmodels.nonparametric.api

    if iter is not None:
        raise RuntimeError('iter no longer used: use robust param instead')

    y = np.asarray(y, dtype='f8')
    nPts = y.shape[axis]

    if x is None:
        x = np.arange(nPts, dtype='float')
    else:
        x = np.asarray(x, dtype='f8')
    if nPts < 2:
        raise ValueError('Input must have length > 1')
    if len(x) != np.shape(y)[axis]:
        raise ValueError('x has len %d, mismatch with y (ax %d), has len %d'
                         % (len(x), axis, np.shape(y)[axis]))

    assert (np.all(np.diff(x)>0)), 'x must be strictly increasing'  # or output will be all nans

    if span > (nPts-1):
        span = nPts-1
    if span < 1:
        frac = span  # percent
    else:
        frac = 1.0*span/nPts  # is number of points, convert to span

    if robust:
        iter = 5
    else:
        iter = 1

    delta = np.min((0.01 * np.ptp(x), span/5))  # hardcode.  first is suggestion in docs

    # iterate over the specified axis.  Note we need a func because lowess() returns a tuple
    def runonvec(y,x,frac,iter,delta):
        # remove nans manually
        ns = np.where(~np.isnan(y))
        ysm = statsmodels.nonparametric.api.lowess(y[ns],x[ns],
                                                    frac, it=iter, delta=delta, missing='raise')[:,1]
        yret = y.copy() # contains nans
        yret[ns] = ysm
        return yret
    smY = np.apply_along_axis(runonvec, axis, y, x, frac, iter, delta)

    return smY


class ReducedRankRegressor(object):
    """
    Reduced rank regressor
    Args:
        - X: n-by-d matrix of features.
        - Y: n-by-D matrix of targets
        - rank: either 'max' or int; rank constraint
        - reg: regularization parameter (optional). either single value OR
            list of values (size equal to X input)

    """
    def __init__(self, X, Y, rank=None, reg=None, reg_matrix = None,
                 verbose=0):
        self.verbose = verbose

        if self.verbose > 0:
            print('initializing regressor...', end=' ')
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))

        max_rank = np.min(X.shape + Y.shape)
        if rank == 'max':
            rank = max_rank
        if rank == None:
            rank = max_rank
            warnings.warn(f'no rank specified - defaulting to max rank {max_rank}')
        if rank < 0 or rank > max_rank:
                raise ValueError(f'rank cannot be negative nor greater than minimum input dimension: {max_rank}')

        self.rank = rank

        if reg is None:
            reg = 0

        self.X = X
        self.Y = Y
        self.reg = reg

        if reg_matrix is not None:
            self.reg_mat = reg_matrix
        else:
            self.reg_mat = None

        if self.verbose > 0:
            print('done.')

    def fit(self):
        """
        Fits reduced rank matrix

        A is shape [rank x Ydim2]
        B is shape [Xdim2 x rank]

        """
        if self.verbose > 0:
            print('fitting regressor...')
        X = self.X
        Y = self.Y

        # find which column in X has the intercept
        intercept_col = np.where(np.sum(X, axis=0) == X.shape[0])
        if len(intercept_col) > 1:
            raise ValueError('multiple intercept columns found ' \
                             '(somehow multiple cols sum to length)' \
                             'this is rare, idk fix it if it comes up')
        else:
            intercept_col = intercept_col[0]

        rank = self.rank
        reg = self.reg

        reg_eye = reg * np.eye(np.size(X, 1), dtype='uint8')

        # don't apply regularization to intercept col
        reg_eye[intercept_col, intercept_col] = 0
        # X = np.vstack((X, reg_eye))
        # Y = np.vstack((Y, np.zeros((X.shape[1], Y.shape[1]))))

        if self.verbose > 0:
            print('setting CXX and CXY...', end=' ')
        CXX = np.dot(X.T, X) + reg_eye
        CXY = np.dot(X.T, Y)

        self.CXX = CXX
        self.CXY = CXY

        if self.verbose > 0:
            print('computing SVD...', end = ' ')
        _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))

        if self.verbose > 0:
            print('done.')
            print('setting A and B...', end=' ')
        self.A = V[0:rank, :].T
        self.B = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.A)).T

        if self.verbose > 0:
            print('done.')

    def predict(self, X, rank=None):
        """
        Predict Y from X based on fit A and B above
        if rank is None, use full rank
        otherwise drop from A
        """
        if rank is None:
            A = self.A
            B = self.B

        else:
            A = self.A[:, :rank]
            B = self.B[:rank, :]

        if np.size(np.shape(X)) == 1:
            X = X.reshape(-1, 1)
        pred = np.dot(X, np.dot(A, B).T)
        # convert from matrix to array and transpose - easier for later
        pred = np.asarray(pred.T)
        return pred


def custom_reg_list(predictors, regs):
    reg_out = []
    for iP, pred in enumerate(predictors):
        ncols = pred.shape[1]
        reg_r = np.ones(ncols) * regs[iP]
        reg_out.extend(reg_r)
    return reg_out


def rrr_optimize(ranks, regs, x_train, y_train, x_test, y_test):
    """
    Attempts to optimize rrr params (regularization, rank)
    Args:
        - ranks: list of ranks to test
        - regs: list of regs to test
    """
    start_time = datetime.datetime.now()

    # make a list of the permutations
    param_combos = list(itertools.product(ranks, regs))
    # and array to store results
    param_results = np.zeros((len(param_combos)))

    # set up a figure to display results as they come in
    im = plt.imshow(np.zeros((len(ranks), len(regs))),
                    extent=[0, len(regs), ranks[-1], ranks[0]],
                    clim=[0, 1], aspect='auto')
    ax = plt.gca()
    ax = ptAL.plotting.apply_heatmap_defaults(ax)
    plt.ylabel('rank')
    plt.xlabel('regularization')

    reglabels = plt.xticks(rotation=45)[0][:-1]
    reglabels_log = regs[reglabels.astype('int')]
    reglabels_log_str = [f'{x:.0e}' for x in reglabels_log]

    ax.set_xticks(reglabels)
    ax.set_xticklabels(reglabels_log_str)

    cb = ptAL.plotting.add_colorbar(ax)
    cb.ax.set_ylabel('cross-validated variance explained')

    dsp2 = display(display_id=True)
    results = np.zeros((len(ranks), len(regs)))

    for iReg, reg in enumerate(regs):
        rrr = ReducedRankRegressor(x_train, y_train, rank='max', reg=reg)
        rrr.fit()

        for iRank, rank in enumerate(ranks):
            y_pred = rrr.predict(x_test, rank=rank)
            score = score_r2(y_test, y_pred.T,
                multioutput='variance_weighted')
            results[iRank, iReg] = score
            max_ix = np.unravel_index(np.argmax(results),
                (len(ranks), len(regs)))

            total_elapsed = (datetime.datetime.now() - start_time).total_seconds()
            dsp2.update(f'elapsed: {total_elapsed/60:.2f} min')

            ax.set_title(f'best: rank={ranks[max_ix[0]]}, reg={regs[max_ix[1]]:.2e}')
            im.set_data(results)
            im.set_clim(0, np.max(results)*1.2)
            cb.ax.set_ylabel('cross-validated $R^2$')
            clear_output(wait=True)

    return results, regs[max_ix[1]]


def rrr_optimize_rank(ranks, x_train, y_train, x_test, y_test, reg=None,
                      doPlot=True):
    """
    Evaluate many ranks with fixed regularization
    Args:
        - ranks: list of ranks to test
    """
    scores = []
    dsp = display(display_id=True)
    for iR, rank in enumerate(ranks):
        dsp.update(f'optimizing rank: starting {iR+1} of {len(ranks)}')
        rrr = ReducedRankRegressor(x_train, y_train, rank=rank, reg=reg)
        rrr.fit()

        y_pred = rrr.predict(x_test)
        score = score_r2(y_test, y_pred.T,
                         multioutput='variance_weighted')
        scores.append(score)
    scores = np.array(scores)
    return scores


def rrr_optimize_multipred(predictors_train, predictors_test,
                           ranks, regs, y_train, y_test):
    """
    """
    n_cols = 2
    n_rows = int(np.ceil(len(predictors_train)/n_cols))

    f = plt.figure(figsize=(8, 3*n_rows))
    gs = mpl.gridspec.GridSpec(n_rows, n_cols)
    optimal_regs = []
    for iP in range(len(predictors_train)):
        ax = plt.subplot(gs[iP])
        scores, opt_reg = rrr_optimize(ranks, regs, predictors_train[iP],
                                       y_train, predictors_test[iP], y_test)
        optimal_regs.append(opt_reg)
    return optimal_regs


def score_mse(y_true, y_pred):
    score = sklearn.metrics.mean_squared_error(y_true, y_pred)
    return score


def score_r2(y_true, y_pred, multioutput=None):
    """
    'raw_values', 'variance_weighted'
    """
    score = sklearn.metrics.r2_score(y_true, y_pred, multioutput=multioutput)
    return score


def score_var_explained(y_true, y_pred, multioutput=None):
    """
    'raw_values', 'variance_weighted'
    """
    score = sklearn.metrics.explained_variance_score(y_true, y_pred, multioutput=multioutput)
    warnings.warn('VARIANCE EXPLAINED ALARM, CHANGE TO R2')
    return score


def continuous_kfold(x, n_splits=10, drop_remainder=False):
    """
    defines indexes for cross validation
    for remainder: adds it to the final one

    Args:
        x: input vector OR int
        n_splits: number of splits to generate
        drop_remainder: whether to drop the last values if uneven division

    Returns:
        splits: n_splits-dimensional array of indexes
    """
    if type(x) is int:
        length = x
    else:
        length = len(x)
    len_splits = length // n_splits
    extra = length % n_splits

    splits = []
    for r in range(n_splits):
        # for the last split, add the remainder unless otherwise specified
        if r == n_splits-1:
            if drop_remainder:
                ixs = np.r_[r*len_splits:(r+1)*len_splits]
                splits.append(ixs)
            else:
                ixs = np.r_[r*len_splits:(r+1)*len_splits+extra]
                splits.append(ixs)

        # otherwise do a normal split
        else:
            ixs = np.r_[r*len_splits:(r+1)*len_splits]
            splits.append(ixs)

    return splits


def make_design_matrix(x, shifts):
    """
    Make design matrix of shape (len(x), len(shifts))
    Each row is rolled according to its value in shifts
    """
    design = np.zeros((len(x), len(shifts)))

    for i, n in enumerate(shifts):
        design[:, i] = np.roll(x, n)

    return design


def iterate_stack_timeshifts(matrix, shifts, n_comps):
    """
    iterate through 2nd dim of matrix up to n_comps
    for each, make timeshift matrix
    then stack
    """
    stack = []
    for n in np.arange(n_comps):
        ts = make_design_matrix(matrix[:, n], shifts=shifts)
        stack.append(ts)
    timeshifts = np.hstack(stack)
    return timeshifts


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


def rolling_average(x, w):
    """
    Args:
        x: input array
        w: averaging window size
    """
    return np.convolve(x, np.ones(w), 'valid') / w


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


def describe(x, axis=None):
    """
    print some stats of current thing
    """
    print(f'mean: {np.mean(x, axis=axis)}')
    print(f'median: {np.median(x, axis=axis)}')
    print(f'min: {np.min(x, axis=axis)}')
    print(f'max: {np.max(x, axis=axis)}')

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


def poly_fit(x, y, order):
    """
    Taken from seaborn regression.py
    Regression using numpy polyfit for higher-order trends.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    grid = np.linspace(x_min, x_max, 100)

    def reg_func(_x, _y):
        return np.polyval(np.polyfit(_x, _y, order), grid)

    yhat = reg_func(x, y)

    yhat_boots = bootstrap_sb(x, y, func=reg_func, n_boot=1000)
    return yhat, yhat_boots


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
