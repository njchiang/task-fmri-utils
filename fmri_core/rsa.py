from scipy.stats import wilcoxon  # , spearmanr <- this has a bug
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import LeaveOneOut
from scipy.stats import mstats_basic
from scipy.stats import rankdata, distributions
import numpy as np
import warnings
from .utils import write_to_logger, mask_img, data_to_img
# from . import searchlight

import numpy as np


def add_diag(x, v):
    """
    memory efficient way to add a diagonal matrix with a square matrix
    :param x: square matrix to be appended
    :param v: vector of diagonal elements
    :return:
    """
    for i in range(len(v)):
        x[i, i] += v[i]
    return x


def sumproduct(x1, x2):
    """
    memory efficient way to multiply two matrices and add the result
    :param x1:
    :param x2:
    :return:
    """
    if x1.shape[1] != x2.shape[0]:
        return None
    try:
        total = 0
        for r in range(x1.shape[0]):
            total += np.dot(x1[r], x2).sum()
    # still use dot product, just with stacked vectors...
    except MemoryError:
        total = 0
        # slow
        for i in range(x1.shape[0]):
            for j in range(x2.shape[1]):
                total += sum(x1[i] * x2[:, j])

    return total


def ssq(x):
    total = 0
    for i in range(x.shape[0]):
        total += (x[i]**2).sum()
    return total


# TODO : Add RSA functionality (needs a .fit)
def shrink_cov(x, df=None, shrinkage=None, logger=None):
    """
    Regularize estimate of covariance matrix according to optimal shrinkage
    method Ledoit& Wolf (2005), translated for covdiag.m (rsatoolbox- MATLAB)
    :param x: T obs by p random variables
    :param df: degrees of freedom
    :param shrinkage: shrinkage factor
    :return: sigma, invertible covariance matrix estimator
             shrink: shrinkage factor
             sample: sample covariance (un-regularized)
    """
    # TODO : clean this code up
    t, n = x.shape
    if df is None:
        df = t-1
    x = x - x.mean(0)
    sampleCov = 1/df * np.dot(x.T, x)
    prior_diag = np.diag(sampleCov)  # diagonal of sampleCov
    if shrinkage is None:
        try:
            d = 1 / n * np.linalg.norm(sampleCov-np.diag(prior_diag),
                                       ord='fro')**2
            r2 = 1 / n / df**2 * np.sum(np.dot(x.T**2, x)) - \
                1 / n / df * np.sum(sampleCov**2)
        except MemoryError:
            write_to_logger("Low memory option", logger)
            d = 1 / n * np.linalg.norm(add_diag(sampleCov, -prior_diag),
                                       ord='fro')**2
            write_to_logger("d calculated")
            r2 = 1 / n / df**2 * sumproduct(x.T**2, x)

            write_to_logger("r2 part 1")

            r2 -= 1 / n / df * ssq(sampleCov)
            write_to_logger("r2 part 2")
        shrink = max(0, min(1, r2 / d))
    else:
        shrink = 0

    # sigma = shrink * prior + (1-shrink) * sampleCov
    # sigma = add_diag((1-shrink) * sampleCov, shrink*prior_diag)
    # return sigma, shrink, sampleCov
    return add_diag((1-shrink)*sampleCov, shrink*prior_diag), \
        shrink, prior_diag


# DON'T NEED ANY OF THIS IF SIGMA and INV_SIGMA are known,
# can use pdist(..., VI=INV_SIGMA)
def noise_normalize_beta(betas, resids, df=None, shrinkage=None, logger=None):
    """
    "whiten" beta estimates according to residuals.
    Generally, because there are more features than
    samples regularization is used to find the covariance matrix (see covdiag)
    :param betas: activity patterns ( nBetas (trials) by nVoxels, for example)
    :param resids: residuals (n voxels by nTRs)
    :param df: degrees of freedom. if none, defaults to nTRs- nBetas
    :param shrinkage: shrinkage (see covdiag)
    :param logger: logger instance
    :return: whitened betas (will have diagonal covariance matrix)
    """
    # find resids, WHAT ARE DEGREES OF FREEDOM
    # TODO : add other measures from noiseNormalizeBeta
    vox_cov_reg, shrink, _ = shrink_cov(resids, df,
                                        shrinkage=shrinkage, logger=logger)
    whiten_filter = whitening_filter(vox_cov_reg)
    whitened_betas = np.dot(betas, whiten_filter)
    # estimated true activity patterns
    return whitened_betas


def whitening_filter(x):
    """
    calculates inverse square root of a square matrix using SVD
    :param x: covariance matrix
    :return: inverse square root of a square matrix
    """
    _, s, vt = np.linalg.svd(x)
    return np.dot(np.diag(1/np.sqrt(s)), vt).T


def spearman_noise_bounds(rdms):
    """
    Calculate upper and lower bounds on spearman correlations
    using double dipping
    (See Nili et al 2014)
    :param rdms: Stacked RDMs
    :return: (upper bound, lower bound)
    """
    # upper bound: each subject's correlation with mean
    mean_rdm = rdms.mean(0)
    upper = 1 - spearman_distance(np.vstack([mean_rdm, rdms]))[0][0, 1:]
    # lower bound: each subject's correlation with mean of other subjects
    loo = LeaveOneOut()
    lower = 1 - \
        np.array([spearman_distance(np.vstack([rdms[test],
                                               rdms[train].mean(0)]))[0]
                  for train, test in loo.split(rdms)
                  ])
    return np.vstack([upper, lower])


def rdm(X, square=False, logger=None, return_p=False, **pdistargs):
    """
    Calculate distance matrix
    :param X: data
    :param square: shape of output (square or vec)
    :param pdistargs: notably: include "metric"
    :return: pairwise distances between items in X
    """
    # add crossnobis estimator
    if logger is not None:
        write_to_logger("Generating RDM", logger)
    p = None
    if "metric" in pdistargs:
        if pdistargs["metric"] == "spearman":
            r, p = spearman_distance(X)
            if r.shape is not ():
                r = squareform(r, checks=False)
                p = squareform(r, checks=False)
        else:
            r = pdist(X, **pdistargs)
    else:
        r = pdist(X, **pdistargs)

    if square:
        r = squareform(r, checks=False)
        if p is not None:
            p = squareform(p, checks=False)

    if p is not None and return_p is True:
        return r, p
    else:
        return r


def spearman_distance(x):
    """
    Spearman distance of a matrix. To find distance between two entities,
    stack them and pass in
    :param x: entries
    :return: Spearman distance (1 - rho)
    """
    rho, p = spearmanr(x, axis=1)
    return 1 - rho, p


def wilcoxon_onesided(x, **kwargs):
    """
    Runs one sided nonparametrix t test
    :param x: data
    :param kwargs: arguments for wilcoxon
    :return: p-values
    """
    _, p = wilcoxon(x, **kwargs)
    if np.median(x) > 0:
        res = p/2
    else:
        res = 1 - p/2
    return res


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


def _chk2_asarray(a, b, axis):
    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        outaxis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)
    if b.ndim == 0:
        b = np.atleast_1d(b)

    return a, b, outaxis


def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)


def spearmanr(a, b=None, axis=0, nan_policy='propagate'):
    """
    Calculate a Spearman rank-order correlation coefficient and the p-value
    to test for non-correlation.
    The Spearman correlation is a nonparametric measure of the monotonicity
    of the relationship between two datasets. Unlike the Pearson correlation,
    the Spearman correlation does not assume that both datasets are normally
    distributed. Like other correlation coefficients, this one varies
    between -1 and +1 with 0 implying no correlation. Correlations of -1 or
    +1 imply an exact monotonic relationship. Positive correlations imply that
    as x increases, so does y. Negative correlations imply that as x
    increases, y decreases.
    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Spearman correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.
    Parameters
    ----------
    a, b : 1D or 2D array_like, b is optional
        One or two 1-D or 2-D arrays containing multiple variables and
        observations. When these are 1-D, each represents a vector of
        observations of a single variable. For the behavior in the 2-D case,
        see under ``axis``, below.
        Both arrays need to have the same length in the ``axis`` dimension.
    axis : int or None, optional
        If axis=0 (default), then each column represents a variable, with
        observations in the rows. If axis=1, the relationship is transposed:
        each row represents a variable, while the columns contain observations.
        If axis=None, then both arrays will be raveled.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.
    Returns
    -------
    correlation : float or ndarray (2-D square)
        Spearman correlation matrix or correlation coefficient (if only 2
        variables are given as parameters. Correlation matrix is square with
        length equal to total number of variables (columns or rows) in ``a``
        and ``b`` combined.
    pvalue : float
        The two-sided p-value for a hypothesis test whose null hypothesis is
        that two sets of data are uncorrelated, has same dimension as rho.
    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.
       Section  14.7
    Examples
    --------
    >>> from scipy import stats
    >>> stats.spearmanr([1,2,3,4,5], [5,6,7,8,7])
    (0.82078268166812329, 0.088587005313543798)
    >>> np.random.seed(1234321)
    >>> x2n = np.random.randn(100, 2)
    >>> y2n = np.random.randn(100, 2)
    >>> stats.spearmanr(x2n)
    (0.059969996999699973, 0.55338590803773591)
    >>> stats.spearmanr(x2n[:,0], x2n[:,1])
    (0.059969996999699973, 0.55338590803773591)
    >>> rho, pval = stats.spearmanr(x2n, y2n)
    >>> rho
    array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
           [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
           [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
           [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
    >>> pval
    array([[ 0.        ,  0.55338591,  0.06435364,  0.53617935],
           [ 0.55338591,  0.        ,  0.27592895,  0.80234077],
           [ 0.06435364,  0.27592895,  0.        ,  0.73039992],
           [ 0.53617935,  0.80234077,  0.73039992,  0.        ]])
    >>> rho, pval = stats.spearmanr(x2n.T, y2n.T, axis=1)
    >>> rho
    array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
           [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
           [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
           [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
    >>> stats.spearmanr(x2n, y2n, axis=None)
    (0.10816770419260482, 0.1273562188027364)
    >>> stats.spearmanr(x2n.ravel(), y2n.ravel())
    (0.10816770419260482, 0.1273562188027364)
    >>> xint = np.random.randint(10, size=(100, 2))
    >>> stats.spearmanr(xint)
    (0.052760927029710199, 0.60213045837062351)
    """
    a, axisout = _chk_asarray(a, axis)
    x = a
    if a.ndim > 2:
        raise ValueError("spearmanr only handles 1-D or 2-D arrays")

    if b is None:
        if a.ndim < 2:
            raise ValueError("`spearmanr` needs at least 2 variables to compare")
    else:
        # Concatenate a and b, so that we now only have to handle the case
        # of a 2-D `a`.
        b, _ = _chk_asarray(b, axis)
        if axisout == 0:
            a = np.column_stack((a, b))
        else:
            a = np.row_stack((a, b))

    n_vars = a.shape[1 - axisout]
    n_obs = a.shape[axisout]
    if n_obs <= 1:
        # Handle empty arrays or single observations.
        return (np.nan, np.nan)

    a_contains_nan, nan_policy = _contains_nan(a, nan_policy)
    variable_has_nan = np.zeros(n_vars, dtype=bool)
    if a_contains_nan:
        if nan_policy == 'omit':
            return mstats_basic.spearmanr(a, axis=axis, nan_policy=nan_policy)
        elif nan_policy == 'propagate':
            if a.ndim == 1 or n_vars <= 2:
                return (np.nan, np.nan)
            else:
                # Keep track of variables with NaNs, set the outputs to NaN
                # only for those variables
                # update J.C.
                # variable_has_nan = np.isnan(a).sum(axis=axisout)
                variable_has_nan = np.isnan(a).sum(axis=axisout).astype(np.bool)

    a_ranked = np.apply_along_axis(rankdata, axisout, a)
    rs = np.corrcoef(a_ranked, rowvar=axisout)
    dof = n_obs - 2  # degrees of freedom

    # rs can have elements equal to 1, so avoid zero division warnings
    olderr = np.seterr(divide='ignore')
    try:
        # clip the small negative values possibly caused by rounding
        # errors before taking the square root
        t = rs * np.sqrt((dof/((rs+1.0)*(1.0-rs))).clip(0))
    finally:
        np.seterr(**olderr)

    prob = 2 * distributions.t.sf(np.abs(t), dof)
#     pdb.set_trace()
    # For backwards compatibility, return scalars when comparing 2 columns
    if rs.shape == (2, 2):
        return (rs[1, 0], prob[1, 0])
    else:
        rs[variable_has_nan, :] = np.nan
        rs[:, variable_has_nan] = np.nan
        return (rs, prob)