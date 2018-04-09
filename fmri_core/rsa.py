from scipy.stats import wilcoxon, spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import LeaveOneOut
from .utils import write_to_logger
import numpy as np


# TODO : Add RSA functionality (needs a .fit)
def shrink_cov(x, df=None, shrinkage=None, logger=None):
    """
    Regularize estimate of covariance matrix according to optimal shrinkage
    method Ledoit& Wolf (2005), translated for covdiag.m (rsatoolbox- MATLAB)
    :param x: T obs by p random variables
    :param df: degrees of freedomc
    :param shrinkage: shrinkage factor
    :return: sigma, invertible covariance matrix estimator
             shrink: shrinkage factor
             sample: sample covariance (un-regularized)
    """
    # TODO : clean this code up
    t, n = x.shape
    if df is None:
        df = t-1
    X = x - x.mean(0)
    sampleCov = 1/df * np.dot(X.T, X)
    prior = np.diag(np.diag(sampleCov))  # diagonal of sampleCov
    if shrinkage is None:
        d = 1 / n * np.linalg.norm(sampleCov-prior, ord='fro')**2
        y = X**2
        r2 = 1 / n / df**2 * np.sum(np.dot(y.T, y)) - \
             1 / n / df * np.sum(sampleCov**2)
        shrink = max(0, min(1, r2 / d))
    else:
        shrink = 0

    sigma = shrink * prior + (1-shrink) * sampleCov
    return sigma, shrink, sampleCov


# DON'T NEED ANY OF THIS IF SIGMA and INV_SIGMA are known, can use pdist(..., VI=INV_SIGMA)
def noise_normalize_beta(betas, resids, df=None, shrinkage=None, logger=None):
    """
    "whiten" beta estimates according to residuals. Generally, because there are more features than
    samples regularization is used to find the covariance matrix (see covdiag above)
    :param betas: activity patterns ( nBetas (trials) by nVoxels, for example)
    :param resids: residuals (n voxels by nTRs)
    :param df: degrees of freedom. if none, defaults to nTRs- nBetas
    :param shrinkage: shrinkage (see covdiag)
    :param logger: logger instance
    :return: whitened betas (will have diagonal covariance matrix)
    """
    # find resids, WHAT ARE DEGREES OF FREEDOM
    # TODO : add other measures from noiseNormalizeBeta
    vox_cov_reg, shrink, vox_cov = shrink_cov(resids, df, shrinkage=shrinkage, logger=logger)
    whiten_filter = whitening_filter(vox_cov_reg)
    whitened_betas = np.dot(betas, whiten_filter)  # estimated true activity patterns
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
    Calculate upper and lower bounds on spearman correlations using double dipping
    (See Nili et al 2014)
    :param rdms: Stacked RDMs
    :return: (upper bound, lower bound)
    """
    # upper bound: each subject's correlation with mean
    mean_rdm = rdms.mean(0)
    upper = 1 - spearman_distance(np.vstack([mean_rdm, rdms]))[0][0, 1:]
    # lower bound: each subject's correlation with mean of other subjects
    loo = LeaveOneOut()
    lower = 1 - np.array([spearman_distance(np.vstack([rdms[test], rdms[train].mean(0)]))[0] for train, test in loo.split(rdms)])
    return np.vstack([upper, lower])


def rdm(X, square=False, logger=None, **pdistargs):
    """
    Calculate distance matrix
    :param X: data
    :param square: shape of output (square or vec)
    :param pdistargs: notably: include "metric"
    :return: pairwise distances between items in X
    """
    # add crossnobis estimator
    write_to_logger("Generating RDM", logger)
    p = None
    if "metric" in pdistargs:
        if pdistargs["metric"] == "spearman":
            r_raw, p = spearman_distance(X)
            r = squareform(r_raw, checks=False)
        else:
            r = pdist(X, **pdistargs)
    else:
        r = pdist(X, **pdistargs)

    if square:
        r = squareform(r)

    if p is not None:
        p = squareform(p, checks=False)
        return r, p
    else:
        return r


def spearman_distance(x):
    """
    Spearman distance of a matrix. To find distance between two entities, stack them and pass in
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
