from scipy.stats import wilcoxon, spearmanr
from scipy.spatial.distance import pdist, squareform
from .utils import write_to_logger
import numpy as np


# TODO : Add RSA functionality (needs a .fit)
def covdiag(x, df=None, shrinkage=None, logger=None):
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
        shrink = shrinkage

    sigma = shrink * prior + (1-shrink) * sampleCov
    return sigma, shrink, sampleCov


def noise_normalize_beta(betas, resids, df, shrinkage=None, logger=None):
    # find resids
    # TODO : add other measures from noiseNormalizeBeta
    vox_cov_reg, shrink, vox_cov = covdiag(resids, df, shrinkage=shrinkage)
    V, L = np.linalg.eig(vox_cov_reg)
    sq = np.dot(V, V.T/np.sqrt(L))
    uhat = np.dot(betas, sq)  # estimated true activity patterns
    # resMS =


def indicator_matrix(logger=None):
    pass


def crossnobis(betas, resid, logger=None):
    # each iteration: find inverse of X and apply to left out iter

    pass


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
    if "metric" in pdistargs:
        if pdistargs["metric"] == "spearman":
            r = squareform(spearman_distance(X), checks=False)
        else:
            r = pdist(X, **pdistargs)
    else:
        r = pdist(X, **pdistargs)

    if square:
        r = squareform(r)
    return r


def spearman_distance(x):
    rho, _ = spearmanr(x, axis=1)
    return 1 - rho


def wilcoxon_onesided(x, **kwargs):
    _, p = wilcoxon(x, **kwargs)
    if np.median(x) > 0:
        res = p/2
    else:
        res = 1 - p/2
    return res