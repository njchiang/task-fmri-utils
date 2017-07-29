import os, sys
import logging
import simplejson
from .utils import write_to_logger, maskImg


#######################################
# Analysis setup
#######################################
# TODO : make this return an image instead of just data? (use dataToImg)
# TODO : also need to make this return reoriented labels, verify it is working
def niPreproc(img, mask=None, sessions=None, **kwargs):
    """
    applies nilearn's NiftiMasker to data
    :param img: image to be processed
    :param mask: mask (optional)
    :param sessions: chunks (optional)
    :param kwargs: kwargs for NiftiMasker from NiLearn
    :return: preprocessed image (result of fit_transform() on img)
    """
    from nilearn.input_data import NiftiMasker
    return NiftiMasker(mask_img=mask, sessions=sessions, **kwargs).fit_transform(img)


def opByLabel(d, l, op=None):
    """
    apply operation to each unique value of the label and returns the data in its original order
    :param d: data (2D numpy array)
    :param l: label to operate on
    :param op: operation to carry (scikit learn)
    :return: processed data
    """
    import numpy as np
    if op is None:
        from sklearn.preprocessing import StandardScaler
        op = StandardScaler()
    opD = np.concatenate([op.fit_transform(d[l.values == i]) for i in l.unique()], axis=0)
    lOrder = np.concatenate([l.index[l.values == i] for i in l.unique()], axis=0)
    return opD[lOrder]  # I really hope this works...


def sgfilter(**sgparams):
    from scipy.signal import savgol_filter
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(savgol_filter, kw_args=sgparams)


#######################################
# Analysis
#######################################
# TODO : Add RSA functionality (needs a .fit)
def covdiag(x, df=None, shrinkage=None):
    """
    Regularize estimate of covariance matrix according to optimal shrinkage method
    Ledoit& Wolf (2005), translated for covdiag.m (rsatoolbox- MATLAB)
    :param x: T obs by p random variables
    :param df: degrees of freedom
    :param shrinkage: shrinkage factor
    :return: sigma, invertible covariance matrix estimator
             shrink: shrinkage factor
             sample: sample covariance (un-regularized)
    """
    # TODO : clean this code up
    import numpy as np
    t, n = x.shape
    if df is None:
        df = t-1
    X = x - x.mean(0)
    sampleCov = 1/df * np.dot(X.T, X)
    prior = np.diag(np.diag(sampleCov)) # diagonal of sampleCov
    if shrinkage is None:
        d = 1 / n * np.linalg.norm(sampleCov-prior, ord='fro')**2
        y = X**2
        r2=1 / n / df**2 * np.sum(np.dot(y.T,y)) - 1 / n / df * np.sum(sampleCov**2)
        shrink = max(0,min(1,r2/d));
    else:
        shrink = shrinkage

    sigma = shrink * prior + (1-shrink) * sampleCov
    return sigma, shrink, sampleCov


def noiseNormalizeBeta(betas, resids, df, shrinkage=None):
    # find resids
    # TODO : add other measures from noiseNormalizeBeta
    import numpy as np
    vox_cov_reg, shrink, vox_cov = covdiag(resids, df, shrinkage=shrinkage)
    V, L = np.linalg.eig(vox_cov_reg)
    sq = np.dot(V, V.T/np.sqrt(L))
    uhat = np.dot(betas, sq)  # estimated true activity patterns
    # resMS =


def indicatorMatrix():
    pass


def crossnobis(betas, resid, ):
    # each iteration: find inverse of X and apply to left out iter

    pass


def rdm(X, square=False, **pdistargs):
    """
    Calculate distance matrix
    :param X: data
    :param square: shape of output (square or vec)
    :param pdistargs: notably: include "metric"
    :return:
    """
    from scipy.spatial.distance import pdist, squareform
    # add crossnobis estimator
    if square:
        r = squareform(pdist(X, **pdistargs))
    else:
        r = pdist(X, **pdistargs)
    return r


# TODO : Add Encoding
def encoding():
    pass


def roi(x, y, clf, m=None, cv=None, **roiargs):
    """
    Cross validation on a masked roi. Need to decide if this function does preprocessing or not
    (probably should pipeline in)
    pa.roi(x, y, clf, m, cv, groups=labels['chunks'])
    :param x: input image
    :param y: labels
    :param clf: classifier or pipeline
    :param m: mask (optional)
    :param cv: cross validator
    :param roiargs: other model_selection arguments, especially groups
    :return: CV results
    """
    import sklearn.model_selection as ms
    if m is not None:
        X = maskImg(x, m)
    else:
        X = x
    return ms.cross_val_score(estimator=clf, X=X, y=y, cv=cv, **roiargs)


def searchlight(x, y, m=None, groups=None, cv=None, write=False, logger=None, **searchlight_args):
    """
    Wrapper to launch searchlight
    :param x: Data
    :param y: labels
    :param m: mask
    :param groups: group labels
    :param cv: cross validator
    :param write: if image for writing is desired or not
    :param logger:
    :param searchlight_args:(default) process_mask_img(None), radius(2mm), estimator(svc),
                            n_jobs(-1), scoring(none), cv(3fold), verbose(0)
    :return: trained SL object and SL results
    """
    write_to_logger("starting searchlight... ", logger)
    from nilearn import image, decoding, masking
    if m is None:
        m = masking.compute_epi_mask(x)
    write_to_logger("searchlight params: " + str(searchlight_args))
    sl = decoding.SearchLight(mask_img=m, cv=cv, **searchlight_args)
    sl.fit(x, y, groups)
    if write:
        return sl, image.new_img_like(image.mean_img(x), sl.scores_)
    else:
        return sl
