import os, sys
import logging
import simplejson


#######################################
# Setup logging
#######################################
def setup_logger(*loc, fname='analysis'):
    """
    Setting up log file
    :param loc: location of file
    :param fname: log file name
    :return: logger instance
    """
    from datetime import datetime
    logging.basicConfig(filename=os.path.join(*loc, fname + '-' + sys.platform + '.log'),
                        datefmt='%m-%d %H:%M',
                        level=logging.DEBUG)
    logger = logging.getLogger(fname)
    logger.info('--------------------------------')
    logger.info("Session started at " + str(datetime.now()))
    return logger


def write_to_logger(msg, logger=None):
    """
    Writes to logger
    :param msg: Message to write
    :param logger: logger handle
    :return: None
    """
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return


#######################################
# File I/O
#######################################
# JSON I/O
def loadConfig(*path):
    """
    Load JSON config file
    :param path: path to json
    :return: dict with JSON contents
    """
    with open(os.path.join(*path), 'r') as f:
        d = simplejson.load(f)
    return d


def writeConfig(d, *path):
    """
    Write JSON file
    :param d: dict to write
    :param path: path to location
    :return: None
    """
    with open(os.path.join(*path), 'wt') as f:
        simplejson.dump(d, f, indent=4 * ' ')
    return


# TODO : change this to keyworded?
def formatBIDSName(*args):
    """
    write BIDS format name (may change this later)
    :param args: items to join
    :return: name
    """
    return ('_').join(args)


# Matlab I/O
def save_mat_data(*fn, **kwargs):
    """
    :param fn: path to file
    :param kwargs: any key value pairs-- keys will become fieldnames of the struct with value.
    :return: None: write a mat file
    """
    from scipy.io import savemat
    savemat(os.path.join(*fn), kwargs)
    return kwargs


def load_mat_data(*args):
    """
    Loads matlab file (just a wrapper)
    :param args: path to file
    :return: dict
    """
    from scipy.io import loadmat
    return loadmat(os.path.join(*args))


# Nilearn
def loadImg(*path, logger=None):
    """
    Simple wrapper for nilearn load_img to load NIFTI images
    :param p: path to subject directory
    :param s: subject
    :param fn: filename (plus leading directories)
    :param logger: logfile ID
    :return: Nifti1Image
    """
    bs = os.path.join(*path)
    from nilearn import image
    write_to_logger("Reading file from: " + bs, logger)
    return image.load_img(bs)


def loadLabels(*args, logger=None, **pdargs):
    """
    Simple wrapper using Pandas to load label files
    :param args: path to file directory
    :param logger: logfile ID
    :param pdargs: pandas read_csv args
    :return: pandas DataFrame with labels
    """
    import pandas as pd
    lp = os.path.join(*args)
    write_to_logger("Loading label file from: " + lp, logger)
    return pd.read_csv(lp, **pdargs)


#######################################
# Image processing
#######################################
def estimateMask(im, st='background'):
    """
    mask the wholehead image (if we don't have one). wrapper for NiLearn implementation
    :param im: image
    :param st: type of automatic extraction. epi for epi images, background for all other.
    :return: mask
    """
    from nilearn import masking
    if st == 'epi':
        mask = masking.compute_epi_mask(im)
    else:
        mask = masking.compute_background_mask(im)
    return mask


def maskImg(im, mask=None, logger=None):
    """
    Wrapper for apply_mask (adds logging)
    :param im: image
    :param mask: mask. if none, will try to estimate mask to generate 2d
    :param logger: logger ID
    :return: masked image
    """
    from nilearn import masking
    if isinstance(im, str):
        write_to_logger("Masking " + im, logger)
        return masking.apply_mask(im, mask)
    else:
        write_to_logger("Masking file")
        return masking._apply_mask_fmri(im, mask)


def dataToImg(d, img, copy_header=False):
    """
    Wrapper for new_image_like
    :param img: Image with header you want to add
    :param d: data
    :param copy_header: Boolean
    :return: Image file
    """
    from nilearn import image
    return image.new_img_like(image.mean_img(img), d, copy_header=copy_header)


def unmaskImg(d, mask):
    """
    Unmasks matrix d according to mask
    :param d: numpy array (2D)
    :param mask: mask
    :return: image file
    """
    from nilearn.masking import unmask
    return unmask(d, mask)


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


#######################################
# Analysis setup
#######################################
# TODO : make this return an image instead of just data? (use dataToImg)
# TODO : also need to make this return reoriented labels, verify it is working
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


def searchlight(x, y, m=None, cv=None, write=False, logger=None, **searchlight_args):
    """
    Wrapper to launch searchlight
    :param x: Data
    :param y: labels
    :param m: mask
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
    sl.fit(x, y)
    if write:
        return sl, image.new_img_like(image.mean_img(x), sl.scores_)
    else:
        return sl
