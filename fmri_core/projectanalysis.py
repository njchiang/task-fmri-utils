import os, sys
import logging
import simplejson

#######################################
# Setup logging
def setup_logger(loc, fname='analysis'):
    from datetime import datetime
    logging.basicConfig(filename=os.path.join(loc, fname + '-' + sys.platform + '.log'),
                        datefmt='%m-%d %H:%M',
                        level=logging.DEBUG)
    logger = logging.getLogger(fname)
    logger.info('--------------------------------')
    logger.info("Session started at " + str(datetime.now()))
    return logger


def write_to_logger(msg, logger=None):
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
    with open(os.path.join(*path), 'r') as f:
        d = simplejson.load(f)
    return d


def writeConfig(d, *path):
    with open(os.path.join(*path), 'wt') as f:
        simplejson.dump(d, f, indent=4 * ' ')
    return


def formatBIDSName(*args):
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
    :param p: path to subject directory
    :param s: subject
    :param l: filename (plus leading directories)
    :param logger: logfile ID
    :param **pdargs: pandas read_csv args
    :return: pandas DataFrame with labels
    """
    import pandas as pd
    lp = os.path.join(*args)
    write_to_logger("Loading label file from: " + lp, logger)
    return pd.read_csv(lp, **pdargs)


#######################################
# Image processing
#######################################
def maskImg(im, mask, logger=None):
    """
    Wrapper for apply_mask (adds logging)
    :param im: image
    :param mask: mask
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


#######################################
# Analysis setup
#######################################
# TODO : set up cross validation (stratified)
def setup_cv():
    import sklearn.model_selection as ms
    # return cv
    pass


def roi(x, y, clf, m=None, cv=None, **kwargs):
    """
    Cross validation on a masked roi. Need to decide if this function does preprocessing or not
    (probably should pipeline in)
    :param x: input image
    :param y: labels
    :param clf: classifier or pipeline
    :param m: mask (optional)
    :param cv: cross validator
    :param kwargs: other model_selection arguments
    :return:
    """
    import sklearn.model_selection as ms
    if m is not None:
        X = maskImg(x, m)
    else:
        X = x
    return ms.cross_val_score(clf, X, y, cv, **kwargs)


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


# TODO : Add SGF filter

    def sgfilter():
        pass

# TODO : Add RSA functionality (needs a .fit)
    def rsa():
        pass

# TODO : Add Encoding
    def encoding():
        pass

