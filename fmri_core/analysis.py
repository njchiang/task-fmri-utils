from .utils import write_to_logger, maskImg


#######################################
# Analysis setup
#######################################
# TODO : make this return an image instead of just data? (use dataToImg)
# TODO : also need to make this return reoriented labels, verify it is working
def nipreproc(img, mask=None, sessions=None, logger=None, **kwargs):
    """
    applies nilearn's NiftiMasker to data
    :param img: image to be processed
    :param mask: mask (optional)
    :param sessions: chunks (optional)
    :param logger: logger instance
    :param kwargs: kwargs for NiftiMasker from NiLearn
    :return: preprocessed image (result of fit_transform() on img)
    """
    from nilearn.input_data import NiftiMasker
    write_to_logger("Running NiftiMasker...", logger)
    return NiftiMasker(mask_img=mask,
                       sessions=sessions,
                       **kwargs).fit_transform(img)


def op_by_label(d, l, op=None, logger=None):
    """
    apply operation to each unique value of the label and
    returns the data in its original order
    :param d: data (2D numpy array)
    :param l: label to operate on
    :param op: operation to carry (scikit learn)
    :return: processed data
    """
    import numpy as np
    write_to_logger("applying operation by label", logger)
    if op is None:
        from sklearn.preprocessing import StandardScaler
        op = StandardScaler()
    opD = np.concatenate([op.fit_transform(d[l.values == i])
                         for i in l.unique()], axis=0)
    lOrder = np.concatenate([l.index[l.values == i]
                            for i in l.unique()], axis=0)
    return opD[lOrder]  # I really hope this works...


def sgfilter(logger=None, **sgparams):
    from scipy.signal import savgol_filter
    from sklearn.preprocessing import FunctionTransformer
    write_to_logger("Creating SG filter", logger)
    return FunctionTransformer(savgol_filter, kw_args=sgparams)


#######################################
# Analysis
#######################################
def make_designmat(frametimes, cond_ids, onsets, durations, amplitudes=None,
                   design_kwargs=None, constant=False, logger=None):
    """
    Creates design matrix from TSV columns
    :param frametimes: time index (in s) of each TR
    :param cond_ids: condition ids. each unique string will become a regressor
    :param onsets: condition onsets
    :param durations: durations of trials
    :param amplitudes: amplitude of trials (default None)
    :param design_kwargs: additional arguments(motion parameters, HRF, etc)
    :param logger: logger instance
    :return: design matrix instance
    """
    if design_kwargs is None:
        design_kwargs = {}
    if "drift_model" not in design_kwargs.keys():
        design_kwargs["drift_model"] = "blank"

    from nipy.modalities.fmri.design_matrix import make_dmtx
    from nipy.modalities.fmri.experimental_paradigm import BlockParadigm

    write_to_logger("Creating design matrix...", logger)
    paradigm = BlockParadigm(con_id=cond_ids,
                             onset=onsets,
                             duration=durations,
                             amplitude=amplitudes)
    dm = make_dmtx(frametimes, paradigm, **design_kwargs)
    if constant is False:
        import numpy as np
        dm.matrix = np.delete(dm.matrix, dm.names.index("constant"), axis=1)
        dm.names = dm.names.remove("constant")
    return dm


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
    import numpy as np
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


def noiseNormalizeBeta(betas, resids, df, shrinkage=None, logger=None):
    # find resids
    # TODO : add other measures from noiseNormalizeBeta
    import numpy as np
    vox_cov_reg, shrink, vox_cov = covdiag(resids, df, shrinkage=shrinkage)
    V, L = np.linalg.eig(vox_cov_reg)
    sq = np.dot(V, V.T/np.sqrt(L))
    uhat = np.dot(betas, sq)  # estimated true activity patterns
    # resMS =


def indicatorMatrix(logger=None):
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
    from scipy.spatial.distance import pdist, squareform
    # add crossnobis estimator
    if square:
        r = squareform(pdist(X, **pdistargs))
    else:
        r = pdist(X, **pdistargs)
    return r


def predict(clf, x, y, logger=None):
    """
    Encoding prediction. Assumes data is pre-split into train and test
    For now, just uses Ridge regression. Later will use cross-validated Ridge.
    :param clf: trained classifier
    :param x: Test design matrix
    :param y: Test data
    :param logger: logging instance
    :return: Correlation scores, weights
    """
    import numpy as np
    pred = clf.predict(x)
    if y.ndim < 2:
        y = y[:, np.newaxis]
    if pred.ndim < 2:
        pred = pred[:, np.newaxis]
    write_to_logger("Predicting", logger)
    corrs = np.array([np.corrcoef(y[:, i], pred[:, i])[0, 1]
                     for i in range(pred.shape[1])])
    return corrs


def roi(x, y, clf, m=None, cv=None, **roiargs):
    """
    Cross validation on a masked roi. Need to decide if this
    function does preprocessing or not
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


def searchlight(x, y, m=None, groups=None, cv=None,
                write=False, logger=None, **searchlight_args):
    """
    Wrapper to launch searchlight
    :param x: Data
    :param y: labels
    :param m: mask
    :param groups: group labels
    :param cv: cross validator
    :param write: if image for writing is desired or not
    :param logger:
    :param searchlight_args:(default) process_mask_img(None),
                            radius(2mm), estimator(svc),
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
