import os, sys
import logging
import simplejson
from .utils import write_to_logger, maskImg


#######################################
# Analysis setup
#######################################
# TODO : make this return an image instead of just data? (use dataToImg)
# TODO : also need to make this return reoriented labels, verify it is working
def niPreproc(img, mask=None, sessions=None, logger=None, **kwargs):
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
    return NiftiMasker(mask_img=mask, sessions=sessions, **kwargs).fit_transform(img)


def opByLabel(d, l, op=None, logger=None):
    """
    apply operation to each unique value of the label and returns the data in its original order
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
    opD = np.concatenate([op.fit_transform(d[l.values == i]) for i in l.unique()], axis=0)
    lOrder = np.concatenate([l.index[l.values == i] for i in l.unique()], axis=0)
    return opD[lOrder]  # I really hope this works...


def sgfilter(logger=None, **sgparams):
    from scipy.signal import savgol_filter
    from sklearn.preprocessing import FunctionTransformer
    write_to_logger("Running SG filter", logger)
    return FunctionTransformer(savgol_filter, kw_args=sgparams)


#######################################
# Analysis
#######################################

def make_lss_reg():
    if isinstance(condition_attr, basestring):
        # must be a list/tuple/array for the logic below
        condition_attr = [condition_attr]

    e = copy.deepcopy(eorig)  # since we are modifying in place
    glm_condition_attrs = []
    for i, con in enumerate(condition_attr):
        glm_condition_attr = 'regressors_' + str(con)
        glm_condition_attrs.append(glm_condition_attr)
        for ei in e:
            if glm_condition_attr in ei:
                raise ValueError("Event %s already has %s defined.  Should not "
                                 "happen.  Choose another name if defined it"
                                 % (ei, glm_condition_attr))
            ei[glm_condition_attr] = \
                'glm_label_' + str(con) + '_' + '+'.join(str(ei[c]) for c in [con])
    pass
# TODO : Add design matrix
# new make_designmat from scratch
def make_designmat(tr, events, design_kwargs=None, regr_attrs=None):
    # make glm regressors for all attributes. so loop through condition_attr and add them all...
    import copy
    from nipy.modalities.fmri.design_matrix import make_dmtx
    import numpy as np
    # Decide/device condition attribute on which GLM will actually be done
    evvars = events2dict(e)
    add_paradigm_kwargs = {}
    if 'amplitude' in evvars:
        add_paradigm_kwargs['amplitude'] = evvars['amplitude']
    if design_kwargs is None:
        design_kwargs = {}
    if regr_attrs is not None:
        names = []
        regrs = []
        for attr in regr_attrs:
            regr = ds.sa[attr].value
            # add rudimentary dimension for easy hstacking later on
            if regr.ndim < 2:
                regr = regr[:, np.newaxis]
            if regr.shape[1] == 1:
                names.append(attr)
            else:
                #  add one per each column of the regressor
                for i in xrange(regr.shape[1]):
                    names.append("%s.%d" % (attr, i))
            regrs.append(regr)
        regrs = np.hstack(regrs)

        if 'add_regs' in design_kwargs:
            design_kwargs['add_regs'] = np.hstack((design_kwargs['add_regs'],
                                                   regrs))
        else:
            design_kwargs['add_regs'] = regrs
        if 'add_reg_names' in design_kwargs:
            design_kwargs['add_reg_names'].extend(names)
        else:
            design_kwargs['add_reg_names'] = names

    x = {}
    for ci, con in enumerate(condition_attr):
        # create paradigm
        if 'duration' in evvars:
            from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
            # NiPy considers everything with a duration as a block paradigm
            paradigm = BlockParadigm(
                con_id=evvars[glm_condition_attrs[ci]],
                onset=evvars['onset'],
                duration=evvars['duration'],
                **add_paradigm_kwargs)
        else:
            from nipy.modalities.fmri.experimental_paradigm \
                import EventRelatedParadigm
            paradigm = EventRelatedParadigm(
                con_id=evvars[glm_condition_attrs[ci]],
                onset=evvars['onset'],
                **add_paradigm_kwargs)
        x[con] = make_dmtx(ds.sa[time_attr].value, paradigm=paradigm, **design_kwargs)
        for i, reg in enumerate(x[con].names):
            ds.sa[reg] = x[con].matrix[:, i]
        if con in ds.sa.keys():
            ds.sa.pop(con)

        for reg in ds.sa.keys():
            if str(con)+'0' in reg:
                ds.sa['glm_label_probe'] = ds.sa.pop(reg)

    # concatenate X... add chunk regressors...
    # if 'chunks' in ds.sa.keys():
    #     for i in ds.sa['chunks'].unique:
    #         ds.sa['glm_label_chunks' + str(i)] = np.array(ds.sa['chunks'].value == i, dtype=np.int)
    return x, ds

# TODO : Add RSA functionality (needs a .fit)
def covdiag(x, df=None, shrinkage=None, logger=None):
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


def encoding(x_train, y_train, x_test, y_test, logger=None):
    """
    Encoding (prediction) analysis. Assumes data is pre-split into train and test
    For now, just uses Ridge regression. Later will use cross-validated Ridge.
    :param x_train: Input design matrix
    :param y_train: Fit data
    :param x_test: Test design matrix
    :param y_test: Test data
    :param logger: logging instance
    :return: Correlation scores, weights
    """
    # Later this will be tikhonov, for now use ridge
    import numpy as np
    import sklearn.linear_model as Tik
    # check that ndim == 2
    if y_train.ndim > 2:
        write_to_logger("Too many input dimensions", logger)
        return None
    clf = Tik.Ridge()
    pred = clf.fit(x_train, y_train).predict(x_test)
    corrs = np.array([np.corrcoef(y_test[:, i], pred[:, i])[0, 1] for i in range(pred.shape[1])])
    return corrs, clf.coef_


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
