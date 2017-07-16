# analysis scripts. THESE DON'T GET MODIFIED
import os, sys
import logging
# import sys, os
# if sys.platform == 'darwin':
#     sys.path.append(os.path.join("/Users", "njchiang", "GitHub", "task-fmri-utils"))
# else:
#     sys.path.append(os.path.join("D:\\", "GitHub", "task-fmri-utils"))


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


########################################
### preprocessing
def beta_extract(ds, events, c='trial_type', design_kwargs=None, return_model=True, logger=None):
    import mvpa2.datasets.eventrelated as er
    # {'add_regs': mc_params[sub], 'hrf_model': 'canonical'}
    if isinstance(c, basestring):
        c = [c]
    c.append('chunks')
    write_to_logger("beta-extracting... ", logger)
    evds = er.fit_event_hrf_model(ds, events, time_attr='time_coords',
                                  condition_attr=tuple(c),
                                  design_kwargs=design_kwargs,
                                  return_model=return_model)
    return evds


def error2acc(d):
    d.samples *= -1
    d.samples += 1
    return d


#######################################
### searchlight
def searchlight(paths, ds, r, clf=None, cv=None, writeopts=None, logger=None, **searchlight_args):
    write_to_logger("starting searchlight... ", logger)
    ## initialize classifier
    fds = ds.copy(deep=False, sa=['targets', 'chunks'], fa=['voxel_indices'], a=['mapper'])

    if cv is None:
        if clf is None:
            from mvpa2.clfs import svm
            clf = svm.LinearNuSVMC()
        from mvpa2.measures.base import CrossValidation
        from mvpa2.generators.partition import NFoldPartitioner
        cv = CrossValidation(clf, NFoldPartitioner())

    from mvpa2.measures.searchlight import sphere_searchlight
    cvsl = sphere_searchlight(cv, radius=r, **searchlight_args)
    import time
    wsc_start_time = time.time()
    res = cvsl(fds)
    write_to_logger(("done in " + str((time.time() - wsc_start_time)) + " seconds"), logger)
    res = error2acc(res)
    if writeopts:
        from mvpa2.datasets.mri import map2nifti
        from mvpa2.base import dataset
        map2nifti(fds, dataset.vstack(res)).\
            to_filename(os.path.join(
                        paths['root'], 'analysis', writeopts['outdir'],
                        writeopts['sub'] + '_' + writeopts['roi'] + '_' + writeopts['con'] + '_cvsl.nii.gz'))
        write_to_logger(("Writing to: " +
                         os.path.join(paths['root'],
                                      'analysis',
                                      writeopts['outdir'],
                                      writeopts['sub'] + '_' + writeopts['roi'] + '_' + writeopts['con'] + '_cvsl.nii.gz')), logger)
    return res


###############################
### encoding
def encoding(paths, ds, des, c, chunklen, nchunks,
             mus=None, covarmat=None, alphas=None, writeopts=None, logger=None, bsargs=None):
    """
    rds: input dataset
    events: events (list)
    c: contrast (or contrasts) of interest
    chunklen: length of a chunk
    nchunks: number of chunks
    mp: motion parameters (to regress)
    alphas: regularization parameters
    nboots: number of bootstraps
    mus: regularization towards
    covarmat: covariance matrix for regularization
    """
    import numpy as np
    if covarmat is not None:
        if mus is None:
            mus = np.zeros(covarmat.shape[1])

    if alphas is None:
        alphas = np.logspace(-1, 3, 50)
    from pythonutils import bootstrapridge as bsr
    if bsargs is None:
        bsargs = {'part_attr': 'chunks', 'mode': 'test', 'single_alpha': True, 'normalpha': False,
                  'nboots': 50, 'corrmin': .2, 'singcutoff': 1e-10, 'joined': None, 'plot': False, 'use_corr': True}
    wts, oalphas, res, ceil = bsr.bootstrap_ridge(ds, des, chunklen=chunklen, nchunks=nchunks,
                                                  cov0=covarmat, mu0=mus, alphas=alphas, **bsargs)
    if writeopts:
        from mvpa2.datasets.mri import map2nifti
        from mvpa2.base import dataset
        write_to_logger(("Writing results to: " +
                         os.path.join(paths['root'], 'analysis', writeopts['outdir'])), logger)
        map2nifti(ds, dataset.vstack(wts)).\
            to_filename(os.path.join(
            paths['root'], 'analysis', writeopts['outdir'],
            writeopts['sub'] + '_' + writeopts['roi'] + '_' + '+'.join(c) + '_wts.nii.gz'))
        map2nifti(ds, dataset.vstack(oalphas)). \
            to_filename(os.path.join(
            paths['root'], 'analysis', writeopts['outdir'],
            writeopts['sub'] + '_' + writeopts['roi'] + '_' + '+'.join(c) + '_alphas.nii.gz'))
        map2nifti(ds, dataset.vstack(res)). \
            to_filename(os.path.join(
            paths['root'], 'analysis', writeopts['outdir'],
            writeopts['sub'] + '_' + writeopts['roi'] + '_' + '+'.join(c) + '_res.nii.gz'))
        map2nifti(ds, dataset.vstack(ceil)). \
            to_filename(os.path.join(
            paths['root'], 'analysis', writeopts['outdir'],
            writeopts['sub'] + '_' + writeopts['roi'] + '_' + '+'.join(c) + '_ceil.nii.gz'))
    return wts, oalphas, res, ceil


######################################
### MVPA/RSA
def rsa(ds, order=False, rank=False, plot=False, plotargs=None):
    import numpy as np
    from mvpa2.measures import rsa
    dsm = rsa.PDist(pairwise_metric='cosine', square=True)
    if order:
        idx = np.argsort(ds.sa.targets)
        mds = ds[idx]
    else:
        mds = ds

    mtx = dsm(mds)
    f = None
    ax = None

    if not plotargs:
        plotargs={'title': None, 'vmin': 0, 'vmax': 1}
    if plot:
        if rank:
            f, ax = plot_mtx(ranktransform(mtx), mds.sa.targets, **plotargs)
        else:
            f, ax = plot_mtx((mtx), mds.sa.targets, **plotargs)
    return mtx, f, ax


def plot_mtx(mtx, labels, title=None, vmin=0, vmax=1):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    im = ax.imshow(mtx, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mtx)))
    ax.set_xticklabels(labels, rotation=-45)
    ax.set_yticks(range(len(mtx)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    # ax.set_clim(0,1)
    f.colorbar(im)
    return f, ax


def ranktransform(mat):
    # scales from 0-1
    import scipy.spatial.distance as ds
    import numpy as np
    vec = ds.squareform(mat)
    ranks = np.arange(vec.size).astype(float)
    sortedidx = np.argsort(vec)
    sortedranks = ranks[sortedidx]
    for i in np.unique(vec):
        sortedranks[vec==i] = np.mean(sortedranks[vec==i])

    sortedranks = np.divide(sortedranks, np.max(sortedranks))
    newmat = ds.squareform(sortedranks)
    return newmat

