# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# this will take in a dataset and design matrix and partitioner
# and run a bootstrap procedure, using each chunk as left-out for testing... (or training?)
# and output 3 datasets, the correlations, the alphas, and the beta estimates.
import numpy as np
from mvpa2.datasets import Dataset


def regperfcurve(r, a):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(len(r)+1, sharex=True)
    for i in np.arange(0, len(r)):
        ax[i].plot(a, np.mean(np.mean(r[i], axis=2), axis=1))
        ax[i].set_xscale('log')
    ax[-1].plot(a, np.mean(np.mean(np.dstack(r), axis=-1), axis=1))
    ax[-1].set_xscale('log')
    ax[0].set_title('Reg-Perf curve')
    plt.show()


def bootstrap_linear(ds, des, part_attr='chunks', mode='test'):
    if not part_attr in ds.sa.keys():
        print "invalid partitioner... exiting"
        return
    import sklearn.linear_model as lm

    data = ds.samples
    wts = []
    wtchunks = []
    corrs=[]
    ceiling=[]

    univ = lm.LinearRegression(fit_intercept=False)
    for c in (ds.sa[part_attr].unique):
        if mode == 'train':
            trainidx=ds.sa[part_attr].value==c
            testidx=ds.sa[part_attr].value!=c
        else:
            trainidx=ds.sa[part_attr].value!=c
            testidx=ds.sa[part_attr].value==c

        univ.fit(des.matrix[trainidx], data[trainidx])
        wt = univ.coef_
        nnpred = univ.predict(des.matrix[testidx])
        vcorr = np.nan_to_num(np.array([np.corrcoef(data[testidx, ii], nnpred[:, ii].ravel())[0, 1]
                                        for ii in range(data[testidx].shape[1])]))

        univ.fit(des.matrix[testidx], data[testidx])
        nnpred = univ.predict(des.matrix[testidx])
        ceilcorr = np.nan_to_num(np.array([np.corrcoef(data[testidx, ii], nnpred[:, ii].ravel())[0, 1]
                                        for ii in range(data[testidx].shape[1])]))
        wts.append(wt.T)
        wtchunks.append(np.repeat(c, des.matrix.shape[1]))
        corrs.append(vcorr)
        ceiling.append(ceilcorr)

    wtBrain = Dataset(np.vstack(wts), sa={part_attr: np.hstack(wtchunks).T}, fa=ds.fa, a=ds.a)
    corrBrain = Dataset(np.vstack(corrs), sa={part_attr: ds.sa[part_attr].unique}, fa=ds.fa, a=ds.a)
    ceilBrain = Dataset(np.vstack(ceiling), sa={part_attr: ds.sa[part_attr].unique}, fa=ds.fa, a=ds.a)
    return wtBrain, corrBrain, ceilBrain


def bootstrap_ridge(ds, des, chunklen=None, nchunks=None, mu0=None, cov0=None,
                    part_attr='chunks', mode='test', alphas=None, single_alpha=True, plot=False,
                    normalpha=False, nboots=100, corrmin=.2, singcutoff=1e-10, joined=None,
                    use_corr=True):
    # make sure ds is a dataset with chunks
    if not part_attr in ds.sa.keys():
        print "invalid partitioner... exiting"
        return

    if cov0 is not None:
        from pythonutils import ridge_with_prior as ridge
    else:
        from pythonutils import ridge as ridge

    if chunklen == None or nchunks == None:
            print 'no chunk length or number of chunks specified... exiting'
            return

    if alphas is None:
        alphas = np.logspace(0,3,20)
    else:
        alphas = alphas

    data = ds.samples
    wts = []
    wtchunks = []
    alpha = []
    ceiling=[]
    corrs=[]
    allRs = []
    for c in (ds.sa[part_attr].unique):
        if mode == 'train':
            trainidx=ds.sa[part_attr].value==c
            testidx=ds.sa[part_attr].value!=c
        else:
            trainidx=ds.sa[part_attr].value!=c
            testidx=ds.sa[part_attr].value==c

        # mode specifies whether each unique part_attr is used to train or test.
        if cov0 is not None:
            wt, vcorr, valphas, allRcorrs, valinds = ridge.bootstrap_ridge(Rstim=des.matrix[trainidx],
                                                                           Rresp=data[trainidx],
                                                                           Pstim=des.matrix[testidx],
                                                                           Presp=data[testidx],
                                                                           mu0=mu0,
                                                                           cov0=cov0,
                                                                           alphas=alphas,
                                                                           nboots=nboots,
                                                                           chunklen=chunklen,
                                                                           nchunks=nchunks,
                                                                           corrmin=corrmin,
                                                                           joined=joined,
                                                                           singcutoff=singcutoff,
                                                                           normalpha=normalpha,
                                                                           single_alpha=single_alpha,
                                                                           use_corr=use_corr)
            ceilcorrs = ridge.ridge_corr(Rstim=des.matrix[testidx], Rresp=data[testidx], Presp=data[testidx],
                                         Pstim=des.matrix[testidx], alphas=[np.mean(valphas[valphas > 0])],
                                         mu0=mu0, cov0=cov0,
                                         normalpha=False, corrmin=0.2, singcutoff=1e-10, use_corr=True)
        else:
            wt, vcorr, valphas, allRcorrs, valinds = ridge.bootstrap_ridge(Rstim=des.matrix[trainidx],
                                                                       Rresp=data[trainidx],
                                                                       Pstim=des.matrix[testidx],
                                                                       Presp=data[testidx],
                                                                       alphas=alphas,
                                                                       nboots=nboots,
                                                                       chunklen=chunklen,
                                                                       nchunks=nchunks,
                                                                       corrmin=corrmin,
                                                                       joined=joined,
                                                                       singcutoff=singcutoff,
                                                                       normalpha=normalpha,
                                                                       single_alpha=single_alpha,
                                                                       use_corr=use_corr)
            ceilcorrs = ridge.ridge_corr(Rstim=des.matrix[testidx], Rresp=data[testidx], Presp=data[testidx],
                                         Pstim=des.matrix[testidx], alphas=[np.mean(valphas[valphas > 0])],
                                         normalpha=False, corrmin=0.2, singcutoff=1e-10, use_corr=True)

        wts.append(wt)
        wtchunks.append(np.repeat(c, len(wt)))
        alpha.append(valphas)
        corrs.append(vcorr)
        ceiling.append(ceilcorrs)
        allRs.append(allRcorrs)

    if plot:
        if len(alphas) > 1:
            regperfcurve(allRs, alphas)

    wtBrain = Dataset(np.vstack(wts), sa={part_attr: np.hstack(wtchunks).T}, fa=ds.fa, a=ds.a)
    alphaBrain = Dataset(np.vstack(alpha), sa={part_attr: ds.sa[part_attr].unique}, fa=ds.fa, a=ds.a)
    corrBrain = Dataset(np.vstack(corrs), sa={part_attr: ds.sa[part_attr].unique}, fa=ds.fa, a=ds.a)
    ceilBrain = Dataset(np.vstack(ceiling), sa={part_attr: ds.sa[part_attr].unique}, fa=ds.fa, a=ds.a)
    return wtBrain, alphaBrain, corrBrain, ceilBrain
    # output as dataset...

