# savgol filter draft
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Savitsky-Golay filtering... in progress. need to add per chunk functionality for it to be actually useful.
 running the filter is actually just training... applying it is the second part. NOTE: AXIS = 0.
 plan: """

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals

if externals.exists('scipy', raise_=True):
    from scipy.signal import resample
    from scipy.signal import savgol_filter

# from mvpa2.base import warning
from mvpa2.base.param import Parameter
from mvpa2.base.dochelpers import _str, borrowkwargs, _repr_attrs
from mvpa2.mappers.base import accepts_dataset_as_samples, Mapper
from mvpa2.datasets import Dataset



class SavGolFilterMapper(Mapper):
    """Mapper using SavGol filters for data transformation.

    This mapper is able to perform any SavGol-based low-pass, high-pass, or
    band-pass frequency filtering. This is a front-end for SciPy's SavGol(),
    hence its usage looks almost exactly identical, and any of SciPy's IIR
    filters can be used with this mapper:

    >>> from scipy import signal
    >>> b, a = signal.butter(8, 0.125)
    >>> mapper = SavGolFilterMapper(window_length, polyorder)

    """


    # scipy.signal.savgol_filter(x, window_length, polyorder,
                           # deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    # x : array_like
    #     The data to be filtered.  If `x` is not a single or double precision
    #     floating point array, it will be converted to type `numpy.float64`
    #     before filtering.
    # window_length : int
    #     The length of the filter window (i.e. the number of coefficients).
    #     `window_length` must be a positive odd integer.
    # polyorder : int
    #     The order of the polynomial used to fit the samples.
    #     `polyorder` must be less than `window_length`.

    axis = Parameter(0,
            doc="""The axis of `x` to which the filter is applied. By default
            the filter is applied to all features along the samples axis""")

    deriv = Parameter(0,
            doc="""The order of the derivative to compute.  This must be a
            nonnegative integer.  The default is 0, which means to filter
            the data without differentiating.""")

    delta = Parameter(1.0, constraints='float',
            doc="""The spacing of the samples to which the filter will be applied.
            This is only used if deriv > 0.  Default is 1.0.""")

    mode = Parameter('interp',
            doc="""Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`.  See the Notes for more details on 'mirror',
            'constant', 'wrap', and 'nearest'.
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.""")

    cval = Parameter(0.0,
            doc="""Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.""")


    def __init__(self, window_length=79, polyorder=3, chunks_attr='chunks',
                **kwargs):
        """
        All constructor parameters are analogs of filtfilt() or are passed
        on to the Mapper base class. Default values are based on Cukur 2013 paper for easy justification

        Parameters
        ----------
        b : (N,) array_like
            The numerator coefficient vector of the filter.
        a : (N,) array_like
            The denominator coefficient vector of the filter.  If a[0]
            is not 1, then both a and b are normalized by a[0].
        """
        self.__sg_w = window_length
        self.__sg_p = polyorder
        self.__chunks_attr = chunks_attr

        # result of train
        self._regs = None

        self._secret_inplace_filter = False

        Mapper.__init__(self, auto_train=True, **kwargs)

    def __repr__(self):
        s = super(SavGolFilterMapper, self).__repr__()
        return s.replace("(",
                         "(window_length=%i, polyorder=%i, chunks_attr=%s, "
                          % (self.__sg_w,
                             self.params.__sg_p,
                             repr(self.__chunks_attr)),
                         1)

    def __str__(self):
        return _str(self)

# takes in data and filters it.
    def _sgfilter(self, data):
        params = self.params
        mapped = savgol_filter(data.samples,
                               self.__sg_w,
                               self.__sg_p,
                               axis=params.axis,
                               deriv=params.deriv,
                               delta=params.delta,
                               mode=params.mode,
                               cval=params.cval)
        return mapped

    def _train(self, ds):
        chunks_attr = self.__chunks_attr
        params = self.params
        if chunks_attr is None:
            reg = self._sgfilter(ds.copy(deep=False))
        else:
            uchunks = ds.sa[chunks_attr].unique
            reg = []
            for n, chunk in enumerate(uchunks):
                cinds = ds.sa[chunks_attr].value == chunk
                thisreg = self._sgfilter(ds[cinds].copy(deep=False))
                reg.append(thisreg)
            reg = np.vstack(reg)

        # combine the regs (time x reg)
        # self._regs = np.hstack(reg)
        self._regs = np.array(reg)

    def _forward_dataset(self, ds):
        # auto-train the mapper if not yet done
        if self._regs is None:
            self.train(ds)
        if self._secret_inplace_filter:
            mds = ds
        else:
            # shallow copy to put the new stuff in
            mds = ds.copy()
        regs = self._regs
        # regression for each feature, but need to get the corresponding feature for each voxel...
        """ is this crap even necessary if i have the filtered data? do you just subtract?
        or do you need to regress and tak residuals..
        for i in np.arange(ds.shape[1]):
            print "voxel " + str(i)
            print regs[:,[i,-1]].shape
            print ds.samples[:,i].shape
            fit = np.linalg.lstsq(regs[:,[i,-1]], ds.samples[:,i])
            # actually we are only interested in the solution
            # res[0] is (nregr x nfeatures)
            y = fit[0]
            # remove all and keep only the residuals
            if self._secret_inplace_filter:
                # if we are in evil mode do evil

                # cast the data to float, since in-place operations below do not
                # upcast!
                if np.issubdtype(mds.samples.dtype, np.integer):
                    mds.samples[:,i] = mds.samples[:,i].astype('float')
                mds.samples[:,i] -= np.dot(regs[:,[i,-1]], y)
            else:
                # important to assign to ensure COW behavior
                mds.samples[:,i] = ds.samples[:,i] - np.dot(regs[:,[i,-1]], y)
        """
        if self._secret_inplace_filter:
            if np.issubdtype(mds.samples.dtype, np.integer):
                mds.samples = mds.samples.astype('float')
            mds.samples -= regs
        else:
            mds.samples = ds.samples - regs
        return mds

    def _forward_data(self, data):
        raise RuntimeError("%s cannot map plain data."
                           % self.__class__.__name__)

def sg_filter(ds, *args, **kwargs):
    """IIR-based frequency filtering.

    Parameters
    ----------
    ds : Dataset
    **kwargs
      For all other arguments, please see the documentation of
      IIRFilterMapper.
    """
    dm = SavGolFilterMapper(*args, **kwargs)
    dm._secret_inplace_filter = True
    # map
    mapped = dm.forward(ds)
    # and append the mapper to the dataset
    # mapped._append_mapper(dm)
    #  for now it's not appended... but just keep that in mind.