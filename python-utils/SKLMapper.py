# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""GLMMapper implementation based on the NiPy package."""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.support.copy import deepcopy
from mvpa2.datasets import Dataset
from mvpa2.mappers.base import Mapper
from mvpa2.base.param import Parameter


class SKLRegressionMapper(Mapper):
    """Mapper for Scikit-learn regression algorithms from sklearn.linear_model
    Essentially runs a regression by building the design matrix and running the algorithm.
    """
    add_constant = Parameter(False, constraints='bool', doc="""\
            If True, a constant will be added as last column in the
            design matrix.""")

    return_design = Parameter(False, constraints='bool', doc="""\
            If True, the mapped dataset will contain a sample attribute
            ``regressors`` with the design matrix columns.""")

    return_model = Parameter(False, constraints='bool', doc="""\
            If True, the mapped dataset will contain am attribute
            ``model`` for an instance of the fitted GLM. The type of
            this instance dependent on the actual implementation used.""")

    def __init__(self, regs=[], add_regs=None, clf=None, part_attr=None, **kwargs):
        """
        Parameters
        ----------
        regs : list
          Names of sample attributes to be extracted from an input dataset and
          used as design matrix columns.
        glmfit_kwargs : dict, optional
          Keyword arguments to be passed to GeneralLinearModel.fit().
          By default an AR1 model is used.
        """
        Mapper.__init__(self, auto_train=True, **kwargs)
        # GLMMapper.__init__(self, regs, **kwargs)
        self._clf = None
        self._pristine_clf = clf
        self.regs = list(regs)
        if add_regs is None:
            add_regs = tuple()
        self.add_regs = tuple(add_regs)
        self._part_attr = part_attr

    def _build_design(self, ds):
        X = None
        regsfromds = list(self.regs)
        reg_names=None
        if len(regsfromds):
            X = np.vstack([ds.sa[reg].value for reg in regsfromds]).T
            reg_names=regsfromds
        if len(self.add_regs):
            regs=[]
            if reg_names is None:
                reg_names = []
            for reg in self.add_regs:
                regs.append(reg[1])
                reg_names.append(reg[0])
            if X is None:
                X = np.vstack(regs).T
            else:
                X = np.vstack([X.T] + regs).T
        if self.params.add_constant:
            constant = np.ones(len(ds))
            if X is None:
                X = constant[None].T
            else:
                X = np.vstack((X.T, constant)).T
            if reg_names is None:
                reg_names = ['constant']
            else:
                reg_names.append('constant')
        if X is None:
            raise ValueError("no design specified")
        return reg_names, X

    def _fit_model(self, ds, X, reg_names):
        # a model of sklearn linear model
        if self._part_attr is None:
            glm = self._get_clf()
            glm.fit(X, ds.samples)
            out = Dataset(glm.coef_.T, sa={'regressor_names': reg_names})
        else:
            glm = []
            out = []
            for i in ds.sa[self._part_attr].unique:
                thism=self._get_clf()
                thism.fit(X[ds.chunks == i], ds[ds.chunks == i].samples)
                outi = Dataset(thism.coef_.T, sa={'regressor_names': reg_names,
                                                  self._part_attr: i*np.ones(X.shape[1])})
                out.append(outi)
            from mvpa2.base import dataset
            out = dataset.vstack(out)
        return glm, out

    def _get_y(self, ds):
        space = self.get_space()
        if space:
            y = ds.sa[space].value
        else:
            y = None
        return y

    def _get_clf(self):
        if self._clf is None:
            self._clf = deepcopy(self._pristine_clf)
        return self._clf

    def _forward_dataset(self, data):
        """Forward-map some data instead of implementing forward_dataset (which will call this on a copy).

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        reg_names, X = self._build_design(data)
        model, out = self._fit_model(data, X, reg_names)
        if self._part_attr is None:
            out.sa['regressor_names'] = [r.split('+')[0] for r in reg_names]
            out.sa['chunks'] = np.array([r.split('+')[-1] for r in reg_names], dtype=np.int)
        out.fa.update(data.fa)
        out.a.update(data.a) # this last one might be a bit to opportunistic
        # determine the output
        if self.params.return_design:
            if not len(out) == len(X.T):
                X = np.repeat(X, len(out)/len(X.T), 1)
                # raise ValueError("cannot include GLM regressors as sample "
                #                  "attributes (dataset probably contains "
                #                  "something other than parameter estimates")
            out.sa['regressors'] = X.T
        if self.params.return_model:
            out.a['model'] = model

        return out

    def _reverse_data(self, data):
        """Reverse-map some data.

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        raise NotImplementedError