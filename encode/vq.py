# -*- coding: utf-8 -*-
"""
VQ - Vector Quantization methods.

Created on Tue Jun  7 10:23:06 2016

@author: chief
"""

from __future__ import (print_function)

__author__ = 'vlad'
__version__ = 0.3
__all__ = ['VQ', 'VQ_GMM']


from abc import ABCMeta, abstractmethod
from future.utils import bytes_to_native_str as nstr
import numpy as np
import scipy.sparse
from sklearn.mixture import GMM
from sklearn.decomposition import PCA

class VQ:
    """
    Base class for Vector Quantizers.
    """
    __metaclass__ = ABCMeta
    name = nstr(b'VQ')

    def __init__(self, codesize_):
        """
        Initialize the quantizer.

        :param codesize_: integer
            The codebook size of the quantizer.
        """
        self.codesize = codesize_
        self.isfit = False


    @abstractmethod
    def fit(self, X):
        """
        Train a vector quantizer based on a data matrix X [n x d].

        :param X: numpy.array
            data matrix (samples by rows)
        """
        pass


    @abstractmethod
    def recode(self, X, **kwargs):
        """
        Transform data into corresponding codes.

        :param X: numpy.array
            data matrix (samples by rows)

        :return: numpy.array
            a vector with codes
        """
        pass
# end VQ


class VQ_GMM(VQ):
    """
    Use a Gaussian Mixture Model for building the codebook.
    """
    name = nstr(b'VQ_GMM')

    def __init__(self, codesize_, pca_space_=None):
        """
        Initialize a VQ_GMM object.

        :param codesize_: integer
            The codebook size.
        :param pca_space_: integer
            If specified, PCA is applied to the data before learning the
            codebook.
        """

        self.codesize = codesize_
        self.isfit = False
        self.doPCA = (pca_space_ is not None)
        self.pca = PCA(n_components=int(pca_space_), whiten=True) if self.doPCA else None
        self.gmm = GMM(n_components=codesize_, covariance_type='diag')


    def fit(self, X):
        if self.doPCA:
            X = self.pca.fit_transform(X)
        self.gmm.fit(X)
        self.isfit = True


    def recode(self, X, **kwargs):
        """
        Recode a data matrix X in terms of a learned GMM dictionary. If a
        method is specified, different codings can be obtained.

        Call:
            recode(X [, method={'hard', 'hard_t', 'soft', 'soft_t', 'soft_n', 'full'}
            [, min_posterior] [, n]])

        :param X: numpy.array [n_samples x n_features]

        Optional arguments:
        method = {'hard', 'hard_t', 'soft', 'soft_t', 'soft_n', 'full'}:
            -'hard': code(x) is the index of the GMM component with the highest
            posterior probability: code(x) = argmax_k p_k(x) where p_k are
            the Gaussians of the mixture. The code is a scalar (integer), hence
            the returned value is a vector with n_samples values.
            -'hard_t': hard coding with thresholding: if the maximum posterior
            is less than a threshold assign no code. The code is a scalar
            (integer), hence the returned values is a vector of n_samples
            values. The <no code> value is coded as '-1'.
                Requires an additional argument: min_posterior.
            -'soft': code(x) = {k, p} where k is the index of the GMM component
            with the highest posterior probability and p is that posterior.
            The returned value is a sparse matrix [n_samples x codesize],
            with one value per row: the corresponding p is in the k-th column.
            -'soft_t': soft coding with thresholding: if none of the posterior
            probabilities is larger than the threshold, assign no code - the
            corresponding row will contain only 0s.
                Requires an additional argument: min_posterior.
            -'soft_n': soft coding with n-largest posteriors (n <= codesize)
                Requires an additional argument: n.
            -'full': return the full matrix of posteriors
        """
        if not self.isfit:
            raise RuntimeError('Fit the VQ_GMM before transform()')

        if self.doPCA:
            X = self.pca.transform(X)

        Z = self.gmm.predict_proba(X)   # posteriors from each GMM component
        i = np.argmax(Z, axis=1)        # indices for most probable components

        if 'method' not in kwargs:
            # simple,hard-assignement
            return i

        if kwargs['method'] == 'hard':
            return i

        if kwargs['method'] == 'full':
            return Z

        if kwargs['method'] == 'soft':
            Y = scipy.sparse.lil_matrix(Z.shape, dtype='single')
            for k in np.arange(Z.shape[0]):
                Y[k, i[k]] = Z[k, i[k]]
            return Y

        if kwargs['method'] == 'soft_t':
            if 'min_posterior' not in kwargs:
                raise RuntimeError('<min_posterior> not provided. Consider using "soft" method.')
            Y = scipy.sparse.lil_matrix(Z.shape, dtype='single')
            for k in np.arange(Z.shape[0]):
                if Z[k, i[k]] < kwargs['min_posterior']:
                    continue
                Y[k, i[k]] = Z[k, i[k]]
            return Y

        if kwargs['method'] == 'hard_t':
            if 'min_posterior' not in kwargs:
                raise RuntimeError('<min_posterior> not provided. Consider using "hard" method.')
            Y = np.zeros((X.shape[0],), dtype=int) - 1
            for k in np.arange(Z.shape[0]):
                if Z[k, i[k]] < kwargs['min_posterior']:
                    continue
                Y[k] = i[k]
            return Y

        if kwargs['method'] == 'soft_n':
            if 'n' not in kwargs:
                raise RuntimeError('<n> not provided. Consider using "soft" method.')
            if kwargs['n'] > self.codesize:
                raise RuntimeError('<n> is larger than codebook size')
            n = int(kwargs['n'])
            Y = scipy.sparse.lil_matrix(Z.shape, dtype='single')
            for k in np.arange(Z.shape[0]):
                # get the closest n components:
                i = np.argsort(Z[k,:])[::-1][:n]
                Y[k, i] = Z[k, i]
            return Y
# end VQ_GMM
