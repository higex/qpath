"""
Fisher Vector and VLAD encoders.
"""

from __future__ import print_function

__author__ = 'vlad'
__version__ = 0.3
__all__ = ['enc_fisher_vector', 'norm_fisher_vector']


from yael import ynumpy
import numpy as np
from encode.vq import VQ_GMM

def enc_fisher_vector(X, vq, what=['mean','sigma','prior']):
    """
    Compute the Fisher vector encoding of a set of descriptors corresponding
    to some image.

    :param X: numpy.array
        Descriptor matrix, descriptors by rows.
    :param vq: VQ_GMM object
        A train vector quantizer.
    :param what: list
        A list of parameters (from the GMM) to include in the Fisher vector.
        Possible values: 'prior', 'mean', 'sigma'. Any combination is allowed.

    :return: numpy.vector
        A Fisher vector summarizing the descriptors from the image. It contains,
        in order, the values corresponding to priors (k-1 values), means (k * d
        values) and sigma (k * d values), where k is the codebook size (number
        of components in the GMM) and d is the descriptor size (eventually after
        PCA transform, as specified in vq).
    """

    if len(what) == 0:
        raise RuntimeError('must specify at least one parameter for Fisher vector')

    # prepare a GMM structure for YAEL:
    g = (vq.gmm.weights_.astype(np.float32),
         vq.gmm.means_.astype(np.float32),
         vq.gmm.covars_.astype(np.float32))
    if vq.doPCA:
        X = vq.pca.transform(X)

    w = list()
    if 'mean' in what:
        w.append('mu')
    if 'sigma' in what:
        w.append('sigma')
    if 'prior' in what:
        w.append('w')

    v = ynumpy.fisher(g, X, include=w)

    return v
# end enc_fisher_vector


def norm_fisher_vector(v, method=['power', 'l2']):
    """
    Normalize a set of fisher vectors.

    :param v: numpy.array
        A matrix with Fisher vectors as rows (each row corresponding to an
        image).
    :param method: list
        A list of normalization methods. Choices: 'power', 'l2'.

    :return: numpy.array
        The set of normalized vectors (as a matrix).
    """

    if 'power' in method:
        v = np.sign(v) * np.abs(v)**0.5

    if 'l2' in method:
        nrm = np.sqrt(np.sum(v**2, axis=1))
        v /= nrm.reshape(-1, 1)

    v[np.isnan(v)] = 100000.0  # some large value

    return v
# end norm_fisher_vector
