from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'Vlad Popovici'
__version__ = 0.25


from abc import ABCMeta, abstractmethod


class LocalDescriptor:
    """
    Base class for all local descriptors: given a patch of the image, compute
    some feature vector. Also, compute the distance between two feature vectors.

    The resulting features are represented as numpy.ndarray's (usually vectors) so
    they can easily used in other computations, that require input data as matrices
    or vectors.
    """
    __metaclass__= ABCMeta
    @abstractmethod
    def compute(self, image):
        pass

    @abstractmethod
    def dist(self, ft1, ft2, method=None):
        pass
# end class LocalDescriptor

