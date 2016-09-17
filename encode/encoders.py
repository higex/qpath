"""
ENCODERS: given a set of images, return their encoded representation.
"""
from __future__ import print_function, division

__author__ = 'Vlad Popovici'
__version__ = 0.1

from abc import ABCMeta, abstractmethod
from future.utils import bytes_to_native_str as nstr

import gzip
import pickle
import numpy as np

import theano.misc.pkl_utils

from lasagne.layers import get_output

## ENCODER: abstract class declaring the basic functionality
class Encoder:
    __metaclass__ = ABCMeta
    name = nstr(b'Encoder')

    @abstractmethod
    def encode(self, X):
        pass

    @abstractmethod
    def loadmodel(self, filename):
        pass


## SdAEncoder: stacked denoising autoencoder (see http://deeplearning.net/tutorial/SdA.html)
class SdAEncoder(Encoder):
    name = nstr(b'SdAEncoder')

    def __init__(self, filename=None):
        self.model = None
        self.input_dim = 0
        self.output_dim = 0

        if filename is not None:
            self.loadmodel(filename)

    def loadmodel(self, filename):
        with open(filename, 'rb') as f:
            self.model = theano.misc.pkl_utils.load(f)
        # get the input / output dimensions
        if self.model.n_layers == 0:
            raise RuntimeError('The encoder model does not contain any layer!')
        self.input_dim = self.model.params[0].eval().shape[0]
        self.output_dim = self.model.params[2*(self.model.n_layers-1)].eval().shape[1]

    def encode(self, X):
        assert(X.ndim == 2)
        assert(self.model is not None)

        n, dim = X.shape

        if dim != self.input_dim:
            raise RuntimeError('The given data dimension does not match the model')

        # Construct the encoding chain (stack)
        y = [X]   # to make the calls uniform
        for k in np.arange(self.model.n_layers):
            y.append(self.model.dA_layers[k].get_hidden_values(y[k]))

        # last element contains the final encoding:
        return y[self.model.n_layers].eval()


## CNNEncoder: convolutional neural network encoder
class CNNEncoder(Encoder):
    name = nstr(b'CNNEncoder')

    def __init__(self, filename=None):
        self.model = None
        self.input_dim = 0
        self.output_dim = 0

        if filename is not None:
            self.loadmodel(filename)

        return


    def loadmodel(self, filename):
        with gzip.open(filename, 'rb') as f:
            self.model = pickle.load(f)
        if len(self.model.layers) == 0:
            raise RuntimeError('The encoder model does not contain any layer!')

        # find the encoding layer, called "encode"
        for i, l in enumerate(self.model.get_all_layers()):
            if l.name == 'encode':
                self.encode_layer = l
                self.encode_layer_idx = i
        self.input_dim = self.model.layers[0][1]['shape'][1:] # get the tensor shape, skip the no. of samples (1st dim)
        self.output_dim = self.model.layers[self.encode_layer_idx][1]['num_units']

        return


    def encode(self, X):
        if X.shape[1:] != self.input_dim:
            raise RuntimeError('The given data dimension does not match the model')
        z = get_output(self.encode_layer, inputs=X)
        return z.eval()
