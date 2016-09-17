__author__ = 'Vlad Popovici'
__version__ = 0.1


import sys

sys.setrecursionlimit(10000)

import glob
import pickle

import numpy as np
from numpy import float32

import theano
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow

from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class SaveModel(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        with open(self.name + '-{0:05d}'.format(epoch) + '.pkl', 'wb') as f:
            pickle.dump(nn, f, -1)



if __name__ == '__main__':
    Xl = list()
    data_path = '/Users/chief/Work/exp_crc/data'
    for s in ['a', 'b', 'c', 'd', 'e', 'outlier']:
        # for s in ['a']:
        for f in glob.glob(data_path + '/' + s + '/*_tiles.npz'):
            # load data from numpy compressed file:
            print(f)
            train_set_x = np.load(f)['X']
            # there might be some NaNs:
            if np.any(np.isnan(train_set_x)):
                i, _ = np.where(np.isnan(train_set_x))  # indexes of images with NaN
                train_set_x = np.delete(train_set_x, i, axis=0)
            idx = np.arange(train_set_x.shape[0])
            np.random.shuffle(idx)
            train_set_x = train_set_x[idx[:min(1000, len(idx))], :]  # keep at most 1000 randomly selected

            Xl.append(train_set_x.reshape((train_set_x.shape[0], 1, 32, 32)))

    X = np.vstack(Xl)
    i = np.arange(X.shape[0])
    np.random.shuffle(i)
    X = X[i, :, :, :]
    Xl = None

    print( 'X type: ', str(X.dtype), ' and shape: ', X.shape )
    print( 'X.min():', X.min() )
    print( 'X.max():', X.max() )

    X_out = X.reshape((X.shape[0], -1))
    print( 'X_out: ', str(X_out.dtype), X_out.shape )

    batch_size = 512
    conv_num_filters = 32
    filter_size = 5
    pool_size = 2
    encode_size = 128
    dense_mid_size = 256
    pad_in = 'valid'
    pad_out = 'full'
    layers = [
        (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
        (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
        (MaxPool2DLayerFast, {'pool_size': pool_size}),
        (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
        (MaxPool2DLayerFast, {'pool_size': pool_size}),
        (ReshapeLayer, {'shape': (([0], -1))}),
        (DenseLayer, {'num_units': dense_mid_size}),
        (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
        (DenseLayer, {'num_units': dense_mid_size}),
        (DenseLayer, {'num_units': 800}),
        (ReshapeLayer, {'shape': (([0], conv_num_filters, 5, 5))}),
        (Upscale2DLayer, {'scale_factor': pool_size}),
        (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
        (Upscale2DLayer, {'scale_factor': pool_size}),
        (Conv2DLayerSlow, {'num_filters': 1, 'filter_size': filter_size, 'pad': pad_out}),
        (ReshapeLayer, {'shape': (([0], -1))}),
    ]

    ae = NeuralNet(
        layers=layers,
        max_epochs=100,

        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.float32(0.03)),
        update_momentum=theano.shared(np.float32(0.9)),

        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            SaveModel('model'),
        ],

        batch_iterator_train=BatchIterator(batch_size),
        batch_iterator_test=BatchIterator(batch_size),

        regression=True,
        verbose=1
    )
    ae.initialize()
    PrintLayerInfo()(ae)

    ae.fit(X, X_out)
