
# coding: utf-8

# In[1]:

import sys
sys.setrecursionlimit(10000)
sys.path.append('/Users/chief/Library/Python/2.7/lib/python/site-packages')
sys.path.append('/Users/chief/ownCloud/Work/DP/WSItk/WSItk')
sys.path.append('/Users/chief/local/lib/python2.7/site-packages')
sys.path.append('/Users/chief/ML/caffe/python')

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy as np

from IPython.display import Image as IPImage
from PIL import Image
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print 'Using cuda_convnet (faster)'
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print 'Using lasagne.layers (slower)'



# In[2]:

#f = np.load('/data/exp_crc/tumors_10x/div4/tiles64x64haem/0250-EF004_t1_tiles_w64_haem.npz')
#X = f['tiles'].astype(np.float32)
#f.close()
#X /= 255.0
#
#X = X.transpose()
import glob
all_x = []

for f in glob.glob('/Users/chief/ownCloud/Work/DP/exp_crc/tiles64x64haem/*.npz')[:20]:
    d = np.load(f)
    X = d['tiles'].astype(np.float32)
    d.close()
    X /= 255.0
    X = X.transpose()
    all_x.append(X)
X = np.vstack(all_x)
all_x = None

print 'X type and shape:', X.dtype, X.shape
print 'X.min():', X.min()
print 'X.max():', X.max()


# In[3]:

X_out = X.reshape((X.shape[0], -1))
print 'X_out:', X_out.dtype, X_out.shape


# In[4]:

conv_num_filters = 16
filter_size = 9
pool_size = 2
encode_size = 128
dense_mid_size = 256
pad_in = 'valid'
pad_out = 'full'
layers = [
    #(InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
    (InputLayer, {'shape': (None, 1, 64, 64)}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters/2, 'filter_size': filter_size, 'pad': pad_in}),
    (MaxPool2DLayerFast, {'pool_size': pool_size}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters/2, 'filter_size': filter_size, 'pad': pad_in}),
    (MaxPool2DLayerFast, {'pool_size': pool_size}),
    (ReshapeLayer, {'shape': (([0], -1))}),
    (DenseLayer, {'num_units': dense_mid_size}),
    (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
    (DenseLayer, {'num_units': dense_mid_size}),
    (DenseLayer, {'num_units': 288}),
    (ReshapeLayer, {'shape': (([0], conv_num_filters/2, 6, 6))}),
    (Upscale2DLayer, {'scale_factor': pool_size}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters/2, 'filter_size': filter_size, 'pad': pad_out}),
    (Upscale2DLayer, {'scale_factor': pool_size}),
    (Conv2DLayerSlow, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
    #(Conv2DLayerSlow, {'num_filters': X.shape[1], 'filter_size': filter_size, 'pad': pad_out}),
    (Conv2DLayerSlow, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
    (Conv2DLayerSlow, {'num_filters': 1, 'filter_size': filter_size, 'pad': pad_out}),
    (ReshapeLayer, {'shape': (([0], -1))}),
]


# In[5]:

ae = NeuralNet(
    layers=layers,
    max_epochs=100,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,

    regression=True,
    verbose=1
)
ae.initialize()
PrintLayerInfo()(ae)


# In[ ]:

ae.fit(X, X_out)


# In[ ]:

from nolearn.lasagne.visualize import plot_loss
plot_loss(ae)


# In[ ]:



