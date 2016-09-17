import numpy as np
import scipy.io as sio

def import_from_mat(filename):
    d = sio.loadmat(filename)
    X = d['X']
    n = d['metadata'].shape[1]
    ids = [d['metadata'][0,k][0,0][0][0] for k in np.arange(n)]
    ntc = [d['metadata'][0,k][0,0][1][0] for k in np.arange(n)]

    meta = dict({'id': ids, 'tc': ntc})

    return X, meta
