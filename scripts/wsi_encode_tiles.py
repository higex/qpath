"""
WSI_ENCODE_TILES: use an encoder to represent the local features.
"""

__author__ = 'Vlad Popovici'
__version__ = 0.1

import argparse as opt
import numpy as np

from encode import encoders
from encode.cnn import SaveModel, AdjustVariable  # needed for loading the model

def main():
    p = opt.ArgumentParser(description="""
            Encode the tiles given in a ndarray object, using the specified model.
            """)

    p.add_argument('tiles_file', action='store', help='file storing an numpy ndarray object.')
    p.add_argument('out_file', action='store', default='descriptors.dat',
               help='Name of the result file')

    p.add_argument('model_file', action='store', help='Model file')
    p.add_argument('-m', '--model_type', action='store', choices=['sda', 'cnn'],
                   default='sda', help='type of the model: stacked denoising autoencoder (sda) or convolutional NN (cnn)')

    args = p.parse_args()

    # create the encoder model
    if args.model_type == 'sda':
        encoder = encoders.SdAEncoder(args.model_file)
    elif args.model_type == 'cnn':
        encoder = encoders.CNNEncoder(args.model_file)
    else:
        raise RuntimeError('Unknown model')

    # read data
    X = np.float32(np.load(args.tiles_file)['X'])
    n = X.shape[0]

    if args.model_type == 'cnn':
        # print('Original X shape:', X.shape)
        if X.shape[1:] != encoder.input_dim:
            # try to fit:
            X = np.reshape(X, [n] + list(encoder.input_dim) )
            # print('Input data reshaped to ', X.shape)


    # encode:
    batch_size = 1024
    nb = np.floor(n / batch_size)
    rem = n - nb*batch_size
    if nb > 0:
        tmp = []
        for b in np.arange(nb):
            tmp.append(encoder.encode(X[b*batch_size:(b+1)*batch_size]))
        if rem > 0:
            tmp.append(encoder.encode(X[(b+1)*batch_size:]))
        Z = np.vstack(tmp)
    else:
        Z = encoder.encode(X)

    np.savez_compressed(args.out_file, Z=Z)

    return


if __name__ == "__main__":
    main()
