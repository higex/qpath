#!/usr/bin/env python2

# Compute the "level-0" bag of features encoding, using soft coding.

from __future__ import print_function

import numpy as np
import glob
import argparse as opt

from segm.bot import soft_codes
from util.storage import ModelPersistence

__version__ = 0.1
__author__ = 'Vlad Popovici'


def image_l0_codes(img_path, encoding_type, vq_centers, p,
                   std_flag=False, Zm=None, Zs=None,
                   w=32, beta=0.5):
    """
    Read and recode, one by one, a series of RGB images (tiles)
    from a repository structured as follows:
    img_path +
             |-<subtype>+
             |          |- <id>
             |          |...
             ...
             |- encoded_WxW_<encoding> +
             |                         |-<subtype>+
             |                         |          |- <id>_tiles_<encoding>.npz
             ...

    img_path/<subtype>/<sample_id>_tiles_<encoding_type>.npz

    where
        <img_path> points to the root of the encoded (CNN/SdA) tiles
        <subtype> can be 'a', 'b', 'c', 'd', 'e', 'outlier'
        <sample_id> is a 5-character string
        <encoding_type> is either 'cnn' (convolutional neural netw.) or
            'sda' (stacked denoising autoencoders)

    All files are read and stored in an array. A second list is returned with
    meta information about the data: sample id, subtype.
    """

    for subtype in ['a', 'b', 'c', 'd', 'e', 'outlier']:
        # get the IDs for the subtype:
        sample_paths = glob.glob(img_path + '/' + subtype + '/*')
        if len(sample_paths) == 0:
            continue

        for sp in sample_paths:
            sid = sp.split('/')[-1]

            # read coords for tiles:
            print('Coords:', sp + '/' + sid + '_coords.npz')
            coords = np.load(sp+'/'+sid+'_coords.npz')['coords']

            # read the codes by local descriptor
            print('Descriptors:', img_path + '/encoded_' + str(w) + 'x' + str(w) + '_' + encoding_type +
                  '/' + subtype + '/' + sid + '_tiles_' + encoding_type + '.npz')
            Z = np.load(img_path + '/encoded_' + str(w) + 'x' + str(w) + '_' + encoding_type +
                        '/' + subtype + '/' + sid + '_tiles_' + encoding_type +
                        ('.npz' if encoding_type == 'cnn' else '_4065000.npz'))['Z']

            # reconstruct the grid of tiles
            min_r, min_c = coords[:, 0].min(), coords[:, 2].min()
            max_r, max_c = coords[:, 1].max(), coords[:, 3].max()

            # for each tile, there is a k-dimensional code vector
            X = np.zeros( (int((max_r-min_r)/w), int((max_c-min_c)/w), vq_centers.shape[1]) )

            # soft-code the local descriptors
            if std_flag:
                # standardize variables if needed
                Z = (Z - Zm[:, None]) / Zs[:, None]
            # print('Z.shape:', Z.shape)
            C = soft_codes(Z, vq_centers, p, beta)

            # map the codes to spatial coordinates:
            for k in np.arange(C.shape[0]):
                i, j = int((coords[k, 0] - min_r)/w), int((coords[k, 2] - min_c)/w)
                if np.any(np.isinf(C[k, :])) or np.any(np.isnan(C[k, :])):
                    continue
                X[i, j, :] = C[k, :]

            # save the l0 soft-coding results
            print('Result:', img_path + '/encoded_' + encoding_type + '_l0/' + subtype + '/' + sid + '_l0.npz')
            print('\n\n')

            np.savez_compressed(img_path + '/encoded_' + encoding_type + '_l0/' +
                                '/' + subtype + '/' + sid + '_l0.npz',
                                Xl0=X,
                                coords=np.array([min_r, max_r, min_c, max_c]),
                                wsize=np.array([w]))
    return


def main():
    p = opt.ArgumentParser(description="""
            Builds an L0 representation based on a learned codebook.
            """)
    p.add_argument('data_folder', action='store',
                   help='path to a folder containing all the files for the subtypes')
    p.add_argument('encoding_type', action='store', choices=['sda', 'cnn'], default='sda',
                   help='specifies the encoding type for the data file (mainly for file name inference')
    p.add_argument('codebook_file', action='store', help='codebook model file name')
    p.add_argument('-p', action='store', type=int, default=4,
                   help='neighborhood size for soft-coding')
    p.add_argument('-b', '--beta', action='store', type=float, default=0.5,
                   help='smoothing term for soft-coding')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose?')

    args = p.parse_args()

    # read the model:
    with ModelPersistence(args.codebook_file, 'r', format='pickle') as d:
        vq = d['codebook']
        Zm = d['shift']
        Zs = d['scale']
        std_flag = d['standardize']

    image_l0_codes(args.data_folder, args.encoding_type, vq.cluster_centers_, args.p,
                   std_flag, Zm, Zs, beta=args.beta)

    # image_l0_codes(args.data_folder, args.encoding_type, None, args.p)

if __name__ == '__main__':
    main()