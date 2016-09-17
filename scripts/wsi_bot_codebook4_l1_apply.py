#!/usr/bin/env python2

# Takes a level-0-max pooling coding and a corresponding B-o-F codebook,
# aconstructs the level-1 coding, applying the
# same B-o-F approach, but on larger neighborhoods and per subtype.

from __future__ import print_function

import numpy as np
import glob
import os.path
import argparse as opt

from util.storage import ModelPersistence
from segm.bot import soft_codes

__version__ = 0.1
__author__ = 'Vlad Popovici'


def image_l1_codes(img_path, encoding_type, subtype, vq_centers, p, beta=0.5):
    """
    Reads the l0 (max-pooling) features for a set of images and produces
    the final vector of codes corresponding to each image.

    Args:
        img_path: string
         Path to the l0 (max-pooling) files
        subtype: string
         a, ..., e
        encoding_type: string
        vq_centers: numpy.ndarray
        p: int
         number of neighbors in soft coding
        beta: float > 0
         smoothing coefficient

    Returns:

    """
    sample_paths = glob.glob(img_path + '/encoded_{}_l1/'.format(encoding_type) +
                             subtype + '/*.npz')

    if len(sample_paths) == 0:
        return None, None

    d = list()
    ids = list()
    for sp in sample_paths:
        sid = sp.split('/')[-1].split('_')[0]

        # read data l0
        Xl0_mp = np.load(sp)['Xl0_maxpool']  # W x H x K
        w, h, k = Xl0_mp.shape
        # soft-coding
        C = soft_codes(Xl0_mp.reshape((w*h,k)), vq_centers, p, beta)
        # max-pooling
        d.append( C.max(axis=0) )  # final codes for the image
        ids.append(sid)

    return np.vstack(d), ids


def main():
    p = opt.ArgumentParser(description="""
            Builds an L1 representation of a set of images based on a learned codebook.
            """)

    p.add_argument('data_folder', action='store',
                   help='path to a folder containing all the files for the subtypes')
    p.add_argument('encoding_type', action='store', choices=['sda', 'cnn'], default='sda',
                   help='specifies the encoding type for the data file (mainly for file name inference')
    p.add_argument('subtype', action='store', choices=['a', 'b', 'c', 'd', 'e', 'outlier'],
                   help='subtype')
    p.add_argument('codebook_file', action='store', help='codebook model file name')
    p.add_argument('out_file', action='store', help='filename for the resulting codes (all codes are saved in a single matrix)')
    p.add_argument('-p', action='store', type=int, default=4,
                   help='neighborhood size for soft-coding')
    p.add_argument('-b', '--beta', action='store', type=float, default=0.5,
                   help='smoothing term for soft-coding')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose?')

    args = p.parse_args()

    # read the model:
    with ModelPersistence(args.codebook_file, 'r', format='pickle') as d:
        vq = d['codebook']

    d, ids = image_l1_codes(args.data_folder, args.encoding_type, args.subtype, vq.cluster_centers_,
                            args.p, args.beta)

    np.savez_compressed(args.out_file, C=d)
    with open(os.path.splitext(args.out_file)[0]+'_ids.dat', 'w') as f:
        f.writelines([id+'\n' for id in ids])

    return 0


if __name__ == '__main__':
    main()
