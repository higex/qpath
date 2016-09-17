#!/usr/bin/env python2

# Takes a level-0 coding and constructs the level-1 coding, applying the
# same B-o-F approach, but on larger neighborhoods and per subtype.

from __future__ import print_function

import numpy as np
import numpy.linalg
import glob
import argparse as opt
from sklearn.cluster import MiniBatchKMeans

from util.storage import ModelPersistence
from util.explore import sliding_window

__version__ = 0.1
__author__ = 'Vlad Popovici'


def load_image_l0_codes(img_path, encoding_type, subtype, w):
    """
    Loads the l0-coding for a series of images, apply max-pooling for
    constructing a local descriptor and saves the results.

    Args:
        img_path: string
            the images are loaded from a path of the form
            <img_path>/encoded_<encoding_type>_l0/<subtype>/

        encoding_type: string
            see img_path

        subtype: string
            see img_path

        w: int
            local neighborhood size (w x w), in coordinates of the l0-image

    Returns:

    """
    # get the IDs for the subtype:
    sample_paths = glob.glob(img_path +
                             '/encoded_{}_l0/'.format(encoding_type) +
                             subtype + '/*.npz')
    # print(img_path + '/encoded_{}_l0/'.format(encoding_type) + subtype + '/*.npz')
    if len(sample_paths) == 0:
        return None

    xlist = list()
    for sp in sample_paths:
        sid = sp.split('/')[-1].split('_')[0]
        # print(sp)
        # print(sid)

        # read data l0
        Xl0 = np.load(sp)['Xl0']  # W x H x K
        coords_l0 = np.load(sp)['coords']
        wsize_l0 = np.load(sp)['wsize']

        r, c, k = Xl0.shape

        # work on l0-image
        st_r = int(np.floor((r % w) / 2))
        st_c = int(np.floor((c % w) / 2))
        nr = int(np.floor(r / w))
        nc = int(np.floor(c / w))
        Xl0_mp = np.zeros((nr, nc, k))
        it = sliding_window( (r, c), (w, w), start=(st_r, st_c), step=(w, w) )

        for r0, r1, c0, c1 in it:
            # max-pooling over W x W
            Xl0_mp[int((r0-st_r)/w), int((c0-st_c)/w), :] = Xl0[r0:r1, c0:c1, :].reshape((w*w, k)).max(axis=0)

        # save l1-image
        dst_file = img_path + '/encoded_{}_l1/'.format(encoding_type) + subtype + '/' + sid + '_l1.npz'
        # print(dst_file)
        np.savez_compressed(dst_file,
                            Xl0_maxpool=Xl0_mp,
                            coords_l0=coords_l0,
                            coords_l1=np.array([st_r, st_r+nr*w, st_c, st_c+nc*w]),
                            wsize_l0=wsize_l0,
                            wsize_l1=np.array([w]))

        xlist.append(Xl0_mp.reshape((nr*nc, k)))

    return xlist


def main():
    p = opt.ArgumentParser(description="""
            Builds an L0-maxpool representation based on a l0 local features and learns
            an l1 codebook.
            """)

    p.add_argument('data_folder', action='store',
                   help='path to a folder containing all the files for the subtypes')
    p.add_argument('encoding_type', action='store', choices=['sda', 'cnn'],
                   help='specifies the encoding type for the data file (mainly for file name inference')
    p.add_argument('subtype', action='store', choices=['a','b','c','d','e','outlier'],
                   help='subtype')
    p.add_argument('codebook_file', action='store', help='resulting l1-codebook model file name')
    p.add_argument('codebook_size', action='store', help='codebook size', type=int)
    p.add_argument('-v', '--verbose', action='store_true', help='verbose?')

    args = p.parse_args()

    x = load_image_l0_codes(args.data_folder, args.encoding_type, args.subtype, 16)
    Z = np.vstack(x)

    rng = np.random.RandomState(123456)
    vq = MiniBatchKMeans(n_clusters=args.codebook_size, random_state=rng, batch_size=500,
                         compute_labels=True, verbose=False)  # vector quantizer
    vq.fit(Z)

    # compute the average distance and std.dev. of the points in each cluster:
    avg_dist = np.zeros(args.codebook_size)
    sd_dist = np.zeros(args.codebook_size)

    for k in range(0, args.codebook_size):
        d = numpy.linalg.norm(Z[vq.labels_ == k, :] - vq.cluster_centers_[k, :], axis=1)
        avg_dist[k] = d.mean()
        sd_dist[k] = d.std()

    with ModelPersistence(args.codebook_file, 'c', format='pickle') as d:
        d['codebook'] = vq
        d['avg_dist_to_centroid'] = avg_dist
        d['stddev_dist_to_centroid'] = sd_dist

    return 0


if __name__ == '__main__':
    main()