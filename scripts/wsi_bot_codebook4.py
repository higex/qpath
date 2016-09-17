#!/usr/bin/env python2

#
# wsi_bot_codebook4
#
# Version 4 of codebook construction:
#
# -adapted to molecular subtype data structures

from __future__ import (absolute_import, division, print_function, unicode_literals)

import glob
import argparse as opt
import numpy as np
import numpy.linalg

from sklearn.cluster import MiniBatchKMeans
from util.storage import ModelPersistence

__version__ = 0.1
__author__ = 'Vlad Popovici'


def load_image_codes(img_path, encoding_type, max_per_subtype=100000):
    """
    Load a series of RGB images (tiles) from a path of the form:

    img_path/<subtype>/<sample_id>_tiles_<encoding_type>.npz

    where
        <subtype> can be 'a', 'b', 'c', 'd', 'e', 'outlier'
        <sample_id> is a 5-character string
        <encoding_type> is either 'cnn' (convolutional neural netw.) or
            'sda' (stacked denoising autoencoders)

    All files are read and stored in an array. A second list is returned with
    meta information about the data: sample id, subtype.
    """

    all_Z = list()
    meta_tmp = {'subtype': list(), 'id': list(), 'index': list()}
    for subtype in ['a', 'b', 'c', 'd', 'e', 'outlier']:
        sample_paths = glob.glob(img_path + '/' + subtype + '/*_tiles_' + encoding_type + '*.npz')
        if len(sample_paths) == 0:
            continue

        Zlist = list()
        id_tmp = list()
        idx_tmp = list()
        # one file per sample
        for sp in sample_paths:
            sid = sp.split('/')[-1].split('_')[0]

            #print('Processing:' + sp)
            Ztmp = np.load(sp)['Z']
            Zlist.append( Ztmp )  # the coding
            id_tmp.extend([sid]*Ztmp.shape[0])
            idx_tmp.extend(np.arange(Ztmp.shape[0]))

        Z = np.vstack(Zlist)
        if np.any(np.isnan(Z)):
            i, _ = np.where(np.isnan(Z))
            i = np.unique(i)
            #print('NaN rows to delete: ', i)
            Z = np.delete(Z, i, axis=0)
            id_tmp = [id_tmp[j] for j in np.arange(len(id_tmp)) if j not in i]
            idx_tmp = [idx_tmp[j] for j in np.arange(len(idx_tmp)) if j not in i]

        if np.any(np.isinf(Z)):
            i, _ = np.where(np.isinf(Z))
            i = np.unique(i)
            #print('Inf rows to delete: ', i)
            Z = np.delete(Z, i, axis=0)
            id_tmp = [id_tmp[j] for j in np.arange(len(id_tmp)) if j not in i]
            idx_tmp = [idx_tmp[j] for j in np.arange(len(idx_tmp)) if j not in i]

        idx = np.random.randint(low=0, high=Z.shape[0], size=min(max_per_subtype, Z.shape[0]))
        Z = Z[idx, :]
        all_Z.append(Z)

        # add meta info about the samples in Z
        meta_tmp['subtype'].extend([subtype]*Z.shape[0])
        meta_tmp['id'].extend([id_tmp[i] for i in idx])
        meta_tmp['index'].extend([idx_tmp[i] for i in idx])

    # final re-shuffling
    if len(all_Z) == 0:
        raise RuntimeWarning('no data was found in the specified path')

    Z = np.vstack(all_Z)
    all_Z = None
    n = Z.shape[0]
    idx = np.random.permutation(n)
    Z = Z[idx, :]
    meta = {'subtype': [meta_tmp['subtype'][i] for i in idx],
            'id': [meta_tmp['id'][i] for i in idx],
            'index': [meta_tmp['index'][i] for i in idx]}

    return Z, meta


def main():
    p = opt.ArgumentParser(description="""
            Builds a codebook based on a set of features.
            """)
    p.add_argument('data_folder', action='store',
                   help='path to a folder containing all the files for the subtypes')
    p.add_argument('encoding_type', action='store', choices=['sda', 'cnn'], default='sda',
                   help='specifies the encoding type for the data file (mainly for file name inference')
    p.add_argument('out_file', action='store', help='resulting model file name')
    p.add_argument('codebook_size', action='store', help='codebook size', type=int)
    p.add_argument('-m', '--max_per_subtype', action='store', default=100,
                   help='how many examples per subtype to use')
    p.add_argument('-s', '--standardize', action='store_true', default=False,
                   help='should the features be standardized before codebook construction?')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose?')

    args = p.parse_args()

    if args.verbose:
        print('Load data...')
    Z, meta = load_image_codes(args.data_folder, args.encoding_type, args.max_per_subtype)
    if args.verbose:
        print('OK')

    Zm = np.mean(Z, axis=0)
    Zs = np.std(Z, axis=0)
    Zs[np.isclose(Zs, 1e-16)] = 1.0

    if args.standardize:
        # make sure each variable (column) is mean-centered and has unit standard deviation
        Z = (Z - Zm[:, None]) / Zs[:, None]

    if args.verbose:
        print('VQ with {} training samples'.format(Z.shape[0]))

    for cbs in np.arange(16,134,16):
        # vector quantization
        rng = np.random.RandomState(12345)
        #vq = MiniBatchKMeans(n_clusters=args.codebook_size, random_state=rng, batch_size=1500,
        #                     compute_labels=True, verbose=False)  # vector quantizer
        vq = MiniBatchKMeans(n_clusters=cbs, random_state=rng, batch_size=1500,
                             compute_labels=True, verbose=False)  # vector quantizer
        vq.fit(Z)

        # compute the average distance and std.dev. of the points in each cluster:
        avg_dist = np.zeros(args.codebook_size)
        sd_dist = np.zeros(args.codebook_size)
        approx_centroids = { 'id' : ['']*args.codebook_size,
                             'index' : np.zeros(args.codebook_size),
                             'subtype' : ['']*args.codebook_size }
        subtypes_centroids = {'a' : np.zeros(args.codebook_size),
                              'b' : np.zeros(args.codebook_size),
                              'c' : np.zeros(args.codebook_size),
                              'd' : np.zeros(args.codebook_size),
                              'e' : np.zeros(args.codebook_size),
                              'outlier' : np.zeros(args.codebook_size)}
        #for k in range(0, args.codebook_size):
        for k in range(0, cbs):
            d = numpy.linalg.norm(Z[vq.labels_ == k, :] - vq.cluster_centers_[k, :], axis=1)
            approx_centroids['id'][k] = meta['id'][np.argmin(d)]
            approx_centroids['index'][k] = meta['index'][np.argmin(d)]
            approx_centroids['subtype'][k] = meta['subtype'][np.argmin(d)]
            avg_dist[k] = d.mean()
            sd_dist[k] = d.std()
            s = [meta['subtype'][i] for i in vq.labels_ if i == k]
            subtypes_centroids['a'][k] = len([x_ for x_ in s if x_ == 'a'])
            subtypes_centroids['b'][k] = len([x_ for x_ in s if x_ == 'b'])
            subtypes_centroids['c'][k] = len([x_ for x_ in s if x_ == 'c'])
            subtypes_centroids['d'][k] = len([x_ for x_ in s if x_ == 'd'])
            subtypes_centroids['e'][k] = len([x_ for x_ in s if x_ == 'e'])
            subtypes_centroids['outlier'][k] = len([x_ for x_ in s if x_ == 'outlier'])

        with ModelPersistence(args.out_file+'{}'.format(cbs), 'c', format='pickle') as d:
            d['codebook'] = vq
            d['shift'] = Zm
            d['scale'] = Zs
            d['standardize'] = args.standardize
            d['avg_dist_to_centroid'] = avg_dist
            d['stddev_dist_to_centroid'] = sd_dist
            d['approx_centroid'] = approx_centroids
            d['subtypes_centroids'] = subtypes_centroids
            d['meta'] = meta

    return True


if __name__ == '__main__':
    main()
