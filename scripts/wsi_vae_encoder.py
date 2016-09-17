#!/usr/bin/env python2

# Variational Autoencoder applied to histopathology images.
#
# See https://github.com/y0ast/Variational-Autoencoder

import numpy as np
import time
import os
import warnings
import cPickle
import gzip
import glob
from joblib import *
import argparse as opt

import skimage.io
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.color import rgb2gray

from stain.he import rgb2he
from encode.VAE import VAE

def preprocess_worker(img, img_name):
    #h, _= rgb2he(img, normalize=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = rgb2gray(img)
        h = equalize_adapthist(h)
        h = h.astype('float32')
        h = rescale_intensity(h, out_range=(0.0,1.0))
        h = h.reshape( (np.prod(h.shape),) )
        h -= h.mean()
        
    return h, img_name


def load_images_crc_subtypes(img_path, tile_size, max_per_subtype=1000,
                             max_per_sample=500, max_total=10000):
    """
    Load a series of RGB images (tiles) from a path of the form:
    
    img_path/<subtype>/<sample_id>/t<I>/<tile_size>/*.jpg
    
    where
        <subtype> can be 'a', 'b', 'c', 'd', 'e', 'outlier'
        <sample_id> is a 5-character string
        <I> can ba 1, 2, 3, ... tumor component index in image
        <tile_size> can be "64" or "224"
        
    Once images are read, the H(aematoxylin) intensity is estimated and stored to an
    array. At the end, the data array is normalized.
    """
        
    # read images
    imgs = list()
    img_names = list()
    total = 0
    for subtype in ['a', 'b', 'c', 'd', 'e', 'outlier']:
        per_subtype = 0
        sample_paths = glob.glob(img_path+'/'+subtype+'/*')
        for sp in sample_paths:
            per_sample = 0
            tumor_comp_paths = glob.glob(sp+'/*')
            for tcp in tumor_comp_paths:
                tile_paths = glob.glob(tcp+'/'+str(tile_size)+'/*.jpg')
                for img_name in tile_paths:
                    img = skimage.io.imread(img_name)
                    imgs.append(img)
                    img_names.append(img_name)
                    per_sample += 1
                    per_subtype += 1
                    total += 1
                    if per_sample >= max_per_sample or \
                        per_subtype >= max_per_subtype or \
                        total >= max_total:
                        break
                if per_sample >= max_per_sample or \
                    per_subtype >= max_per_subtype or \
                    total >= max_total:
                    break

            if per_subtype >= max_per_subtype or total >= max_total:
                break
        if total >= max_total:
            break
        
    res = Parallel(n_jobs=cpu_count()) \
        ( delayed(preprocess_worker)(img_, n_) for img_, n_ in zip(imgs, img_names) )

    # deparse results
    # could use:
    # imgs, img_names = zip(*res)
    # imgs = list(imgs)
    # img_names = list(img_names)
    # but it does not look efficient storage-wise for very large lists...
    imgs = [r[0] for r in res]
    img_names = [r[1] for r in res]

    X = np.vstack(imgs)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    
    return X, img_names


def main():
    p = opt.ArgumentParser(description="""
            Encodes a set of images using the Variational Auto-encoder.
            """)
    p.add_argument('img_path', action='store', help='path to image folders')
    p.add_argument('model_path', action='store', help='path where to store the model')
    p.add_argument('enc_num_units', action='store', help='number of hidden units of the encoder', type=int)
    p.add_argument('dec_num_units', action='store', help='number of hidden units of the decoder', type=int)
    p.add_argument('code_size', action='store', help='number of elements in the encoded representation', type=int)
    p.add_argument('-b', '--batch_size', action='store', help='batch size', type=int, default=100)
    p.add_argument('-t', '--tile_size', action='store', help='size of the rectangular tile', type=int, default=64)
    p.add_argument('-s', '--max_per_subtype', action='store', help='maximum sample size per subtype',
                   type=int, default=100)
    p.add_argument('-m', '--max_per_sample', action='store', help='maximum number of tiles per biological sample',
                   type=int, default=100)
    p.add_argument('-n', '--max_total', action='store', help='maximum total sample size',
                   type=int, default=1000)
    p.add_argument('-r', '--learning_rate', action='store', help='learning rate', type=float, default=0.001)
    p.add_argument('-p', '--epochs', action='store', help='number of training epochs',
                   type=int, default=50)
    p.add_argument('-c', '--continuous', action='store_true', help='continuous model?')
    
    
    args = p.parse_args()
    
    print "Loading data..."
    X, imgs_names = load_images_crc_subtypes(args.img_path,
                                             args.tile_size,
                                             args.max_per_subtype,
                                             args.max_per_sample,
                                             args.max_total)
    print "\t --> loaded {0} images of {1} elements". format(X.shape[0], X.shape[1])
    
    print "Initializing the model..."
    model = VAE(args.continuous,
                args.enc_num_units,
                args.dec_num_units,
                args.code_size,
                X,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate    
                )
    print "\t --> done."
    
    batch_order = np.arange(int(model.N / model.batch_size))
    LB_list = list()
    
    print "Training..."
    for epoch in np.arange(args.epochs):
        print "\tEpoch {0}: ".format(epoch)
        tic = time.time()
        LB = 0.0
        np.random.shuffle(batch_order)
        
        for batch in batch_order:
            LB += model.update(batch, epoch)

        LB /= len(batch_order)
        LB_list.append(LB)
        toc = time.time()
        print "\t\t --> finished. LB: {1}, time: {2}".format(epoch, LB, toc-tic)
        #np.save(args.model_path + "LB_list.npy", LB_list)
        #model.save_parameters(args.model_path)
    print("\t --> done.")
    
    valid_LB = model.likelihood(X)
    print "LB on training set: {0}".format(valid_LB)
    
    
if __name__ == '__main__':
    main()