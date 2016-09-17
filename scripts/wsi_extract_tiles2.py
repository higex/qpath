#!/usr/bin/env python2
"""
WSI_EXTRACT_TILES2: extract a number of tiles (rectangular regions) from an image and saves them as images.
"""
from __future__ import (division, print_function, unicode_literals)

__author__ = 'Vlad Popovici'
__version__ = 0.1

import os
import argparse as opt
import warnings

from util.explore import random_window, sliding_window
import skimage.io
from skimage.color import rgb2lab, rgb2hsv

from skimage.util import img_as_bool
from skimage.transform import rescale
import skimage.exposure
import numpy as np
from stain.he import rgb2he


def main():
    p = opt.ArgumentParser(description="""
        Extract tiles (rectangular regions) from an RGB image. If a mask is provided,
        the tiles are extracted only within the regions covered by the mask. The
        resulting images can optionally be converted to another color space. 
        """)
    p.add_argument('img_file', action='store', help='RGB image file of an H&E slide')
    p.add_argument('out_file', action='store', default='tiles',
                   help='Prefix of the resulting images <prefix>_0001,...')

    p.add_argument('--scale', action='store', type=float, default=1.0,
                   help='Scale the image? (both image and mask are scaled) (default: 1.0)')
    p.add_argument('--wsize', action='store', type=int, default=50,
                   help='Sliding window size (default: 50)')
    p.add_argument('--wstep', action='store', type=int, default=5,
                   help='Sliding window step (default:5)')
    p.add_argument('--sampling', action='store', choices=['random', 'slide'],
                   help='Sampling strategy: random or slide (default: random)', default='random')
    p.add_argument('--nw', action='store', type=int, default=100,
                   help='Maximum number of tiles to extract (default: 100)')
    p.add_argument('--mask', action='store', help='Image mask (binary image) or "alpha" for mask in alpha channel',
                   default=None)
    p.add_argument('--haematoxylin', action='store_true', help='Save intensity images corresponding to Haematoxylin staining',
                   default=None)
    p.add_argument('--eosin', action='store_true', help='Save intensity images corresponding to Eosin staining',
                   default=None)
    p.add_argument('--colspace', action='store', choices=['rgb','lab','hsv'],
                   help='color space of the resulting images', default='rgb')
    
    
    args = p.parse_args()
    img_file = args.img_file
    
    w_area = args.wsize**2             # consider rectangular patches. for the moment
    w_area_coefficient = 1.0           # later as a command line parameter?
    w_area *= w_area_coefficient
    
    scale = args.scale
    out_file = args.out_file

    base_name = os.path.basename(img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    try:
        img = skimage.io.imread(img_file)
    except Warning:
        pass
    
    mask = None
    if args.mask is not None:
        if args.mask == 'alpha':
            if img.ndim == 3 and img.shape[2] == 4:
                # use the mask from the transparency (alpha) channel
                mask = img[:, :, 3]
            else:
                raise RuntimeError("The image has no alpha (transparency) channel")
        else:
            # assume the name of a file was provided
            mask = skimage.io.imread(args.mask)
        with warnings.catch_warnings(): # avoid "Possible precision loss when converting from uint8 to bool"
            warnings.simplefilter("ignore")
            mask = img_as_bool(mask)

    if args.scale != 1.0:
        img = rescale(img, args.scale, preserve_range=True)
        if mask is not None:
            mask = rescale(mask, args.scale, preserve_range=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask = img_as_bool(mask)

    if args.haematoxylin or args.eosin:
        if img.ndim == 2:
            raise RuntimeWarning('Image is not color - no conversion to H or E')
        else:
            if args.haematoxylin:
                img, _ = rgb2he(img, normalize=True)
            else:
                _, img = rgb2he(img, normalize=True)
                
            img = skimage.exposure.rescale_intensity(img, out_range=(0,255))
            with warnings.catch_warnings(): # avoid "Possible precision loss when converting from uint8 to bool"
                warnings.simplefilter("ignore")
                img = img.astype(np.uint8)

    if img.ndim == 2:                  # intensity image?
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))  # add an extra dimension
        
    if img.ndim != 3:
        raise RuntimeError('Image array dimensions != (2 or 3)')
    
    all_white = 255*w_area*img.shape[2]   # if all pixels are white, then they would sum to this value over the patch
        
    tiles = list()
    tiles_coords = list()
    
    if mask is None:  # outside the loops
        if args.sampling == 'random':
            img_iterator = random_window(img.shape[:-1], (args.wsize, args.wsize), n=-1)
            k = 0
            while k < args.nw:
                w = img_iterator.next()
                if img[w[0]:w[1],w[2]:w[3],:].sum() >= all_white:
                    # too much white (no info)
                    continue
                tiles.append(img[w[0]:w[1],w[2]:w[3],:])
                tiles_coords.append('\t'.join([str(k+1), str(w[0]), str(w[1]), str(w[2]), str(w[3])])+'\n')
                k += 1
        else:
            img_iterator = sliding_window(img.shape[:-1], (args.wsize, args.wsize), step=(args.wstep, args.wstep))
            k = 0
            for w in img_iterator:
                if img[w[0]:w[1],w[2]:w[3],:].sum() >= all_white:
                    # too much white (no info)
                    continue
                tiles.append(img[w[0]:w[1],w[2]:w[3],:])
                tiles_coords.append('\t'.join([str(k+1), str(w[0]), str(w[1]), str(w[2]), str(w[3])])+'\n')
                k += 1
                if k > args.nw:
                    break
    else:
        if args.sampling == 'random':
            img_iterator = random_window(img.shape[:-1], (args.wsize, args.wsize), n=-1)
            k = 0
            while k < args.nw:
                w = img_iterator.next()
                if img[w[0]:w[1],w[2]:w[3],:].sum() >= all_white:
                    # too much white (no info)
                    continue
                # check if w is fully inside the mask
                if mask[w[0]:w[1], w[2]:w[3]].sum() >= w_area:
                    #print(mask[w[0]:w[1], w[2]:w[3]].sum())
                    tiles.append(img[w[0]:w[1],w[2]:w[3],:])
                    tiles_coords.append('\t'.join([str(k+1), str(w[0]), str(w[1]), str(w[2]), str(w[3])])+'\n')
                    k += 1
        else:
            img_iterator = sliding_window(img.shape[:-1], (args.wsize, args.wsize), step=(args.wstep, args.wstep))
            k = 0
            for w in img_iterator:
                if img[w[0]:w[1],w[2]:w[3],:].sum() >= all_white:
                    # too much white (no info)
                    continue                
                # check if w is fully inside the mask
                if mask[w[0]:w[1], w[2]:w[3]].sum() >= w_area:
                    tiles.append(img[w[0]:w[1],w[2]:w[3],:])
                    tiles_coords.append('\t'.join([str(k+1), str(w[0]), str(w[1]), str(w[2]), str(w[3])])+'\n')
                    k += 1
                if k > args.nw:
                    break

    i = 1

    for patch in tiles:
        try:
            if args.colspace == 'rgb':
                skimage.io.imsave(args.out_file+'{:07d}'.format(i)+'.jpg', patch)
            elif args.colspace == 'lab':
                skimage.io.imsave(args.out_file+'{:07d}'.format(i)+'.jpg', rgb2lab(patch))
            elif args.colspace == 'hsv':
                skimage.io.imsave(args.out_file+'{:07d}'.format(i)+'.jpg', rgb2hsv(patch))
        except UserWarning:
            pass
        i += 1
        
    with open('tiles_coordinates.dat', 'w') as f:
        f.writelines(tiles_coords)
    
if __name__ == "__main__":
    main()