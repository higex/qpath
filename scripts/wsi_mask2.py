#!/usr/bin/env python2

# WSI_MASK2: use OpenCV for faster operations.

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import argparse as opt
import skimage.io
import skimage.external.tifffile
import numpy as np

from segm.tissue import tissue_region_from_rgb_fast

__version__ = 0.01
__author__ = 'Vlad Popovici'


def main():
    p = opt.ArgumentParser(description="""
            Produces a mask covering the tissue region in the image.
            """)
    p.add_argument('img_file', action='store', help='RGB image file')
    p.add_argument('out_file', action='store', help='Tissue image after masking. Note that .tiff is added.')
    p.add_argument('--prefix', action='store',
                   help='optional prefix for the result files: prefix_tissue_mask.jpeg',
                   default=None)
    p.add_argument('--minarea', action='store', type=int,
                   help='object smaller than this will be removed',
                   default=150)
    p.add_argument('--gth', action='store', type=int,
                   help='if provided, indicates the threshold in the green channel',
                   default=220)
    p.add_argument('--ksize', action='store', type=int,
                   help='size of structuring element in fill gaps - depends on the magnification',
                   default=33)
    p.add_argument('-m', '--mask', action='store_true', help='Should the mask be saved?')

    args = p.parse_args()
    base_name = os.path.basename(args.img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    if args.prefix is not None:
        pfx = args.prefix
    else:
        pfx = base_name

    img = skimage.io.imread(args.img_file)
    if img.ndim == 3:
        # drop alpha channel
        img = img[:,:,:3]

    mask = tissue_region_from_rgb_fast(img, _min_area=args.minarea, _g_th=args.gth, _ker_size=args.ksize)

    if img.ndim == 3:
        img[np.logical_not(mask), :] = np.zeros(img.shape[2])
    else:
        img[np.logical_not(mask)] = 0

    skimage.external.tifffile.imsave(args.out_file+'.tiff', img, compress=9, bigtiff=True)

    if args.mask:
        skimage.io.imsave(pfx+'_tissue_mask.jpeg', 255*mask.astype('uint8'))

    return
# end main()

if __name__ == '__main__':
    main()
