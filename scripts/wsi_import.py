#!/usr/bin/env python2

"""
WSI_IMPORT - uses OpenSlide library to import scanned whole-slide images into a common representation suitable for
image analysis. This representation stores several fields of view (scales) and their division into small tiles in a
hierarchy of folders.

.../original image name/
                        +---- meta.json    <- meta data about the file
                        +---- first downsampling level/
                                           +---- meta.json
                                               | tile_i_j.ppm...
                        +---- second downsampling level/
                                           +---- meta.json 
                                               | tile_i_j.ppm...
                        ...etc...

"""

from __future__ import print_function, division, with_statement

__author__  = 'vlad'
__version__ = 0.15


import openslide as osl

import argparse as opt
import re
import os
import numpy as np
from math import floor
import simplejson as json

from skimage.external import tifffile
from skimage.io import imsave

from segm.tissue import tissue_mask

img_file   = ''
res_prefix = ''
res_format = ''
s_factors  = []
tile_geom  = []
n_levels   = 0
res_xy     = (1, 1)


def generate_tiles(img, save_path, lv):
    global img_file, res_prefix, s_factors, tile_geom, res_xy, n_levels

    tg = (min(tile_geom[lv][0], img.Xsize), min(tile_geom[lv][1], img.Ysize))
    nh = int(floor(img.Xsize / tg[0])) + (1 if img.Xsize % tg[0] != 0 else 0)
    nv = int(floor(img.Ysize / tg[1])) + (1 if img.Ysize % tg[1] != 0 else 0)

    tile_meta = dict({'n_tiles_horiz': nh,
                      'n_tiles_vert': nv,
                      'tile_width': tg[0],
                      'tile_height': tg[1]})

    for i in range(nv):
        for j in range(nh):
            im_sub = img[i*tg[1], j*tg[0], (i+1)*tg[1], (j+1)*tg[0], :]
            tile_meta['tile_'+str(i)+'_'+str(j)] = dict({'name': save_path + '/tile_'+str(i)+'_'+str(j)+'.tiff',
                                                         'i': i, 'j':j,
                                                         'x': j*tg[0], 'y': i*tg[1]})
            imsave(save_path + '/tile_' + str(i) + '_' + str(j) + '.' + res_format, im_sub)

    return tile_meta


def run():
    global img_file, res_prefix, s_factors, tile_geom, res_xy, n_levels

    img = osl.OpenSlide(img_file)

    n_levels = min(n_levels, img.level_count)

    meta = {}
    meta['objective'] = img.properties[osl.PROPERTY_NAME_OBJECTIVE_POWER]
    meta['mpp_x'] = float(img.properties[osl.PROPERTY_NAME_MPP_X])
    meta['mpp_y'] = float(img.properties[osl.PROPERTY_NAME_MPP_Y])

    # Find the region with the object of interest in the slide. This can be done either using
    # the info from the scanner - if available - or by detecting the presence of the tissue on
    # the glass. Once a bounding box for the highest resolution is found, it will be used for
    # subsequent levels by scaling.
    if len(osl.PROPERTY_NAME_BOUNDS_HEIGHT) > 0:
        # if properties for ROI are known
        img_mask_bbox = (int(img.properties[osl.PROPERTY_NAME_BOUNDS_X]),
                         int(img.properties[osl.PROPERTY_NAME_BOUNDS_Y]),
                         int(img.properties[osl.PROPERTY_NAME_BOUNDS_WIDTH]) +
                         int(img.properties[osl.PROPERTY_NAME_BOUNDS_X]) - 1,
                         int(img.properties[osl.PROPERTY_NAME_BOUNDS_HEIGHT]) +
                         int(img.properties[osl.PROPERTY_NAME_BOUNDS_Y]) - 1)


    else:
        # Find a suitable level for detecting the tissue mask:
        for mask_level in range(img.level_count):
            if img.dimensions[mask_level][0] < 5000 and \
                            img.dimensions[mask_level][1] < 5000:
                break
        img_data = img.read_region((0,0), mask_level, img.level_dimensions[mask_level])

        # Segment the tissue mask:
        _, img_mask_bbox = tissue_mask(np.asarray(img_data), percent=0.25, min_tissue_probability=None)

        # Scale-up the bbox for the level 0:
        img_mask_bbox = (img_mask_bbox[0] * 2 ** mask_level,
                         img_mask_bbox[1] * 2 ** mask_level,
                         img_mask_bbox[2] * 2 ** mask_level,
                         img_mask_bbox[3] * 2 ** mask_level)


    path = res_prefix + '/' + os.path.basename(img_file)
    if os.path.exists(path):
        print('Warning: Overwriting old files!')
    else:
        os.mkdir(path)

    for lv in range(n_levels+1):
        # read the tissue region in a PIL Image object
        img_data = img.read_region((img_mask_bbox[0], img_mask_bbox[1]), lv,
                                   (img_mask_bbox[2] - img_mask_bbox[0] + 1, img_mask_bbox[3] - img_mask_bbox[1] + 1))
        img_data_array = np.asarray(img_data)  # get image data into numpy array object

        # extract a mask for the tissue
        img_mask, _ = tissue_mask(img_data_array, percent=0.25, min_tissue_probability=None)

        meta['level_'+str(lv)] = dict({'name': path + '/level_'+str(lv)+'.tiff',
                                       'width': img_data.size[0],
                                       'height': img_data.size[1],
                                       'channels': 3 if len(img_data.getbands()) >= 3 else 1,
                                       'scale': 1,
                                       'from_original_x': img_mask_bbox[0],
                                       'from_original_y': img_mask_bbox[1],
                                       'from_original_width': img_mask_bbox[2] - img_mask_bbox[0] + 1,
                                       'from_original_height': img_mask_bbox[3] - img_mask_bbox[1] + 1})

        # save image
        with tifffile.TiffWriter(meta['level_'+str(lv)]['name'], bigtiff=True) as tif:
            tif.save(img_data_array, compress=7, tile=(512,512))

        # save mask
        with tifffile.TiffWriter(path + '/level_'+str(lv)+'_mask.tiff', bigtiff=True) as tif:
            tif.save(img_mask, compress=9, tile=(512,512))
        # save tiles
        mt = generate_tiles(img_data_array, path+'/level_'+str(lv), lv)

        meta['level_'+str(lv)]['tiles'] = mt

        # Scale-down the bbox for the next level:
        img_mask_bbox = (img_mask_bbox[0] / 2,
                         img_mask_bbox[1] / 2,
                         img_mask_bbox[2] / 2,
                         img_mask_bbox[3] / 2)

    with open(path+'/meta.json', 'w') as fp:
        json.dump(meta, fp, separators=(',', ':'), indent='  ', sort_keys=False)

    img.close()

    return


def main():
    global img_file, res_prefix, s_factors, tile_geom, res_format, n_levels, res_xy

    p = opt.ArgumentParser(description="Import a whole slide image into a convenient structure for processing.")
    p.add_argument('img_file', action='store', help='image file and the name of the root folder for the results')
    p.add_argument('--prefix', action='store', help='path where to store the results', default='./')
    p.add_argument('--levels', action='store', help='number of levels in pyramid', type=int, default=1)
    p.add_argument('--tile', action='store',
                   help='tile geometry: a list of pairs (w,h) for each level or a single pair for all levels',
                   default='(1,1)')
    p.add_argument('--resolution', action='store', help='resolution of the original image (micrometer/px)',
                   default='(1,1)')
    p.add_argument('--format', action='store', help='output image format',
                   choices=['ppm', 'tiff', 'jpeg'], default='ppm')

    args = p.parse_args()
    img_file = args.img_file
    res_prefix = args.prefix
    res_format = args.format
    n_levels = args.levels

    # all non-integer scaling factors are rounded and ensured to be positive
    s_factors = [int(2**_f) for _f in np.arange(args.levels+1)]

    # tile geometry at each level:
    rx = re.compile(r'(\d+,\d+)')
    tile_geom = [(int(abs(float(h))), int(abs(float(v))))
                for h, v in [_p.split(',') for _p in rx.findall(args.tile)]]

    if len(tile_geom) == 1:
        # a single tile geometry for all levels, make the list long enough so
        # the treatment is uniform
        tile_geom = tile_geom*(n_levels+1)

    print(tile_geom)

    _h, _v = rx.findall(args.resolution)[0].split(',')
    res_xy = (int(abs(float(_h))), int(abs(float(_v))))

    if len(tile_geom) > 1:
        # there should be 1 geometry per level:
        if len(s_factors) != len(tile_geom):
            raise(ValueError('There must be (1+levels) tile geometry specifiers' +
                             ' (first specifiers refers to original scale)'))

    run()

    return

if __name__ == '__main__':
    main()
