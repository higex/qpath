# -*- coding: utf-8 -*-
"""
NDPI2REGIONS2: All regions marked as t1, t2,... are saved in individual files with
all pixels in the background set to 0.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import argparse as opt
import xml.etree.ElementTree as ET
#from xml.dom import minidom
import openslide as osl
import numpy as np
from skimage.draw import polygon
import skimage.io


__version__ = 0.1
__author__ = 'Vlad Popovici'


def main():
    p = opt.ArgumentParser(description="""
            Save annotated regions in external files.
            """)
    p.add_argument('ndpi', action='store', help='Hamamatsu NDPI file')
    p.add_argument('ndpa', action='store', help='Hamamatsu NDPA annotation file corresponding to the NDPI file')
    p.add_argument('out', action='store', help='result image file name')
    p.add_argument('-p', '--prefix', action='store', help='prefix of contour names to be saved', default='t')
    p.add_argument('-l', '--level', action='store', type=int, default=0,
                   help='magnification level from which the regions are to be extracted (0: maximum magnification)')
    p.add_argument('-m', '--mask', action='store_true', help='save region mask?', default=False)
    p.add_argument('-i', '--image', action='store', default=None,
                   help='already existing image of the desired field of view')

    args = p.parse_args()

    d = 2**args.level

    out_ext = args.out.split('.')[-1]           # extension
    out_basename = '.'.join(args.out.split('.')[:-1])

    ndpi = osl.OpenSlide(args.ndpi)
    x_off = long(ndpi.properties['hamamatsu.XOffsetFromSlideCentre'])
    y_off = long(ndpi.properties['hamamatsu.YOffsetFromSlideCentre'])
    x_mpp = float(ndpi.properties['openslide.mpp-x'])
    y_mpp = float(ndpi.properties['openslide.mpp-y'])
    dimX0, dimY0 = ndpi.level_dimensions[0]

    print('Slide: ', dimX0, 'x', dimY0)
    xml_file = ET.parse(args.ndpa)
    xml_root = xml_file.getroot()
    
    region_found = False

    for ann in list(xml_root):
        name = ann.find('title').text
        if not name.startswith(args.prefix):
            continue
        p = ann.find('annotation')
        if p is None:
            continue
        if p.find('closed').text != "1":
            continue                   # not a closed contour
        p = p.find('pointlist')
        if p is None:
            continue
        region_found = True
        xy_coords = []
        for pts in list(p):
            # coords in NDPI system, relative to the center of the slide
            x = long(pts.find('x').text)
            y = long(pts.find('y').text)

            # convert the coordinates:
            x -= x_off                 # relative to the center of the image
            y -= y_off

            x /= (1000L*x_mpp)          # in pixels, relative to the center
            y /= (1000L*y_mpp)
            
            x = long((x + dimX0 / 2)/d) # in pixels, relative to UL corner
            y = long((y + dimY0 / 2)/d)

            xy_coords.append([x, y])

        if len(xy_coords) < 5:
            # too short
            continue

        # check the last point to match the first one
        if (xy_coords[0][0] != xy_coords[-1][0]) or (xy_coords[0][1] != xy_coords[-1][1]):
            xy_coords.append(xy_coords[0])

        xy_coords = np.array(xy_coords, dtype=np.int64)
        xmn, ymn = xy_coords.min(axis=0)
        xmx, ymx = xy_coords.max(axis=0)

        if xmn < 0 or ymn < 0:
            raise RuntimeError('Negative coordinates!'+' (x='+str(xmn)+', y='+str(ymn)+')')

        if args.image is None:
            # must read from the NDPI image
            # top-left corner (below) must be in level-0 coordinates!!
            img0 = ndpi.read_region((xmn*d, ymn*d), args.level, (xmx-xmn+1, ymx-ymn+1))
            img = np.array(img0)
        else:
            img = skimage.io.imread(args.image)
            # check if image size matches the indicated level
            if ndpi.level_dimensions[args.level][0] != img.shape[1] or ndpi.level_dimensions[args.level][1] != img.shape[0]:
                print("Image size: %ld x %ld" % (img.shape[0], img.shape[1]))
                print("Field size: %ld x %ld" % (ndpi.level_dimensions[args.level][1], ndpi.level_dimensions[args.level][0]))
                raise RuntimeError("Provided image does not match the indicated magnification level")
            img = img[ymn:ymx+1, xmn:xmx+1, ]
        with open(out_basename+'_'+name+'_region.dat','w') as f:
            f.writelines(['\t'.join([str(s) for s in [args.level, ymn, xmn, ymx, xmx]])])

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.ubyte)
        rr, cc = polygon(xy_coords[:,1]-ymn, xy_coords[:,0]-xmn, mask.shape)
        mask[rr, cc] = 1
        if img.ndim == 2:
            img *= mask
        else:
            for k in np.arange(img.shape[2]):
                img[:,:,k] *= mask

        if args.mask:
            skimage.io.imsave(out_basename+'_'+name+'_mask.'+out_ext, 255*mask)

        skimage.io.imsave(out_basename+'_'+name+'.'+out_ext, img)

        # end for ann...

    if not region_found:
        print('No region found with the specified prefix!')

    return


if __name__ == '__main__':
    main()