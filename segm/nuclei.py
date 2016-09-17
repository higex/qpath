"""
SEGM.NUCLEI: segmentation of nuclei in different contexts.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'Vlad Popovici'
__version__ = 0.2

import numpy as np

import skimage.morphology as morph
import skimage.filters
import skimage.segmentation as sgm
import skimage.measure as ms

import mahotas


# NUCLEI_REGIONS
def nuclei_regions(comp_map, img_h=None, config=None):
    """
    NUCLEI_REGIONS: extract "support regions" for nuclei. This function
    expects as input a "tissue components map" (as returned, for example,
    by segm.tissue_components) where values of 1 indicate pixels
    corresponding to nuclei.

    It returns a set of compact support regions corresponding to the
    nuclei.


    :param comp_map: numpy.ndarray
       A mask identifying different tissue components, as obtained
       by classification in RGB space. The value 0 is used to denote
       the background.
       
    :param img_h: numpy.ndarray (optional, default: None)
       The corresponding Haematoxylin plane extracted from the image.
       If provided, it is used to refine the segmentation of the regions.

       See segm.tissue.tissue_components()

    :return: numpy.ndarray
       A mask with nuclei regions.
    """

    mask = (comp_map == 1)   # use the components classified by color
    
    if img_h is not None:
        img_h = img_h * mask
        th = skimage.filters.threshold_otsu(img_h, nbins=32)
        mask = img_h > th

    mask = mahotas.close_holes(mask)
    morph.remove_small_objects(mask, in_place=True)

    dst  = mahotas.stretch(mahotas.distance(mask))
    if config is not None and 'nuclei_regions_strel1' in config:
        Bc = np.ones(config['nuclei_regions_strel1'])
    else:
        Bc   = np.ones((19,19))  # to adapt based on magnification

    lmax = mahotas.regmax(dst, Bc=Bc)
    spots, _ = mahotas.label(lmax, Bc=Bc)
    regions = mahotas.cwatershed(lmax.max() - lmax, spots) * mask

    return regions
# end NUCLEI_REGIONS
    

def detect_nuclei1(im, color_models, config):
    im_h, _, _ = rgb2he2(im)
    cmap = tissue_components(im, color_models)

    regs = nuclei_regions(cmap, im_h, config)

    im_n = im_h * (regs != 0)
    
    regs, _, _ = sgm.relabel_sequential(regs)  # ensure regions ID 1,2,...
    reg_props = ms.regionprops(regs, im_h)

    for r in reg_props:
    if r.area < 100:
        regs[regions == r.label] = 0
        continue
    if r.solidity < 0.1:
        regs[regions == r.label] = 0
        continue
    if r.minor_axis_length / r.major_axis_length < 0.1:
        regs[regions == r.label] = 0
        continue
    regs = relabel(regs)
    reg_props = regionprops(regs, im_h1)

    blobs = blob_dog(im_n, min_sigma=config['detect_nuclei_blob_min_sg'],
                     max_sigma=config['detect_nuclei_blob_max_sg'],
                     threshold=config['detect_nuclei_blob_thr'])
    blobs[:, 2] = blobs[:, 2] * sqrt(2)

    lb = np.zeros((blobs.shape[0],1))
    for k in np.arange(blobs.shape[0]):
        y, x = blobs[k,0:2]
        lb[k, 0] = regs[y, x]
    blobs = np.concatenate((blobs, lb), axis=1)

    final_blobs = []
    for r in reg_props:
        yc, xc = r.centroid
        idx = np.where(blobs[:,3] == r.label)[0]
        if len(idx) == 0:
            continue
        d = np.zeros(len(idx))
        for i in range(len(idx)):
            y, x = blobs[idx[i],0:2]
            d[i] = (x-xc)**2 + (y-yc)**2
        i = np.argmin(d)
        final_blobs.append(blobs[idx[i],])
        
    return final_blobs
