"""
SEGM.TISSUE: try to segment the tissue regions from a pathology slide.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.01
__author__ = 'Vlad Popovici'

__all__ = ['tissue_region_from_rgb', 'tissue_components', 'superpixels']

import numpy as np

import skimage.morphology as skm
from skimage.segmentation import slic
from skimage.util import img_as_bool

from sklearn.cluster import MiniBatchKMeans

import mahotas as mh

from util.intensity import _R, _G, _B
from stain.he import rgb2he2


def tissue_region_from_rgb(_img, _min_area=150, _g_th=None):
    """
    TISSUE_REGION_FROM_RGB detects the region(s) of the image containing the
    tissue. The original image is supposed to represent a haematoxylin-eosin
    -stained pathology slide.

    The main purpose of this function is to detect the parts of a large image
    which most probably contain tissue material, and to discard the background.

    Usage:
        tissue_mask = tissue_from_rgb(img, _min_area=150, _g_th=None)

    Args:
        img (numpy.ndarray): the original image in RGB color space
        _min_area (int, default: 150): any object with an area smaller than
            the indicated value, will be discarded
        _g_th (int, default: None): the processing is done on the GREEN channel
            and all pixels below _g_th are considered candidates for "tissue
            pixels". If no value is given to _g_th, one is computed by K-Means
            clustering (K=2), and is returned.

    Returns:
        numpy.ndarray: a binary image containing the mask of the regions
            considered to represent tissue fragments
        int: threshold used for GREEN channel
    """

    if _g_th is None:
        # Apply vector quantization to remove the "white" background - work in the
        # green channel:
        vq = MiniBatchKMeans(n_clusters=2)
        _g_th = int(np.round(0.95 * np.max(vq.fit(_G(_img).reshape((-1,1)))
                                           .cluster_centers_.squeeze())))

    mask = _G(_img) < _g_th

    skm.binary_closing(mask, skm.disk(3), out=mask)

    skm.remove_small_objects(mask, min_size=_min_area, in_place=True)


    # Some hand-picked rules:
    # -at least 5% H and E
    # -at most 25% background
    # for a region to be considered tissue

    h, e, b = rgb2he2(_img)

    mask &= (h > np.percentile(h, 5)) | (e > np.percentile(e, 5))
    mask &= (b < np.percentile(b, 50))               # at most at 50% of "other components"

    mask = mh.close_holes(mask)

    return img_as_bool(mask), _g_th
# tissue_region_from_rgb


def tissue_components(_img, _models, _min_prob=0.4999999999):
    """
    TISSUE_COMPONENTS segment basic tissue components from RGB image: chromatin, connective tissue
    and fat regions. The assignment is based on maximum a posteriori (MAP) rule.

    :param _img: numpy.ndarray
     An RGB image.
    :param _models: dict
     A dictionary with models for predicting the posterior probability that a given pixel (as a 3-element
     vector: R, G, B) belongs to the class of interest.
    :param _min_prob: float (0..1)
     The minimum probability for a pixel to be  considered belonging to a given class.
    :return: numpy.ndarray
     A map of predicted classes: 0 = background, 1 = chromatin, 2 = connective, 3 = fat
    """
    w, h, _ = _img.shape
    n = w * h

    # "background": if no class has a posterior of at least _min_prob
    # the pixel is considered "background"
    prbs = np.zeros((n, 4))
    prbs.fill(_min_prob)

    prbs[:,1] = _models['chromatin'].predict_proba(_img.reshape((-1,3)))[:,1]
    prbs[:,2] = _models['connective'].predict_proba(_img.reshape((-1,3)))[:,1]
    prbs[:,3]  = _models['fat'].predict_proba(_img.reshape((-1,3)))[:,1]
    
    comp_map = np.argmax(prbs, axis=1)   # 0 = background, 1 = chromatin, 2 = connective, 3 = fat
    comp_map = comp_map.reshape((w, h))

    return comp_map
# end tissue_components


def superpixels(img, slide_magnif='x40'):
    """
    SUPERPIXELS: produces a super-pixel representation of the image, with the new
    super-pixels being the average (separate by channel) of the pixels in the
    original image falling in the same "cell".

    :param img: numpy.ndarray
      RGB image

    :param slide_magnif: string
      Indicates the microscope magnification at which the image was acquired.
      It is used to set some parameters, depending on the magnification.

    :return: numpy.ndarray
      The RGB super-pixel image.
    """
    params = dict([('x40', dict([('n_segments', int(100*np.log2(img.size/3))), ('compactness', 1000), ('sigma', 0.0)])),
                   ('x20', dict([('n_segments', int(100*np.log2(img.size/3))), ('compactness', 50), ('sigma', 1.5)]))])

    p = params[slide_magnif]


    sp = slic(img, n_segments=p['n_segments'], compactness=p['compactness'], sigma=p['sigma'],
              multichannel=True, convert2lab=True)

    n_sp = sp.max() + 1
    img_res = np.ndarray(img.shape, dtype=img.dtype)

    for i in np.arange(n_sp):
        img_res[sp == i, 0] = int(np.mean(img[sp == i, 0]))
        img_res[sp == i, 1] = int(np.mean(img[sp == i, 1]))
        img_res[sp == i, 2] = int(np.mean(img[sp == i, 2]))

    return img_res
# end superpixels
