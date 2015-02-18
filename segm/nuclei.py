"""
SEGM.NUCLEI: segmentation of nuclei in different contexts.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'Vlad Popovici'
__version__ = 0.2

import numpy as np
import skimage.morphology as morph

import mahotas


# NUCLEI_REGIONS
def nuclei_regions(comp_map):
    """
    NUCLEI_REGIONS: extract "support regions" for nuclei. This function
    expects as input a "tissue components map" (as returned, for example,
    by segm.tissue_components) where values of 1 indicate pixels
    corresponding to nuclei.

    It returns a set of compact support regions corresponding to the
    nuclei.


    :param comp_map: numpy.ndarray
       A mask identifying different tissue components, as obtained
       by classification in RGB space. The value 0

       See segm.tissue.tissue_components()

    :return: numpy.ndarray
       A mask with nuclei regions.
    """

    mask = (comp_map == 1)   # use the components classified by color

    mask = mahotas.close_holes(mask)
    morph.remove_small_objects(mask, in_place=True)

    dst  = mahotas.stretch(mahotas.distance(mask))
    Bc   = np.ones((9,9))
    lmax = mahotas.regmax(dst, Bc=Bc)
    spots, _ = mahotas.label(lmax, Bc=Bc)
    regions = mahotas.cwatershed(lmax.max() - lmax, spots) * mask

    return regions
# end NUCLEI_REGIONS