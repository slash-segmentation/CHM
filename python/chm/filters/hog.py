"""
HOG Filter. Mostly implemented in Cython and C++.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

class HOG(Filter):
    """
    Computes the HOG (histogram of oriented gradients) features of the image as per Dalal and Bill
    Triggs (2005). This produces 36 features.
    
    This uses cells of 8x8 pixels and blocks of 2x2 cells. Each cell uses 9 orientation bins,
    covering angles from 0 to pi (unsigned). Each block is normalized using the L2-hys norm,
    clipping at 0.2 before renomarlizing. The 36 features come from the 9 orientation bins for each
    of the 4 cells per block.

    The compat mode adds an implicit 0-padding on each 15x15 region that the HOG filter is
    calculated for. In non-compat mode, only pixels along the edge of the image have any padding
    and it is a reflection padding instead of 0-padding.

    Note that the original used float32 values for many intermediate values so the outputs from
    this function are understandably off by up to 1e-7 even in compat mode.

    Uses minimal intermediate memory.
    """
    
    __compat = False
    
    def __init__(self, compat=False):
        super(HOG, self).__init__(7 if compat else 8, 36)
        self.__compat = compat
        
    def __call__(self, im, out=None, region=None, nthreads=1):
        # TODO: there is a pre-computed division in the C code, should it be kept?
        from ._base import get_image_region
        from ._hog import hog_entire #pylint: disable=no-name-in-module
        im, region = get_image_region(im, self.padding, region)
        return hog_entire(im, 15, self.__compat, out, nthreads)
