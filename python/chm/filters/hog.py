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
    Computes the HOG (histogram of oriented gradients) features of the image. This produces 36
    features.

    Note that the original used float32 values for many intermediate values so the outputs from
    this function are understandably off by up to 1e-7.

    Uses minimal intermediate memory.
    """
    def __init__(self):
        super(HOG, self).__init__(7, 36)
    def __call__(self, im, out=None, region=None, nthreads=1):
        # OPT: the HOG calculator could be sped up slightly by not zero-padding each block it processes
        # TODO: there is a pre-computed division in the C code, should it be kept?
        from ._base import get_image_region
        from ._hog import hog_entire #pylint: disable=no-name-in-module
        im, region = get_image_region(im, 7, region)
        return hog_entire(im, 15, out=out, nthreads=nthreads)
