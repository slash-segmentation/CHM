"""
Frangi Filter. Mostly implemented in Cython.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

class Frangi(Filter):
    """
    Computes the Frangi features of the image using the eigenvectors of the Hessian to compute the
    likeliness of an image region to contain vessels or other image ridges, according to the method
    described by Frangi (1998). This produces 14 features from the seven Gaussian sigmas 2, 3, 4,
    5, 7, 9, and 11 each done with the image and the inverted image.

    This filter does not have a compat mode since it never existed in the MATLAB version and does
    not have a scale flag since it is already fixed between 0 and 1.
    
    Uses 4 times the image size as tempoarary memory usage.
    """
    def __init__(self):
        super(Frangi, self).__init__(11*3, 14)
    def __call__(self, im, out=None, region=None, nthreads=1):
        from numpy import empty
        from ._base import get_image_region
        from ._frangi import frangi #pylint: disable=no-name-in-module
        
        sigmas = (2, 3, 4, 5, 7, 9, 11) # TODO: these sigmas or different ones? (less would be good so it is faster...)
        im,region = get_image_region(im, 11*3, region, nthreads=nthreads)
        H,W = region[2]-region[0], region[3]-region[1]
        
        if out is None: out = empty((14, H, W))
        elif out.shape != (14, H, W): raise ValueError('Invalid output')
        
        # Run non-inverted image
        for i,sigma in enumerate(sigmas):
            frangi(get_image_region(im, sigma*3, region, nthreads=nthreads)[0], float(sigma), out[i], nthreads)

        # Run inverted image
        im = 1.0 - im
        for i,sigma in enumerate(sigmas, 7):
            frangi(get_image_region(im, sigma*3, region, nthreads=nthreads)[0], float(sigma), out[i], nthreads)

        return out
