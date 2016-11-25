"""
Edge detection filter.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

class Edge(Filter):
    """
    Computes the edge features of the image. This calculates the gradient magnitude using a second
    derivative of the Gaussian with a sigma of 1.0 then returns all neighboring offsets in a 7x7
    block.

    The compat flag causes a padding of 0s to be used when needed instead of reflection. This is
    not a good approach since a transition from 0s to the image data will be an edge. Also it takes
    extra effort and memory to do so because the FilterBank class adds reflection padding
    inherently, so we have to detect that and correct it.
    
    The scale flag causes the output data to be multiplied by 0.75, resulting in most data being in
    the range 0 to 1 with a mean of 0.269 (the unscaled theoretical range is 0 to 1.96658337, but
    the vast majority of data is below 4/3). It defaults to the opposite of the compat flag.

    Uses O(2*im.size).
    """
    # Defaults for Python models created before these flags were added
    __compat = True
    __scale = False
    
    def __init__(self, compat=False, scale=None):
        super(Edge, self).__init__(6, Edge.__inten.features)
        self.__compat = compat
        self.__scale = (not compat) if scale is None else scale

    def __call__(self, im, out=None, region=None, nthreads=1):
        # CHANGED: no longer returns optional raw data
        from ._base import get_image_region, replace_sym_padding, hypot
        from ._correlate import correlate_xy #pylint: disable=no-name-in-module
        if self.__compat:
            # In compatibility mode, instead of symmetric reflections we pad with 0s
            im, region = replace_sym_padding(im, 3, region, 6, nthreads)
        else:
            im, region = get_image_region(im, 6, region, nthreads=nthreads)
        # OPT: if out exists, could one of these intermediates be avoided?
        imx = correlate_xy(im, Edge.__kernel[0], Edge.__kernel[1], nthreads=nthreads) # INTERMEDIATE: im.shape + (6,3)
        imy = correlate_xy(im, Edge.__kernel[1], Edge.__kernel[0], nthreads=nthreads) # INTERMEDIATE: im.shape + (6,3)
        hypot(imx, imy, imx, nthreads)
        if self.__scale: imx *= 0.75
        return Edge.__inten(imx, out=out, region=region, nthreads=nthreads)


    ##### Static Fields #####
    __kernel = None
    __inten = None

    @staticmethod
    def __d2dgauss(n, sigma):
        # CHANGED: no longer accepts 2 ns or sigmas or a rotation
        from numpy import ogrid, sqrt
        n2 = -(n+1)/2+1
        ox, oy = ogrid[n2:n2+n, n2:n2+n]
        h = Edge.__gauss(oy, sigma) * Edge.__dgauss(ox, sigma)
        h /= sqrt((h*h).sum())
        h.flags.writeable = False
        return h
    @staticmethod
    def __gauss(x,std): from numpy import exp, sqrt, pi; return exp(-x*x/(2*std*std)) / (std*sqrt(2*pi))
    @staticmethod
    def __dgauss(x,std):
        # CHANGED: removed - sign to create correlation kernel instead of convolution kernel
        # first order derivative of gauss function
        return x * Edge.__gauss(x,std) / (std*std)
    @staticmethod
    def __static_init__():
        from ._base import separate_filter
        from .intensity import Intensity
        Edge.__kernel = separate_filter(Edge.__d2dgauss(7, 1.0))
        Edge.__inten = Intensity.Square(3)
Edge.__static_init__()
