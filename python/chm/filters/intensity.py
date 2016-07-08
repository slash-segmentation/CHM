"""
Intensity/neighborhood filter. Mostly implemented in Cython.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

__all__ = ['Intensity',
           'SquareNeighborhood', 'SquareFootprint',
           'StencilNeighborhood', 'StencilFootprint']

class Intensity(Filter):
    """
    Computes the neighborhood/intensity features of the image. Basically, offsets is a 2xN array
    where each column is a relative offset from each pixel. A feature is generated for each relative
    set of coordinates (so the number of features if the number of columns in the offsets).

    An intensity filter with either square or stencil neighborhoods can be easily generated using
    Intensity.Square and Intensity.Stencil.

    The original function was always used with an image that had a padding that reflected the
    foreground image, so that is now permanently integrated into this function and now takes the
    original, un-padded, image. The padding used is the max offset given.

    This function uses the region information in a special way so that no physical padding is ever
    added, instead handling reflections as needed directly. Implemented in Cython for speed and
    multi-threading support. Uses very minimal intermediate memory.
    """
    def __init__(self, offsets):
        # Fix an issue where intp isn't always stored as intp
        from numpy import intp
        if offsets.dtype != intp:
            if offsets.dtype.kind == 'i' and offsets.dtype.itemsize == intp(0).itemsize:
                offsets = offsets.view(intp)
            else:
                offsets = offsets.astype(intp)
        super(Intensity, self).__init__(abs(offsets.ravel()).max(), offsets.shape[1])
        self.__offsets = offsets
    def __call__(self, im, out=None, region=None, nthreads=1):
        from ._intensity import intensity #pylint: disable=import-error
        return intensity(im, self.__offsets, out=out, region=region, nthreads=nthreads)

    __squares = {}
    @staticmethod
    def Square(radius):
        """A square neighborhood with every value of size radius*2+1 on each side"""
        F = Intensity.__squares.get(radius)
        if F is None: Intensity.__squares[radius] = F = Intensity(SquareNeighborhood(radius))
        return F

    __stencils = {}
    @staticmethod
    def Stencil(radius):
        """A neighborhood of size radius*2+1 on each side with the X and + parts filled out."""
        F = Intensity.__stencils.get(radius)
        if F is None: Intensity.__stencils[radius] = F = Intensity(StencilNeighborhood(radius))
        return F

def __memoize_neighborbood(f):
    """
    This decorator causes neighborhoods to be cached. Since each neighbor type is really only used
    at 1 or 2 different radii, this means they only need to be calculated once.
    """
    #pylint: disable=protected-access
    f.__memos = memos = {}
    def memoized(radius):
        x = memos.get(radius)
        if x is None:
            x = f(radius)
            x.flags.writeable = False
            memos[radius] = x
        return x
    return memoized

# Scipy ndimage functions take footprints (boolean masks) instead of neighborhoods (list of indices)
# like the MATLAB functions, so we provide both here. Basically the neighborhoods are the footprints
# with a call to indices or where and subtracting radius.

@__memoize_neighborbood
def SquareNeighborhood(radius):
    """A square neighborhood with every value of size radius*2+1 on each side"""
    from numpy import indices, intp
    diam = 2*radius + 1
    return (indices((diam, diam)) - radius).reshape((2, -1)).astype(intp, copy=False).view(intp) # last two are to make sure it is intp dtype - numpy bug needs both to overcome!

@__memoize_neighborbood
def SquareFootprint(radius):
    from numpy import ones
    diam = 2*radius + 1
    return ones((diam, diam), dtype=bool)

@__memoize_neighborbood
def StencilNeighborhood(radius):
    """A neighborhood of size radius*2+1 on each side with the X and + parts filled out."""
    from numpy import ogrid, array, where, intp
    r,c = ogrid[-radius:radius+1,-radius:radius+1]
    return array(where((abs(r)==abs(c)) + (r*c==0))[::-1], dtype=intp) - radius

@__memoize_neighborbood
def StencilFootprint(radius):
    from numpy import ogrid
    r,c = ogrid[-radius:radius+1,-radius:radius+1]
    return ((abs(r)==abs(c)) + (r*c==0))
