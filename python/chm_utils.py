"""
Utility functions used by the CHM segmentation algorithm.

These are based on the MATLAB functions originally used.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function


########## Conversion ##########
def im2double(im):
    """Converts an image to doubles from 0.0 to 1.0 if not already a floating-point type."""
    from numpy import float64, iinfo
    k, t, im = im.dtype.kind, im.dtype.type, im.astype(float64, copy=False)
    # NOTE: The divisions here could be pre-calculated for ~60% faster code but this is always the
    # first step and the errors will propogate (although minor, max at ~1.11e-16 or 1/2 EPS and
    # averaging ~5e-18). This will only add a milisecond or two per 1000x1000 block.
    # TODO: If I ever extend "compat" mode to this function, it would be a good candidate.
    if k == 'u': im /= iinfo(t).max
    elif k == 'i':
        ii = iinfo(t)
        im -= ii.min
        im /= ii.max - ii.min
    elif k not in 'fb': raise ValueError('Unknown image format')
    return im


########## Resizing Image ##########
def MyUpSample(im, L):
    """
    Increases the image size by 2**L. So if L == 0, image is returned unchanged, if L == 1 the
    image is doubled, and so forth. The upsampling is done with no interpolation (nearest neighbor).

    Supports both 2d and 3d upsampling.
    """
    from numpy.lib.stride_tricks import as_strided
    if L == 0: return im
    N = 2**L
    H, W = im.shape
    return as_strided(im, (H, N, W, N), (im.strides[0], 0, im.strides[1], 0)).reshape((H*N, W*N))

    # Old method:
    #from numpy import repeat
    #return repeat(repeat(im, N, axis=0), N, axis=1)

def MyDownSample(im, L):
    """
    Decreases the image size by 2**L. So if L == 0, image is returned unchanged, if L == 1 the
    image is halved, and so forth. The downsampling uses bicubic iterpolation. If the image size is
    not a multiple of 2**L then extra rows/columns are added to make it so by replicating the edge
    pixels.

    Supports both 2d and 3d downsampling.
    """
    from numpy import pad
    #from numpy import vstack, hstack
    from imresize import imresize_fast # built exactly for our needs
    #from imresize import imresize
    if L==0: return im
    nr,nc = im.shape[:2]
    if nr&1 or nc&1:
        im = pad(im, ((0,nr&1), (0,nc&1)), mode=b'edge')
        #nr += nr&1
        #nc += nc&1
    #if nr&1: im, nr = vstack((im, im[-1:,:,...])), nr+1
    #if nc&1: im, nc = hstack((im, im[:,-1:,...])), nc+1
    return MyDownSample(imresize_fast(im), L-1)
    #return MyDownSample(imresize(im, (nr//2, nc//2)), L-1)


########## Extracting and Padding Image ##########
def get_image_region(im, padding=0, region=None, mode='symmetric'):
    """
    Gets the desired subregion of an image with the given amount of padding. If possible, the
    padding is taken from the image itself. If not possible, the pad function is used to add the
    necessary padding.
        padding is the amount of extra space around the region that is desired
        region is the portion of the image we want to use, or None to use the whole image
            given as top, left, bottom, right - negative values do not go from the end of the axis
            like normal, but instead indicate before the beginning of the axis; the right and bottom
            values should be one past the end just like normal
        mode is the padding mode, if padding is required
    Besides returning the image, a new region is returned that is valid for the returned image to be
    processed again.
    """
    from numpy import pad
    if region is None:
        region = (padding, padding, padding + im.shape[0], padding + im.shape[1]) # the new region
        if padding == 0: return im, region
        return pad(im, int(padding), mode=str(mode)), region
    T, L, B, R = region #pylint: disable=unpacking-non-sequence
    region = (padding, padding, padding + (B-T), padding + (R-L)) # the new region
    T -= padding; L -= padding
    B += padding; R += padding
    if T < 0 or L < 0 or B > im.shape[0] or R > im.shape[1]:
        padding = [[0, 0], [0, 0]]
        if T < 0: padding[0][0] = -T; T = 0
        if L < 0: padding[1][0] = -L; L = 0
        if B > im.shape[0]: padding[0][1] = B - im.shape[0]; B = im.shape[0]
        if R > im.shape[1]: padding[1][1] = R - im.shape[1]; R = im.shape[1]
        return pad(im[T:B, L:R], padding, mode=str(mode)), region
    return im[T:B, L:R], region
