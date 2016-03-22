"""
imresize function that mostly emulates MATLAB's imresize function. Additionally a 'fast' variant is
provided that always halves the image size with bicubic and and antialiasing.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from pysegtools.general.delayed import delayed
from pysegtools.general import cython
from numpy import finfo, float64

cython.install()
import __imresize

__methods = delayed(lambda:{
    'nearest' : (box,1), 'bilinear' : (triangle,2), 'bicubic' : (cubic,4),
    'box'     : (box,1), 'triangle' : (triangle,2), 'cubic'   : (cubic,4),
    'lanczos2' : (lanczos2,4), 'lanczos3' : (lanczos3,6),
}, dict)
__eps = delayed(lambda:finfo(float64).eps, float64)


def imresize(im, scale_or_output_shape, method='bicubic', antialiasing=None):
    """
    Resize an image.

    scale_or_output_shape is one of:
        floating-point number for scale
        tuple/list of 2 floating-point numbers for multi-scales
        tuple/list of 2 ints for output_size (supporting None for calculated dims)

    method is one of:
        'nearest' or 'box'
        'bilinear' or 'triangle'
        'bicubic' or 'cubic' (default)
        'lanczos2'
        'lanczos3'
        tuple/list of a kernel function and a kernel width

    antialiasing defaults to True except for nearest which is False (box is True as well)

    Unlike MATLAB's imresize, the following features are not supported:
      * gpu-arrays
      * indexed images (so none of the params map, Colormap, Dither), however this can be
        accomplished outside of the function
      * 0 or 1 dimensional images, however this can be accomplished with adding length-1 dimensions
        outside the function
    """
    from numpy import uint8, require, ascontiguousarray
    
    if im.dtype.kind not in 'buif' or im.size == 0 or im.ndim < 2: raise ValueError("Invalid image")
    
    scale, output_shape = __scale_shape(im, scale_or_output_shape)
    antialiasing = (antialiasing is None and method != 'nearest') or bool(antialiasing)
    kernel, kernel_width = __methods.get(method, method)

    im = require(im, im.dtype, 'A')
    if not im.flags.forc: im = ascontiguousarray(im)
    logical = im.dtype.kind == 'b'
    im = uint8(im)*255 if logical else im

    # Calculate interpolation weights and indices for each dimension.
    wghts1, inds1 = __contributions(im.shape[0], output_shape[0], scale[0], kernel, kernel_width, antialiasing)
    wghts2, inds2 = __contributions(im.shape[1], output_shape[1], scale[1], kernel, kernel_width, antialiasing)

    if wghts1 is None and wghts2 is None:
        im = im[inds1,inds2.T,...]
    elif scale[0] <= scale[1]:
        im = __imresize_dim(im, wghts1, inds1, 0)
        im = __imresize_dim(im, wghts2, inds2, 1)
    else:
        im = __imresize_dim(im, wghts2, inds2, 1)
        im = __imresize_dim(im, wghts1, inds1, 0)
        
    return im > 128 if logical else im

def __imresize_dim(im, weights, indices, dim):
    if weights is None:
        # nearest neighbor
        subscripts = [slice(None)] * im.ndim
        subscripts[dim] = indices
        return im[subscripts]

    if dim == 1: im = im.swapaxes(1, 0)
    sh = im.shape[1:]
    im = im.reshape((im.shape[0], -1))
    im = __imresize.imresize(im, weights, indices)
    im = im.reshape((im.shape[0],) + sh)
    return im.swapaxes(1, 0) if dim == 1 else im

def __scale_shape(im, scale_or_shape):
    from math import ceil
    from numbers import Real, Integral
    from collections import Sequence
    
    if isinstance(scale_or_shape, Real) and scale_or_shape > 0:
        scale = float(scale_or_shape)
        return (scale, scale), (ceil(scale*im.shape[0]), ceil(scale*im.shape[1]))

    if isinstance(scale_or_shape, Sequence) and len(scale_or_shape) == 2:
        if all((isinstance(ss, Integral) and ss > 0) or ss is None for ss in scale_or_shape) and any(ss is not None for ss in scale_or_shape):
            shape = tuple(scale_or_shape)
            if shape[0] is None:
                shape[0], sz_dim = shape[1] * im.shape[0] / im.shape[1], 1
            elif shape[1] is None:
                shape[1], sz_dim = shape[0] * im.shape[1] / im.shape[0], 0
            else:
                sz_dim = None
            shape = (int(ceil(shape[0])), int(ceil(shape[1])))

            if sz_dim is not None:
                scale = shape[sz_dim] / im.shape[sz_dim]
                scale = (scale, scale)
            else:
                scale = (shape[0]/im.shape[0], shape[1]/im.shape[1])
            return scale, shape
        
        elif all(isinstance(ss, Real) and ss > 0 for ss in scale_or_shape):
            scale0, scale1 = float(scale_or_shape[0]), float(scale_or_shape[1])
            return (scale0, scale1), (ceil(scale0*im.shape[0]), ceil(scale1*im.shape[1]))
    
    raise ValueError("Invalid scale/output_shape")

def __contributions(in_len, out_len, scale, kernel, kernel_width, antialiasing):
    from numpy import arange, floor, ceil, intp, where, logical_not, delete

    antialiasing = antialiasing and scale < 1
    
    # Use a modified kernel to simultaneously interpolate and antialias
    if antialiasing: kernel_width /= scale

    # Output-space coordinates.
    x = arange(out_len, dtype=float64)
    
    # Input-space coordinates. Calculate the inverse mapping such that 0.5 in output space maps to
    # 0.5 in input space, and 0.5+scale in output space maps to 1.5 in input space.
    x /= scale
    x += 0.5

    # What is the left-most pixel that can be involved in the computation?
    left = x - kernel_width/2
    left = floor(left, out=left).astype(intp)
    # left is the slice: int(-kernel_width/2) to int((out_len-1)/scale - kernel_width/2) stepping by 1/scale (kinda)
    
    # What is the maximum number of pixels that can be involved in the computation? Note: it's OK
    # to use an extra pixel here; if the corresponding weights are all zero, it will be eliminated
    # at the end of this function.
    P = int(ceil(kernel_width)) + 2

    # The indices of the input pixels involved in computing the k-th output pixel are in row k of the indices matrix.
    indices = left[:,None] + arange(P, dtype=intp)
    
    # The weights used to compute the k-th output pixel are in row k of the weights matrix.
    x = x[:,None] - indices
    if antialiasing:
        x *= scale
        weights = kernel(x)
        weights *= scale
    else:
        weights = kernel(x)
    
    # Normalize the weights matrix so that each row sums to 1.
    weights /= weights.sum(1, keepdims=True)
    
    # Clamp out-of-range indices; has the effect of replicating end-points.
    indices = indices.clip(0, in_len-1, out=indices)

    # If a column in weights is all zero, get rid of it.
    while not weights[:,0].any():
        if len(weights) == 1: return None, indices
        weights = weights[:,1:]
        indices = indices[:,1:]
    while not weights[:,-1].any():
        weights = weights[:,:-1]
        indices = indices[:,:-1]
    kill = where(logical_not(weights.any(0)))[0]
    if len(kill) > 0:
        weights = delete(weights, kill, axis=1)
        indices = delete(indices, kill, axis=1)

    # Detect if using nearest neighbor
    if (weights.ndim == 1 or weights.shape[1] == 1) and (weights == 1).all():
        return None, indices
    return weights, indices


########## Fast imresize ##########

def imresize_fast(im):
    """
    Like imresize but with the following assumptions:
        scale is always (0.5, 0.5)
        method is always 'bicubic'
        antialiasing is always on
    But it does support everything else (2/3-D images, logical/integral/floating point types)
    """
    from numpy import uint8, require, ascontiguousarray
    if im.dtype.kind not in 'buif' or im.size == 0 or im.ndim < 2: raise ValueError("Invalid image")
    im = require(im, im.dtype, 'A')
    if not im.flags.forc: im = ascontiguousarray(im)
    logical = im.dtype.kind == 'b'
    im = uint8(im)*255 if logical else im
    im = __imresize_dim_fast(im, 0)
    im = __imresize_dim_fast(im, 1)
    return im > 128 if logical else im

def __imresize_dim_fast(im, dim):
    if dim == 1: im = im.swapaxes(1, 0)
    sh = im.shape
    im = im.reshape((sh[0], -1))
    im = __imresize.imresize_fast(im)
    im = im.reshape((im.shape[0],) + sh[1:])
    return im.swapaxes(1, 0) if dim == 1 else im 


########## Filters ##########
   
def cubic(x): # bicubic
    """
    See Keys, "Cubic Convolution Interpolation for Digital Image Processing," IEEE Transactions on
    Acoustics, Speech, and Signal Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
    """
    from numpy import less_equal, less, multiply, abs # pylint: disable=redefined-builtin
    absx = abs(x, out=x)
    absx2 = absx*absx
    absx3 = absx2*absx
    
    absx2 *= 2.5
    
    A = 1.5*absx3; A -= absx2; A += 1
    
    absx3 *= -0.5
    B = absx3; B += absx2; B -= multiply(4, absx, out=absx2); B += 2
    
    A *= less_equal(absx, 1, out=absx2)
    B *= less(1, absx, out=absx2)
    B *= less_equal(absx, 2, out=absx2)

    A += B
    return A

    #a = -0.5 # MATLAB's constant, OpenCV uses -0.75
    #return ((a+2)*absx3 - (a+3)*absx2 + 1) * (absx<=1) + \
    #       (    a*absx3 -   5*a*absx2 + 8*a*absx - 4*a) * logical_and(1<absx, absx<=2)
    ## Hardcoded a value
    #return (1.5*absx3 - 2.5*absx2 + 1) * (absx<=1) + \
    #       (-.5*absx3 + 2.5*absx2 - 4*absx + 2) * logical_and(1<absx, absx<=2)
    
def box(x): # nearest
    from numpy import logical_and, less
    # logical_and(-0.5 <= x,x < 0.5)
    return logical_and(-0.5<=x, less(x,0.5, out=x), out=x)
def triangle(x): # bilinear
    from numpy import less_equal
    # (x+1)*logical_and(-1 <= x,x < 0) + (1-x)*logical_and(0 <= x,x <= 1)
    A = x + 1
    ls = x<0
    A *= ls
    A *= less_equal(-1,x,out=ls)
    B = 1 - x
    B *= less_equal(0,x,out=ls)
    B *= less_equal(x,1,out=ls)
    A += B
    return A
def lanczos2(x):
    """See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990, pp. 156-157."""
    from numpy import sin, pi, abs # pylint: disable=redefined-builtin
    abs(x, out=x)
    c = x<2
    x *= pi
    return (sin(x)*sin(0.5*x)+__eps)/((0.5*x*x)+__eps)*c
def lanczos3(x):
    """See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990, pp. 157-158."""
    from numpy import sin, pi, abs # pylint: disable=redefined-builtin
    abs(x, out=x)
    c = x<3
    x *= pi
    return (sin(x)*sin(1/3*x)+__eps)/((1/3*x*x)+__eps)*c
