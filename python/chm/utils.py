"""
Utility functions used by the CHM segmentation algorithm.

These are based on the MATLAB functions originally used along with a few new ones.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function


########## Conversion ##########
def im2double(im, out=None, region=None, nthreads=1):
    """
    Converts an image to doubles from 0.0 to 1.0 if not already a floating-point type.

    To match the filters, it supports out, region, and nthreads arguments. However, nthreads is
    mostly ignored.
    """
    from numpy import float64, iinfo
    if region is not None: im = im[region[0]:region[2], region[1]:region[3]]
    # OPT: more use of nthreads (astype, /, -)
    if out is None: out = im.astype(float64, copy=False)
    else:           copy(out, im, nthreads)
    # NOTE: The divisions here could be pre-calculated for ~60% faster code but this is always the
    # first step and the errors will propogate (although minor, max at ~1.11e-16 or 1/2 EPS and
    # averaging ~5e-18). This will only add a milisecond or two per 1000x1000 block.
    # TODO: If I ever extend "compat" mode to this function, it would be a good candidate.
    k, t = im.dtype.kind, im.dtype.type
    if k == 'u': out /= iinfo(t).max
    elif k == 'i':
        ii = iinfo(t)
        out -= ii.min
        out /= ii.max - ii.min
    elif k not in 'fb': raise ValueError('Unknown image format')
    return out


########## Resizing Image ##########
def MyUpSample(im, L, out=None, region=None, nthreads=1):
    """
    Increases the image size by 2**L. So if L == 0, image is returned unchanged, if L == 1 the
    image is doubled, and so forth. The upsampling is done with no interpolation (nearest
    neighbor).

    To match the filters, it supports out, region, and nthreads arguments.
    """
    # CHANGED: only supports 2D
    from numpy.lib.stride_tricks import as_strided
    if region is not None: im = im[region[0]:region[2], region[1]:region[3]]
    if L == 0:
        if out is None: return im.view()
        copy(out, im, nthreads)
        return out
    N = 1<<L
    H,W = im.shape[:2]
    sh = H, N, W, N
    im = as_strided(im, sh, (im.strides[0], 0, im.strides[1], 0))
    if out is None: return im.reshape((H*N, W*N))
    copy_flat(out, im, nthreads)
    return out

def MyDownSample(im, L, out=None, region=None, nthreads=1):
    """
    Decreases the image size by 2**L. So if L == 0, image is returned unchanged, if L == 1 the
    image is halved, and so forth. The downsampling uses bicubic iterpolation. If the image size is
    not a multiple of 2**L then extra rows/columns are added to make it so by replicating the edge
    pixels.

    To match the filters it supports out, region, and nthreads arguments. This can be done in-place
    where out and im overlap.
    """
    # CHANGED: only supports 2D although the underlying imresize_fast supports 3D
    from .imresize import imresize_fast # built exactly for our needs
    if region is not None: im = im[region[0]:region[2], region[1]:region[3]]
    if L == 0:
        if out is None: return im.view()
        copy(out, im, nthreads)
        return out
    im = pad_to_even(im, nthreads)
    if L==1: return imresize_fast(im, out, nthreads)
    return MyDownSample(imresize_fast(im, None, nthreads), L-1, out, None, nthreads)

def MyMaxPooling(im, L, out=None, region=None, nthreads=1):
    """
    Decreases the image size by 2**L. So if L == 0, image is returned unchanged, if L == 1 the
    image is halved, and so forth. The downsampling uses max-pooling.

    To match the filters it supports out, region, and nthreads arguments.
    """    
    # CHANGED: only supports 2D (the 3D just did each layer independently [but all at once])
    # Using Cython, ~4-5 times faster than blockviews and is parallelized, and ~16-20 faster than original
    from numpy import empty
    from ._utils import max_pooling #pylint: disable=no-name-in-module
    if region is not None: im = im[region[0]:region[2], region[1]:region[3]]
    if L == 0:
        if out is None: return im.view()
        copy(out, im, nthreads)
        return out
    sz = 2 << (L-1)
    sh = ((im.shape[0]+sz-1)>>L, (im.shape[1]+sz-1)>>L)
    if out is None: out = empty(sh, dtype=im.dtype)
    elif out.shape != sh or out.dtype != im.dtype: raise ValueError('Invalid output array')
    if im.dtype == bool:
        # Cython doesn't handle boolean arrays very well, so view it as unsigned chars
        max_pooling['npy_bool'](im.astype('u1', copy=False), L, out.astype('u1', copy=False), nthreads)
    else: max_pooling(im, L, out, nthreads)
    return out


########## Extracting and Padding Image ##########
def get_image_region(im, padding=0, region=None, mode='symmetric', nthreads=1):
    """
    Gets the desired subregion of an image with the given amount of padding. If possible, the
    padding is taken from the image itself. If not possible, the pad function is used to add the
    necessary padding.
        padding is the amount of extra space around the region that is desired
        region is the portion of the image we want to use, or None to use the whole image
            given as top, left, bottom, right - negative values do not go from the end of the axis
            like normal, but instead indicate before the beginning of the axis; the right and
            bottom values should be one past the end just like normal
        mode is the padding mode, if padding is required, and defaults to symmetric
        ntheads is the number of threads used during any padding operations and defaults to 1
    Besides returning the image, a new region is returned that is valid for the returned image to be
    processed again.
    """
    if region is None:
        region = (padding, padding, padding + im.shape[0], padding + im.shape[1]) # the new region
        if padding == 0: return im, region
        return fast_pad(im, padding, mode, nthreads), region
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
        return fast_pad(im[T:B, L:R], padding, mode, nthreads), region
    return im[T:B, L:R], region

def replace_sym_padding(im, padding, region=None, full_padding=None, nthreads=1):
    """
    Takes a region of an image that may have symmetric padding added to it and replaces the
    symmetric padding with 0s. In any situation where 0 padding is added, the image data is copied.
    In this situation, sides that do not have 0 padding added are copied with up to full_padding
    pixels from the original image data. The symmetric padding must originate right at the edge of
    the region to be detected. Also supports nthreads argument.
    
    If full_padding > padding, this will sometimes use symmetric padding to fill in the gap between
    the two.

    Returns image and region, like `get_image_region`.
    """
    if region is None:
        # Using entire image, just add the constant padding
        return get_image_region(im, padding, region, 'constant', nthreads)
    
    # Region indices
    T,L,B,R = region #pylint: disable=unpacking-non-sequence
    max_pad = (T, L, im.shape[0]-B, im.shape[1]-R)
    if not any(max_pad):
        # Region specifies the entire image, just add the constant padding
        return get_image_region(im, padding, region, 'constant', nthreads)

    # Number of pixels available for padding on each side
    if full_padding is None: full_padding = padding
    pT,pL,pB,pR = (min(mp, padding) for mp in max_pad)
    im_pad = im[T-pT:B+pB, L-pL:R+pR]
    
    # Check if the padding is a symmetric reflection of the data
    zs = (__is_sym(im_pad,       pT), __is_sym(im_pad.T,       pL),
          __is_sym(im_pad[::-1], pB), __is_sym(im_pad.T[::-1], pR))

    # If all sides have symmetrical padding (or no padding), just 0 pad all sides
    if all(zs): return get_image_region(im[T:B,L:R], padding, None, 'constant', nthreads)
    
    # If no sides have symmetrical padding, just extract the image region
    if not any(zs): return get_image_region(im, full_padding, region, 'constant', nthreads)
    
    # At this point `zs` has at least one True and one False
    # Each side that has a True in `zs` needs padding of size `padding` full of 0s
    # The other sides need image data/symmetic padding of size `full_padding`
    pT,pL = ((padding if z else full_padding) for z in zs[:2])
    return __replace_sym_padding(im, zs, padding, full_padding, region, nthreads), (pT, pL, pT+(B-T), pL+(R-L))
    
def __replace_sym_padding(im, zs, padding, full_padding, region, nthreads):
    """
    Internal function for replace_sym_padding for dealing with the most difficult case when some
    sides need zero padding and some sides need image or symmetic padding. The `zs` argument should
    have at least one True and one False. Each side that has a True in `zs` needs padding of size
    `padding` full of 0s and the other sides need image/symmetic padding of size `full_padding`.
    Returns just the newly created array and not the region
    """
    # Get image region and various paddings
    from itertools import izip
    T,L,B,R = region
    max_pad = (T, L, im.shape[0]-B, im.shape[1]-R)
    ips = [(0 if z else min(full_padding,p)) for z,p in izip(zs, max_pad)] # image padding
    sps = [(0 if z else full_padding-p) for z,p in izip(zs, ips)] # symmetric padding
    xps = [padding if z else p for z,p in izip(zs,sps)] # non-image padding
    
    # Create a 0-padded copy of image
    out = fast_pad(im[T-ips[0]:B+ips[2],L-ips[1]:R+ips[3]], (xps[::2],xps[1::2]), 'constant', nthreads)
    # Now add symmetric padding between padding and full_padding
    if any(sps): __fill_symmetric_padding(out, sps, False)
    # Return output
    return out
    
def __is_sym(im, n_pad):
    """
    Returns True if the top `n_pad` pixels of `im` are a symmetric reflection of the remainder of
    the image. The left, bottom, or right edges can be checked using transposing and/or flipping.
    This also returns True is `n_pad` is 0.
    """
    # Get the padding and image portions of `im`
    n_im  = im.shape[0] - n_pad
    assert n_im > 0
    im_orig, pad, im = im, im[:n_pad], im[n_pad:]
    
    # In situations where the image is smaller than the padding, loop through expanding the search each time
    while n_pad > n_im:
        if (pad[:-n_im-1:-1] != im).any(): return False
        n_pad -= n_im; n_im += n_im
        pad, im = im_orig[:n_pad], im_orig[n_pad:]
    # Final check
    return (pad[::-1] == im[:n_pad]).all()


########## General Utilities ##########
from ._utils import par_hypot as hypot #pylint: disable=no-name-in-module, unused-import
def copy(dst, src, nthreads=1):
    """
    Equivilent to numpy.copyto(dst, src, 'unsafe', None) but is parallelized with the given number
    of threads. Note however that since copying is such a fast operation in most cases, the
    threshold for number of threads is quite high so this will frequently just use copyto without
    threading.
    """
    if dst.shape != src.shape: raise ValueError('dst and src must be same shape')
    if nthreads <= 1 or dst.size < 200000000:
        from numpy import copyto
        copyto(dst, src, 'unsafe')
    else:
        from ._utils import par_copy #pylint: disable=no-name-in-module
        par_copy(dst, src, nthreads)
    
def copy_flat(dst, src, nthreads=1):
    """
    Copies all of the data from one array to another array of the same size even if they don't have
    the same shape. This is performed with only a small amount of extra copying and memory.
    """
    try:
        # If the shape is directly convertible without creating a copy
        dst = dst.view()
        dst.shape = src.shape
        copy(dst, src, nthreads)
    except AttributeError:
        from ._utils import par_copy_any #pylint: disable=no-name-in-module
        par_copy_any(dst, src, nthreads)
        # Could also be done with nditers which appear to be slightly faster but do require more
        # memory and cannot be made parallel nearly as easily.
        #from numpy import nditer
        #from itertools import izip
        #if src.size != dst.size: raise ValueError("arrays have different size.")
        #ii = nditer(src, order='C', flags=['zerosize_ok', 'buffered', 'external_loop'])
        #oi = nditer(dst, order='C', flags=['zerosize_ok', 'buffered', 'external_loop', 'refs_ok'], op_flags=["writeonly"])
        #for i, o in izip(ii, oi): o[...] = i

def pad_to_even(X, nthreads=1):
    """
    Equivilent to:
        if any((n&1) == 1 for n in X.shape):
            return np.pad(X, [(0,n&1) for n in X.shape], mode='edge')
        else:
            return X.view()
    with the bulk of the work possibily parallelized and even when not it is ~8.5x faster.
    """
    from numpy import empty
    
    # Get the new shape and check that it will change
    new_sh = [(n+(n&1)) for n in X.shape]
    if new_sh == X.shape: return X.view()
    
    # Allocate the output array
    out = empty(new_sh, dtype=X.dtype)
    
    # Copy the core over directly
    copy(out[[slice(n) for n in X.shape]], X, nthreads)
    
    # Perform edge padding
    dst = [slice(0, n) for n in X.shape]
    src = [slice(0, n) for n in X.shape]
    for i,n in enumerate(X.shape):
        if n&1 != 1: continue
        dst[i],src[i] = n,n-1
        out[dst] = out[src]
        dst[i],src[i] = slice(None),slice(None)
    return out

__pad_modes = frozenset(('constant', 'edge', 'reflect', 'symmetric'))
def fast_pad(X, pad_width, mode, nthreads=1, constant_values=0, reflect_type='even'):
    """
    Equivilent to:
        return np.pad(X, pad_width, mode)
    with the bulk of the work possibily parallelized and even when not it is much faster and only
    the following modes are supported:
     * 'constant'   ('constant_values' keyword is supported)
     * 'edge'
     * 'reflect'    ('even' type only)
     * 'symmetric'  ('even' type only)
    """
    from itertools import izip
    from numpy import empty
    
    # Check arguments
    if mode not in __pad_modes:
        raise ValueError('Unknown mode '+mode)
    pad_width = __get_all_values('pad_width', pad_width, X.ndim)
    if mode == 'constant':
        constant_values = __get_all_values('constant_values', constant_values, X.ndim, X.dtype.type)
    elif mode == 'edge': pass # no extra arguments
    else: # if mode in ('reflect', 'symmetric'):
        if reflect_type not in ('even', 'odd'): raise ValueError('Unknown reflect_type '+reflect_type)
        if reflect_type == 'odd': raise ValueError('reflect_type of odd not supported')
    
    # Create the output and fill in the bulk of the data
    new_sh = [(n+p1+p2) for n,(p1,p2) in izip(X.shape, pad_width)]
    out = empty(new_sh, dtype=X.dtype)
    copy(out[[slice(p1,p1+n) for n,(p1,p2) in izip(X.shape, pad_width)]], X, nthreads)

    # Fill in padding
    # Not done in parallel since it is assumed to be a small amount compared to the core. However
    # could make it 'choose' if copy(A[...], B[...]) is used instead of A[...] = B[...] (but this
    # doesn't help constant padding which uses the fill function).
    if mode == 'constant':
        __fill_constant_padding(out, pad_width, constant_values)
    elif mode == 'edge':
        __fill_edge_padding(out, pad_width)
    else: # if mode in ('reflect', 'symmetric'):
        __fill_symmetric_padding(out, pad_width, mode == 'reflect')
    return out
def __fill_constant_padding(X, pad_width, constant_values):
    from itertools import izip
    slices = [slice(p1, n-p2) for n,(p1,p2) in izip(X.shape, pad_width)]
    for i,(n,(p1,p2),(c1,c2)) in enumerate(izip(X.shape, pad_width, constant_values)):
        if p1 != 0:
            slices[i] = slice(p1)
            X[slices].fill(c1)
        if p2 != 0:
            slices[i] = slice(n-p2, n)
            X[slices].fill(c2)
        slices[i] = slice(None)
def __fill_edge_padding(X, pad_width):
    from itertools import izip
    from numpy.lib.stride_tricks import as_strided
    dst = [slice(p1, n-p2) for n,(p1,p2) in izip(X.shape, pad_width)]
    src = [slice(p1, n-p2) for n,(p1,p2) in izip(X.shape, pad_width)]
    def repeat_dim(X, dim, n):
        sh = list(X.shape);   sh.insert(dim, n)
        st = list(X.strides); st.insert(dim, 0)
        return as_strided(X, sh, st)
    for i,(n,(p1,p2)) in enumerate(izip(X.shape, pad_width)):
        if p1 != 0:
            dst[i],src[i] = slice(0, p1), p1
            X[dst] = repeat_dim(X[src], i, p1)
        if p2 != 0:
            dst[i],src[i] = slice(n-p2, n), n-p2-1
            X[dst] = repeat_dim(X[src], i, p2)
        dst[i],src[i] = slice(None), slice(None)
def __fill_symmetric_padding(X, pad_width, reflect):
    from itertools import izip
    off = 1 if reflect else 0 # do not duplicate edge when reflecting
    dst = [slice(p1, n-p2) for n,(p1,p2) in izip(X.shape, pad_width)]
    src = [slice(p1, n-p2) for n,(p1,p2) in izip(X.shape, pad_width)]
    for i,(n,(p1,p2)) in enumerate(izip(X.shape, pad_width)):
        n = n-p1-p2 # unpadded size
        while n-off < p1:
            # Copy n-off at a time until we have enough to do p1 all at once
            dst[i],src[i] = slice(p1-n+off, p1), slice(p1+n-1, p1-1+off, -1)
            X[dst] = X[src]
            p1 -= n-off; n += n-off
        if p1 != 0:
            # Do all of p1 at once
            dst[i],src[i] = slice(0, p1), slice(2*p1-1+off, p1-1+off, -1)
            X[dst] = X[src]
        while n-off < p2:
            # Copy n-off at a time until we have enough to do p2 all at once
            dst[i],src[i] = slice(p1+n, p1+2*n-off), slice(p1+n-1-off, p1-1, -1)
            X[dst] = X[src]
            p2 -= n-off; n += n-off
        if p2 != 0:
            # Do all of p2 at once
            dst[i],src[i] = slice(p1+n, p1+n+p2), slice(p1+n-1-off, p1+n-p2-1-off, -1)
            X[dst] = X[src]
        dst[i],src[i] = slice(None), slice(None)
def __get_all_values(name, val, ndim, cast=int):
    from collections import Sequence
    if not isinstance(val, Sequence):
        # Single number
        return ((cast(val),cast(val)),)*ndim
    if len(val) == 0:
        # Only could be 0-dim
        if ndim == 0: return ()
        raise ValueError('Wrong number of items for '+name)
    if not isinstance(val[0], Sequence):
        # Must be (val,)
        if len(val) == 1: return ((cast(val[0]), cast(val[0])),)*ndim
        raise ValueError('Wrong number of items for '+name)
    if len(val) == 1:
        # Must be ((before,after),)
        if len(val[0]) == 2: return ((cast(val[0][0]), cast(val[0][1])),)*ndim
        raise ValueError('Wrong number of items for '+name)
    if len(val) != ndim or any(len(x) != 2 for x in val):
        raise ValueError('Wrong number of items for '+name)
    return tuple((cast(b),cast(a)) for b,a in val)


########## FFT Utility ##########
__reg_nums = {}
def next_regular(target):
    """
    Finds the next regular/Hamming number (all prime factors are 2, 3, and 5).
    
    Implements scipy.fftpack.helper.next_fast_len / scipy.signal.signaltools._next_regular except
    that it is the newer, optimized version which is only in SciPy >= 0.18 with an additional
    caching mechanism for larger values.
    """
    from bisect import bisect_left
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72,
            75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200,
            216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432,
            450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800,
            810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296,
            1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025,
            2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000,
            3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000, 4050, 4096,
            4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400, 5625, 5760, 5832, 6000,
            6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000,
            8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)
    if target <= 6 or not (target & (target-1)): return target
    if target <= hams[-1]: return hams[bisect_left(hams, target)]
    if target in __reg_nums: return __reg_nums[target]
    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            N = (2**((-(-target // p35) - 1).bit_length())) * p35
            if N == target: __reg_nums[target] = N; return N
            elif N < match: match = N
            p35 *= 3
            if p35 == target: __reg_nums[target] = p35; return p35
        if p35 < match: match = p35
        p5 *= 5
        if p5 == target: __reg_nums[target] = p5; return p5
    if p5 < match: match = p5
    __reg_nums[target] = match
    return match


########## Set Library Threads ##########
def set_lib_threads(nthreads):
    """
    Sets the maximum number of used threads used for various computational libraries, in particular
    for the OpenMP, BLAS, and MKL libraries. If this cannot determine the location of the
    libraries, it does nothing.
    """
    global __set_num_thread_funcs, __last_set_num_threads #pylint: disable=global-statement
    if __set_num_thread_funcs is None:
        import os, sys
        __set_num_thread_funcs = []
        if   sys.platform == 'win32':  __init_set_library_threads_win32()
        elif sys.platform == 'cygwin': __init_set_library_threads_cygwin()
        elif sys.platform == 'darwin': __init_set_library_threads_darwin()
        elif os.name == 'posix':       __init_set_library_threads_posix()
    nthreads = int(nthreads)
    if __last_set_num_threads is None or __last_set_num_threads != nthreads:
        for f in __set_num_thread_funcs:
            try: f(nthreads)
            except OSError: pass
    __last_set_num_threads = nthreads
        
__set_num_thread_funcs = None
__last_set_num_threads = None
def __add_set_num_threads_func(dll, func='omp_set_num_threads', ref=False):
    """
    Adds a C function to the __set_num_thread_funcs list. The function comes the given `dll` (which
    is sent to ctypes.utils.find_library if it does not already have an extension) called `func`
    (which defaults to 'omp_set_num_threads'). If `ref` is True, then the function takes a pointer
    to an int instead of just an int. Return True if the function was added, False otherwise.
    """
    import os.path, ctypes, ctypes.util
    if '.' in dll: path = dll
    else:
        path = ctypes.util.find_library(dll)
        if path is None: return False
        if not os.path.isabs(path) and hasattr(ctypes.util, '_findLib_gcc'):
            # If the library was found on LIBRARY_PATH but not LD_LIBRARY_PATH the absolute path is removed...
            path = ctypes.util._findLib_gcc(dll) #pylint: disable=protected-access
    try: f = getattr(getattr(getattr(ctypes, 'windll', ctypes.cdll), path), func)
    except (AttributeError, OSError): return False
    f.restype = None
    f.argtypes = (ctypes.POINTER(ctypes.c_int),) if ref else (ctypes.c_int,)
    __set_num_thread_funcs.append(f)
    return True

def __init_set_library_threads_win32():
    """
    Adds libiomp5md.dll for Intel MKL OpenMP, libgomp-1.dll for MingW, and vcomp###.dll for MSVC.
    """
    import sys
    __add_set_num_threads_func('libiomp5md')
    if 'GCC' in sys.version: __add_set_num_threads_func('libgomp-1')
    elif 'MSC' in sys.version:
        import re
        msc_vers = {
            # Mapping of _MSC_VER to runtime version number (msvcr###.dll)
            # Below v1400 there is no OpenMP support (and there is no v13.0)
            # Using this table is much faster than __win_get_msvcr_ver
            1400:80, 1500:90, 1600:100, 1700:110, 1800:120, 1900:140, # '05, '08, '10, '12, '13, '15
        }
        try:
            v = int(re.search(r'MSC v.(\d+) ', sys.version).group(1))
            v = msc_vers[v] if v in msc_vers else __win_get_msvcr_ver('python27')
            __add_set_num_threads_func('vcomp%d'%v)
        except (AttributeError, ValueError): pass

def __win_get_msvcr_ver(name):
    """
    Searches the imports of the DLL with the given name for a DLL like MSVCR###.dll and returns
    the ### as an integer. Raises a ValueError if it cannot be found.
    """
    #pylint: disable=no-name-in-module
    from ctypes import c_char_p, sizeof, cast, POINTER
    from ctypes.wintypes import DWORD, LONG
    try:
        from win32api import LoadLibraryEx, FreeLibrary, error
        from win32con import LOAD_LIBRARY_AS_DATAFILE
    except ImportError: raise ValueError()
    import re
    try: lib = LoadLibraryEx(name, None, LOAD_LIBRARY_AS_DATAFILE)
    except error: raise ValueError()
    try:
        LPDWORD = POINTER(DWORD)
        off = cast(lib+60, POINTER(LONG))[0]
        imp_data_dir = cast(lib+off+(144 if (sizeof(c_char_p)==8) else 128), LPDWORD)
        off = lib+imp_data_dir[0]+12
        rx = re.compile(r'^MSVCR(\d+).dll$', re.I)
        for off in xrange(off, off+imp_data_dir[1], 20):
            off = cast(off, LPDWORD)[0]
            if off == 0: break
            m = rx.match(cast(lib+off, c_char_p).value)
            if m is not None: return int(m.group(1))
        raise ValueError()
    finally: FreeLibrary(lib)

def __init_set_library_threads_cygwin():
    """Adds cyggomp-1.dll for GNU OpenMP along with OpenBLAS libraries."""
    __add_set_num_threads_func('cyggomp-1.dll')
    __add_set_num_threads_func('openblas', 'goto_set_num_threads')
    __add_set_num_threads_func('openblas', 'openblas_set_num_threads')

def __init_set_library_threads_darwin():
    """
    Adds libiomp5.so for Intel MKL and libgomp.so for GNU OpenMP along with a method for setting the
    number of threads for vecLib in the Apple Accelerate Library.
    """
    import os
    __add_set_num_threads_func('iomp5')
    __add_set_num_threads_func('gomp')
    def set_veclib_max_threads(nthreads): os.environ['VECLIB_MAXIMUM_THREADS']=str(nthreads)
    __set_num_thread_funcs.append(set_veclib_max_threads)

def __init_set_library_threads_posix():
    """
    Adds libiomp5.so for Intel MKL and libgomp.so for GNU OpenMP. Also searches for the MKL library
    in various forms and OpenBLAS libraries.
    """
    __add_set_num_threads_func('iomp5')
    __add_set_num_threads_func('gomp')
    __add_set_num_threads_func('mkl.so', 'mkl_serv_set_num_threads')
    __add_set_num_threads_func('mkl_rt', 'MKL_Set_Num_Threads', True)
    __add_set_num_threads_func('openblas', 'goto_set_num_threads')
    __add_set_num_threads_func('openblas', 'openblas_set_num_threads')
    __add_set_num_threads_func('openblas64', 'goto_set_num_threads')
    __add_set_num_threads_func('openblas64', 'openblas_set_num_threads')
