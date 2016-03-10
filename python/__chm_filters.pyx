#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: language=c++
#distutils: sources=HoG.cpp HoG-orig.cpp

"""
CHM Filters written in Cython or C. This includes Haar, HoG, parts of SIFT, and optimized
correlation functions. Haar and HoG were originally in MEX (MATLAB's Cython-like system)
and C++. SIFT was originally in MATLAB but the last step was converted to Cython because
it was slow. The correlations are adapated from scipy's ndimage correlate1d.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

# When True uses Capsules intead of CObjects
# This should only be set to True in Python 3.2+ when SciPy switches over to using Capsules for function wrappers
DEF USE_PYCAPSULES = False

include "npy_helper.pxi"

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport abs, fabs, ceil, floor, sqrt, cos, sin, M_PI
from libc.float cimport DBL_EPSILON

from cython.parallel cimport prange, parallel, threadid
from cython cimport view
from openmp cimport *

cdef ndarray __get_out(ndarray out, intp N, intp H, intp W):
    cdef intp[3] dims
    if out is None:
        dims[0] = N; dims[1] = H; dims[2] = W
        return PyArray_EMPTY(3, dims, NPY_DOUBLE, False)
    elif not PyArray_ISBEHAVED(out) or PyArray_TYPE(out) != NPY_DOUBLE or PyArray_NDIM(out) != 3 or \
         PyArray_DIM(out, 0) != N or PyArray_DIM(out, 1) != H or PyArray_DIM(out, 2) != W or \
         PyArray_STRIDE(out, 1) != <intp>(W*sizeof(double)) or PyArray_STRIDE(out, 2) != sizeof(double):
        raise ValueError('Invalid output array')
    return out

cdef ndarray __get_out_2d(ndarray out, intp H, intp W):
    cdef intp[2] dims
    if out is None:
        dims[0] = H; dims[1] = W
        return PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
    elif not PyArray_ISBEHAVED(out) or PyArray_TYPE(out) != NPY_DOUBLE or PyArray_NDIM(out) != 2 or \
         PyArray_DIM(out, 0) != H or PyArray_DIM(out, 1) != W or PyArray_STRIDE(out, 1) != sizeof(double):
        raise ValueError('Invalid output array')
    return out

cdef inline int __get_nthreads(int nthreads, intp max_threads) nogil:
    # equivilent to max(min(nthreads, max_threads, omp_get_max_threads()), 1)
    if nthreads > max_threads: nthreads = <int>max_threads
    if nthreads > omp_get_max_threads(): nthreads = omp_get_max_threads()
    return 1 if nthreads < 1 else nthreads


########## Neighborhood Calculator ##########

def ConstructNeighborhoods(ndarray im, ndarray offsets, ndarray out=None, tuple region=None, int nthreads=1):
    # CHANGE: permanently now like ConstructNeighborhoods(padReflect(im, *), ____Neighborhood(*), 0)
    # CHANGE: region is handled in such a way that the image is never padded, but instead reflection is handled directly
    # CHANGE: converted to Cython so that it could be multi-threaded and faster

    # TODO: all DOUBLE_PTR_R and DOUBLE_PTR_CR should be aligned (A) but doing so causes illegal argument core dumps...

    # Check image and offsets
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2 or PyArray_STRIDE(im, 1) != sizeof(double): raise ValueError("Invalid im")
    if not PyArray_ISBEHAVED_RO(offsets) or PyArray_TYPE(offsets) != NPY_INTP or PyArray_NDIM(offsets) != 2 or PyArray_DIM(offsets, 0) != 2 or PyArray_STRIDE(offsets, 1) != sizeof(intp): raise ValueError("Invalid offsets")
    cdef intp i, n_offs = PyArray_DIM(offsets, 1), im_stride = PyArray_STRIDE(im, 0)
    
    # Get the image region and offset radii
    cdef intp H, W, T, L, B, R, T_rad, L_rad, B_rad, R_rad
    if region is None:
        T, L, B, R = 0, 0, 0, 0 # everything outside needs reflection
        H, W = PyArray_DIM(im, 0), PyArray_DIM(im, 1)
    else:
        T, L, B, R = region
        H, W = B - T, R - L
        B, R = PyArray_DIM(im, 0) - B, PyArray_DIM(im, 1) - R
    L_rad, T_rad = -PyArray_Min(offsets, 1, NULL)
    R_rad, B_rad =  PyArray_Max(offsets, 1, NULL)
    if L_rad < 0 or R_rad < 0 or T_rad < 0 or B_rad < 0: raise ValueError("Offsets cannot be purely positive or negative")

    # Make sure we have somewhere to save to
    out = __get_out(out, n_offs, H, W)
    cdef intp out_stride = PyArray_STRIDE(out, 0)

    # Get the image rows, including the reflected ones
    cdef intp n_rows = H+T_rad+B_rad
    cdef ndarray im_rows = PyArray_EMPTY(1, &n_rows, NPY_INTP, False)
    cdef const char** irp = <const char**>PyArray_DATA(im_rows)
    cdef const char* im_p = <const char*>PyArray_DATA(im)
    cdef pre_y  = __clip_neg( T_rad-T)
    cdef off_y  = __clip_neg(-T_rad+T)
    cdef post_y = __clip_neg( B_rad-B)
    cdef mid_y  = H - pre_y - post_y + T_rad + B_rad + off_y
    for i in xrange(pre_y-1, -1, -1):             irp[0] = im_p+i*im_stride; irp += 1 # reflected start
    for i in xrange(off_y, mid_y):                irp[0] = im_p+i*im_stride; irp += 1
    for i in xrange(mid_y-1, mid_y-post_y-1, -1): irp[0] = im_p+i*im_stride; irp += 1 # reflected end

    #irp = <const char**>PyArray_DATA(im_rows)
    #for i in xrange(n_rows):
    #    print("%08x" % <intp>irp[i])

    # Get pointers
    cdef const DOUBLE_PTR_CR* im_rows_p = (<const DOUBLE_PTR_CR*>PyArray_DATA(im_rows)) + T_rad
    cdef const intp* xs = <const intp*>PyArray_DATA(offsets)
    cdef const intp* ys = xs + PyArray_STRIDE(offsets, 0) // sizeof(intp)
    cdef char* out_p = <char*>PyArray_DATA(out)
    R = L_rad-R_rad-R # TODO: test this when L_rad != R_rad to make sure this is right...

    # Copy every offset
    for i in prange(n_offs, nogil=True, schedule='static', num_threads=nthreads):
        __cpy_off(<DOUBLE_PTR_R>(out_p + i*out_stride), im_rows_p + ys[i], W, H, xs[i], L, R)
    return out

cdef inline void __cpy_off(DOUBLE_PTR_R out, const DOUBLE_PTR_CR* im, const intp W, const intp H, const intp X, const intp L, const intp R) nogil:
    cdef intp i, j
    cdef intp pre_x  = __clip_neg(-X-L)
    cdef intp off_x  = __clip_neg( X+L)
    cdef intp post_x = __clip_neg( X+R)
    cdef intp mid_x  = W - pre_x - post_x
    cdef intp end_x  = mid_x-post_x-1
    cdef DOUBLE_PTR_CR row
    for i in xrange(H):
        row = im[0]; im = im + 1
        for j in xrange(pre_x-1, -1, -1): out[0] = row[j]; out += 1 # first few reflected
        row += off_x; memcpy(out, row, mid_x*sizeof(double)); out += mid_x # middle straight copied
        for j in xrange(mid_x-1, end_x, -1): out[0] = row[j]; out += 1 # last few reflected

cdef inline intp __clip_neg(intp x) nogil: return 0 if x < 0 else x


########## Correlation Functions ##########

INPLACE = ndarray((0,0))

def correlate_xy(ndarray im, ndarray weights_x, ndarray weights_y, ndarray out=None, int nthreads=1):
    """
    Run a separable 2D cross correlation, with the filter already separated into weights_x and
    weights_y.
    Equivilent to:
        scipy.ndimage.correlate(im, weights_x[:,None].dot(weights_y[None,:]), output=out)
    With the following differences:
        * edges are dropped instead of being reflected, constant, ... (so the output is equal to
          out[r:-r,r:-r])
        * im must be a 2D double, behaved, C-contiguous, with the final stride being equal to the
          size of a double
        * weights must be a 1D double vector with the final stride being equal to the size of a
          double
        * this supports multi-threading (although defaults to a single thread)
        * partial in-place operation
    
    The out argument can be one of the following:
        * None, in which case a single array is allocated and used as both the termporary and final
          output
        * an array that is the width of the input but the height of the output, in which case it is
          used for the temporary and output (no major allocations)
              Note that if the output provided is a view of the input, an additional allocation and
              copy will be made, making it slower than the following two options (which don't have
              the extra copy even when the output is a view of the input)
        * an array that is exactly the output size, in which case an array is allocated for the
          temporary
        * the special value INPLACE in which case an array is allocated for the temporary but the
          final output is to the input
    """
    cdef ndarray out_x, out_y
    cdef intp[2] dims
    if PyArray_NDIM(im) != 2: raise ValueError("Invalid im")
    if PyArray_NDIM(weights_x) != 1 or PyArray_NDIM(weights_y) != 1: raise ValueError("Invalid weights")
    cdef intp H_in = PyArray_DIM(im, 0), W_in = PyArray_DIM(im, 1)
    cdef intp fs_y = PyArray_DIM(weights_y, 0) - 1, fs_x = PyArray_DIM(weights_x, 0) - 1
    cdef intp H_out = H_in - fs_y, W_out = W_in - fs_x
    if out is None:
        # Temporary and Output are allocated as a single array
        dims[0] = H_out; dims[1] = W_in
        out_y = PyArray_EMPTY(2, dims, NPY_DOUBLE, False) 
        out_x = __view_trim2D(out_y, 0, 0, 0, fs_x)
    elif out is INPLACE:
        # Temporary is allocated and final output is to the input
        dims[0] = H_out; dims[1] = W_in
        out_y = PyArray_EMPTY(2, dims, NPY_DOUBLE, False) 
        out_x = __view_trim2D(out, 0, fs_y, 0, fs_x)
    elif PyArray_NDIM(out) != 2: raise ValueError("Invalid output array")
    elif PyArray_DIM(out, 0) == H_out and PyArray_DIM(out, 0) == W_out:
        # Temporary is allocated and final output is to the output
        dims[0] = H_out; dims[1] = W_in
        out_y = PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
        out_x = out
    elif PyArray_DIM(out, 0) == H_out and PyArray_DIM(out, 0) == W_in:
        # The output can be the temporary and is also the final output (no allocation)
        out_y = out
        out_x = __view_trim2D(out, 0, 0, 0, fs_x)
    else:
        raise ValueError("Invalid output array")
    correlate_y(im, weights_y, out_y, nthreads)
    return correlate_x(out_y, weights_x, out_x, nthreads)

cdef int __get_sym(ndarray weights) nogil:
    """
    Checks if a filter is symmetrical (returns 1), anti-symmetrical (returns -1), or other (0).
    """
    cdef intp filter_size = PyArray_DIM(weights, 0)
    if (filter_size & 1) == 0: return 0
    cdef double* fw = <double*>PyArray_DATA(weights)
    cdef intp size = filter_size // 2, size1 = size+1, i
    for i in xrange(1, size):
        if fabs(fw[size + i] - fw[size - i]) > DBL_EPSILON:
            for i in xrange(1, size1):
                if fabs(fw[size + i] + fw[size - i]) > DBL_EPSILON: return 0
            return -1
    return 1

ctypedef void (*correlate_internal)(intp H, intp W,
                                    char* out, intp out_stride,
                                    const double* in_, intp in_stride,
                                    double* tmp, const double* fw, intp size1, intp size2) nogil

def correlate_x(ndarray im, ndarray weights, ndarray out=None, int nthreads=1):
    """
    Run cross correlation in 1D along the "x" axis (axis=1) of a 2D image. Equivilent to:
        scipy.ndimage.correlate1d(im, weights, axis=1, output=out)
    With the following differences:
        * edges are dropped instead of being reflected, constant, ... (so the output is equal to out[:,r:-r])
        * im must be a 2D double, behaved, C-contiguous, with the final stride being equal to the size of a double
        * weights must be a 1D double vector with the final stride being equal to the size of a double
        * this supports multi-threading (although defaults to a single thread)
        * this supports in-place operation so out can be a view of im (e.g. out=im[:, r:-r] or out=im[:, :-2*r] or out=INPLACE)
    """

    cdef intp i, j, k
    
    # Check weights
    if not PyArray_ISCARRAY_RO(weights) or PyArray_TYPE(weights) != NPY_DOUBLE or PyArray_NDIM(weights) != 1: raise ValueError("Invalid weights")
    cdef int symmetric = __get_sym(weights)
    cdef intp filter_size = PyArray_DIM(weights, 0), size1 = filter_size // 2, size2 = filter_size - size1 - 1
    cdef double* fw = <double*>PyArray_DATA(weights) + size1

    # Check im and out
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2 or PyArray_STRIDE(im, 1) != sizeof(double): raise ValueError("Invalid im")
    cdef intp H = PyArray_DIM(im, 0), W = PyArray_DIM(im, 1) - filter_size + 1
    if W < 1: raise ValueError("Filter wider than image")
    if out is INPLACE: out = im[:, size1:-size2]
    out = __get_out_2d(out, H, W)
    cdef intp im_stride = PyArray_STRIDE(im, 0) // sizeof(double), out_stride = PyArray_STRIDE(out, 0)
    nthreads = __get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows
    
    # Get the pointers
    cdef char* out_p = <char*>PyArray_DATA(out)
    cdef double* in_p = <double*>PyArray_DATA(im) + size1
    cdef double* tmp = <double*>malloc(W * sizeof(double) * nthreads)
    if tmp is NULL: raise MemoryError()

    # Get the internal function
    cdef correlate_internal f
    if   symmetric > 0: f = correlate_x_sym
    elif symmetric < 0: f = correlate_x_antisym
    else:               f = correlate_x_any
        
    cdef double inc
    cdef intp a, b
    
    # Run it!
    with nogil:
        if nthreads == 1: f(H, W, out_p, out_stride, in_p, im_stride, tmp, fw, size1, size2)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = H / <double>nthreads # a floating point number, use the floor of adding it together
                i = threadid()
                a = <intp>floor(inc*i)
                b = H if i == nthreads - 1 else (<intp>floor(inc*(i+1)))
                f(b-a, W, out_p+a*out_stride, out_stride, in_p+a*im_stride, im_stride, tmp+i*W, fw, size1, size2)

    free(tmp)
    return out

cdef void correlate_x_sym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double)
    in_stride -= W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k] + in_[-k]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride

cdef void correlate_x_antisym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double)
    in_stride -= W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k] - in_[-k]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride

cdef void correlate_x_any(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double)
    in_stride -= W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[size2] * fw[size2]
            for k in xrange(-size1, size2): tmp[j] += in_[k] * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride

def correlate_y(ndarray im, ndarray weights, ndarray out=None, int nthreads=1):
    """
    Run cross correlation in 1D along the "y" axis (axis=0) of a 2D image. Equivilent to:
        scipy.ndimage.correlate1d(im, weights, axis=0, output=out)
    With the following differences:
        * edges are dropped instead of being reflected, constant, ... (so the output is equal to out[r:-r,:])
        * im must be a 2D double, behaved, C-contiguous, with the final stride being equal to the size of a double
        * weights must be a 1D double vector with the final stride being equal to the size of a double
        * this supports multi-threading (although defaults to a single thread)
    Note: unlike with correlate_x, when operating in-place an internal buffer is allocated for the size of the image
    and the data has to be copied from it to the output, so nothing is saved by working in-place.
    """

    from numpy import may_share_memory
    
    cdef intp i, j, k
    
    # Check weights
    if not PyArray_ISCARRAY_RO(weights) or PyArray_TYPE(weights) != NPY_DOUBLE or PyArray_NDIM(weights) != 1: raise ValueError("Invalid weights")
    cdef int symmetric = __get_sym(weights)
    cdef intp filter_size = PyArray_DIM(weights, 0), size1 = filter_size // 2, size2 = filter_size - size1 - 1
    cdef double* fw = <double*>PyArray_DATA(weights) + size1

    # Check im and out
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2 or PyArray_STRIDE(im, 1) != sizeof(double): raise ValueError("Invalid im")
    cdef intp H = PyArray_DIM(im, 0) - filter_size + 1, W = PyArray_DIM(im, 1)
    if H < 1: raise ValueError("Filter taller than image")
    if out is INPLACE: out = im[size1:-size2, :]
    out = __get_out_2d(out, H, W)
    cdef ndarray real_out = None
    if may_share_memory(im, out):
        real_out = out
        out = PyArray_EMPTY(2, PyArray_SHAPE(out), NPY_DOUBLE, False)
    cdef intp im_stride = PyArray_STRIDE(im, 0) // sizeof(double), out_stride = PyArray_STRIDE(out, 0)
    nthreads = __get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows

    # Get the pointers
    cdef char* out_p = <char*>PyArray_DATA(out)
    cdef double* in_p = <double*>PyArray_DATA(im) + size1 * im_stride
    cdef double* tmp = <double*>malloc(W * sizeof(double) * nthreads)
    if tmp is NULL: raise MemoryError()

    # Get the internal function
    cdef correlate_internal f
    if   symmetric > 0: f = correlate_y_sym
    elif symmetric < 0: f = correlate_y_antisym
    else:               f = correlate_y_any
        
    cdef double inc
    cdef intp a, b
    
    # Run it!
    with nogil:
        if nthreads == 1: f(H, W, out_p, out_stride, in_p, im_stride, tmp, fw, size1, size2)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = H / <double>nthreads # a floating point number, use the floor of adding it together
                i = threadid()
                a = <intp>floor(inc*i)
                b = H if i == nthreads - 1 else (<intp>floor(inc*(i+1)))
                f(b-a, W, out_p+a*out_stride, out_stride, in_p+a*im_stride, im_stride, tmp+i*W, fw, size1, size2)

    free(tmp)
    
    if real_out is not None:
        PyArray_CopyInto(real_out, out)
        out = real_out
    return out
    
cdef void correlate_y_sym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double), in_stride_rem = in_stride - W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k*in_stride] + in_[-k*in_stride]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride_rem

cdef void correlate_y_antisym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double), in_stride_rem = in_stride - W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k*in_stride] - in_[-k*in_stride]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride_rem
        
cdef void correlate_y_any(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double), in_stride_rem = in_stride - W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[size2*in_stride] * fw[size2]
            for k in xrange(-size1, size2): tmp[j] += in_[k*in_stride] * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride_rem


########## Haar Features Calculator ##########

DEF EPS = 0.00001

def cc_cmp_II(ndarray im not None):
    """
    cc_cmp_II
    This function computes the integral image over the input image
    Michael Villamizar
    mvillami@iri.upc.edu
    2009
    Recoded and converted to Cython by Jeffrey Bush 2015

    input:
        <- Input Image
    output:
        -> Integral Image (II)
    """

    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2: raise ValueError("Invalid im")
    cdef intp H = PyArray_DIM(im, 0), W = PyArray_DIM(im, 1)
    cdef ndarray[npy_double, ndim=2, mode='c'] II_arr = PyArray_EMPTY(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
    cdef flatiter _itr = PyArray_IterNew(im)
    cdef PyArrayIterObject* itr = <PyArrayIterObject*>_itr
    cdef DOUBLE_PTR_AR II = <DOUBLE_PTR_AR>PyArray_DATA(II_arr)
    cdef DOUBLE_PTR_AR II_last = II

    cdef double last = 0.0
    cdef intp i, j
    with nogil:
        II[0] = (<DOUBLE_PTR_CAR>PyArray_ITER_DATA(itr))[0]
        PyArray_ITER_NEXT(itr)
        for j in xrange(1, W):
            II[j] = last = (<DOUBLE_PTR_CAR>PyArray_ITER_DATA(itr))[0] + last
            PyArray_ITER_NEXT(itr)
        for i in xrange(1, H):
            II += W
            II[0] = last = (<DOUBLE_PTR_CAR>PyArray_ITER_DATA(itr))[0] + II_last[0]
            PyArray_ITER_NEXT(itr)
            for j in xrange(1, W):
                II[j] = last = (<DOUBLE_PTR_CAR>PyArray_ITER_DATA(itr))[0] + last + II_last[j] - II_last[j-1]
                PyArray_ITER_NEXT(itr)
            II_last += W
    return II_arr

def cc_Haar_features(ndarray II not None, intp S, ndarray out=None, int nthreads=1):
    """
    cc_Haar_features
    This function computes the Haar-like features in X and Y
    Michael Villamizar
    mvillami@iri.upc.edu
    2009
    Recoded, multithreaded, and converted to Cython by Jeffrey Bush 2015
    
    input:
        <- Integral Image (II)
        <- Haar size
    output:
        -> Haar maps : 1) Hx, 2) Hy
    tips:
        * Haar size must be even
    """
    if not PyArray_ISCARRAY_RO(II) or PyArray_TYPE(II) != NPY_DOUBLE or PyArray_NDIM(II) != 2: raise ValueError("Invalid im")
    cdef intp H = PyArray_DIM(II,0)-S, II_W = PyArray_DIM(II,1), W = II_W-S
    out = __get_out(out, 2, H, W)
    nthreads = __get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows
    
    cdef DOUBLE_PTR_AR X = <DOUBLE_PTR_AR>PyArray_DATA(out)             # Haar X output
    cdef DOUBLE_PTR_AR Y = X + PyArray_STRIDE(out, 0) // sizeof(double) # Haar Y output
    cdef DOUBLE_PTR_CAR II_p = <DOUBLE_PTR_CAR>PyArray_DATA(II)

    # Variables used in parallel block
    cdef intp i, a, b
    cdef double inc
    
    with nogil:
        if nthreads == 1: __cc_Haar_features(II_p, H, W, S, X, Y)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = H / <double>nthreads # a floating point number, use the floor of adding it together
                i = threadid()
                a = <intp>floor(inc*i)
                b = H if i == nthreads - 1 else (<intp>floor(inc*(i+1)))
                __cc_Haar_features(II_p+a*II_W, b-a, W, S, X+a*W, Y+a*W)
    return out

cdef void __cc_Haar_features(DOUBLE_PTR_CAR II, intp H, intp W, intp S, DOUBLE_PTR_AR X, DOUBLE_PTR_AR Y) nogil:
    cdef intp i, j, C = S//2
    cdef DOUBLE_PTR_CAR T = II         # top
    cdef DOUBLE_PTR_CAR M = II+C*(W+S) # middle
    cdef DOUBLE_PTR_CAR B = II+S*(W+S) # bottom
    cdef double area
    for i in xrange(0, H):
        for j in xrange(0, W):
            area = B[S] + T[0] - T[S] - B[0]
            if area > 0:
                area = 1/(area+EPS)
                X[0] = (B[S] + B[0] - T[S] - T[0] + 2*(T[C] - B[C]))*area
                Y[0] = (B[S] - B[0] + T[S] - T[0] + 2*(M[0] - M[S]))*area
            else:
                X[0] = 0
                Y[0] = 0
            X += 1; Y += 1
            T += 1; M += 1; B += 1
        T += S; M += S; B += S


########## HoG Features Calculator ##########

cdef extern from "HoG.h" nogil:
    cdef intp HoG_init(const intp w, const intp h, const intp *n)
    cdef void HoG_run(const double *pixels, const intp w, const intp h, double *out, double *H)
    cdef intp _HoG "HoG" (const double *pixels, const intp w, const intp h, double *out, const intp n)
    cdef void _HoG_orig "HoG_orig" (double *pixels, double *params, int *img_size, double *dth_des, bint grayscale)
    
def HoG(ndarray[npy_double, ndim=2, mode='c'] pixels not None, ndarray[npy_double, ndim=1, mode='c'] out not None):
    """
    Implements the HoG filter. Does not support changing the parameters directly (they are compiled in).
    Additionally RGB images are not supported. Instead of allocating the memory for the results, you must
    pass in the output array. The destination array must be a 1-D double array and have at least as many
    elements as are needed for the output. This function returns the number of elements stored in the
    array.
    """
    # CHANGED: the interface is completely changed: no optional params [hard-coded to defaults] and takes the output data array
    # CHANGED: no longer accepts RGB images
    # CHANGED: a much faster method is to bypass this and use ConstructHOG
    if not PyArray_ISBEHAVED_RO(pixels): raise ValueError("Invalid im")
    if not PyArray_ISBEHAVED(out): raise ValueError("Invalid out")
    cdef intp n
    with nogil: n = _HoG(<DOUBLE_PTR_CAR>PyArray_DATA(pixels), PyArray_DIM(pixels, 1), PyArray_DIM(pixels, 0),
                         <DOUBLE_PTR_AR>PyArray_DATA(out), PyArray_DIM(out, 0))
    if n == -1: raise ValueError("Output array too small")
    if n == -2: raise MemoryError()
    return n

def HoG_orig(ndarray pixels not None, params = [9, 8.0, 2, False, 0.2]):
    """
    Implements HoG filter as per the original MEX file. Supports all features of the original and
    uses an almost unmodified C++ source file for it (just compile-time warnings suppression added).
    It prefers Fortran-ordered double arrays. It is 3-4 times slower than the newer version and is
    less accurate due to using 32-bit floats for intermediate calculations.
    """
    cdef double _params[5]
    _params[0] = float(params[0])
    _params[1] = float(params[1])
    _params[2] = float(params[2])
    _params[3] = float(params[3])
    _params[4] = float(params[4])
    
    if params[0] <= 0: raise ValueError("Number of orientation bins must be positive")
    if params[1] <= 0: raise ValueError("Cell size must be positive")
    if params[2] <= 0: raise ValueError("Block size must be positive")
    
    pixels = PyArray_CheckFromAny(pixels, PyArray_DescrFromType(NPY_DOUBLE), 2, 3, NPY_ARRAY_FARRAY_RO|NPY_ARRAY_NOTSWAPPED, NULL)
    cdef bint grayscale = PyArray_NDIM(pixels) == 2
    if not grayscale and PyArray_DIM(pixels, 2) != 3: raise ValueError("Third dimension must be length 3")
    
    cdef int img_size[2]
    img_size[0] = <int>PyArray_DIM(pixels, 0)
    img_size[1] = <int>PyArray_DIM(pixels, 1)

    cdef int nb_bins    = <int>_params[0]    
    cdef int block_size = <int>_params[2]
    cdef int hist1= 2+<int>ceil(-0.5 + img_size[0]/_params[1])
    cdef int hist2= 2+<int>ceil(-0.5 + img_size[1]/_params[1])
    cdef intp n = (hist1-2-(block_size-1))*(hist2-2-(block_size-1))*nb_bins*block_size*block_size
    
    cdef ndarray[npy_double, ndim=1] dth_des = PyArray_EMPTY(1, &n, NPY_DOUBLE, True)
    with nogil:
        _HoG_orig(<double*>PyArray_DATA(pixels), _params, img_size, <double*>PyArray_DATA(dth_des), grayscale)
    return dth_des

##### Highly optimized version for entire image #####
ctypedef void (*filter_func)(DOUBLE_PTR_CAR, const intp, const intp, DOUBLE_PTR_AR, void*) nogil

cdef bint __generic_filter(double[:, :] input,
                           filter_func function, void* data, intp filter_size,
                           double[:, :, ::view.contiguous] output) nogil:
    """
    Similar to SciPy's scipy.ndimage.filters.generic_filter with the following new features:
     * The output has one additional dimension over the input and the filter generates an array of
       data for each pixel instead of a scalar
     * The GIL is no longer required, meaning it can be used multi-threaded
     * Instead of extending the edges of the image only the 'valid' pixels are processed

    To simplify the code, the following features were dropped (but could be re-added if necessary):
     * only accepts 2D double matrices as input
     * the last N axes of output must be C-order
     * the footprint is always a square and origin is always zeros
    """
    cdef intp i, j, x
    cdef intp f_rad = filter_size // 2
    cdef intp stride0 = input.strides[0], stride1 = input.strides[1], out_stride = output.strides[0]
    cdef intp H = input.shape[0] - filter_size + 1, W = input.shape[1] - filter_size + 1
    cdef intp in_sz = filter_size * filter_size, out_sz = output.shape[0]
    
    # Allocate buffers
    cdef INTP_PTR_AR off = <INTP_PTR_AR>malloc(in_sz * sizeof(intp))
    cdef DOUBLE_PTR_AR in_buf = <DOUBLE_PTR_AR>malloc(in_sz * sizeof(double))
    cdef DOUBLE_PTR_AR out_buf = <DOUBLE_PTR_AR>malloc(out_sz * sizeof(double))
    if off is NULL or in_buf is NULL or out_buf is NULL: free(off); free(in_buf); free(out_buf); return False

    # Calculate the offsets
    for i in xrange(filter_size):
        for j in xrange(filter_size):
            off[i*filter_size + j] = (i - f_rad) * stride0 + (j - f_rad) * stride1

    # Get memory pointers
    cdef CHAR_PTR_CA8R pi_row = (<CHAR_PTR_CA8R>&input[0,0]) + f_rad * (stride0 + stride1)
    cdef CHAR_PTR_CA8R pi
    cdef CHAR_PTR_A8R po = <CHAR_PTR_A8R>&output[0,0,0]

    # Process each pixel
    for i in xrange(H):
        pi = pi_row
        for j in xrange(W):
            for x in xrange(in_sz): in_buf[x] = (<DOUBLE_PTR_CAR>(pi + off[x]))[0]
            pi += stride1

            function(in_buf, filter_size, filter_size, out_buf, data)

            for x in xrange(out_sz): (<DOUBLE_PTR_AR>(po + x*out_stride))[0] = out_buf[x]
            po += sizeof(double)
        pi_row += stride0

    free(off)
    free(in_buf)
    free(out_buf)
    
    return True

def ConstructHOG(ndarray im not None, int filt_width=15, ndarray out=None, int nthreads=1):
    """
    The entire ConstructHOG function in Cython. Uses a modified scipy.ndimge.filters.generic_filter
    for calling the HoG function. Some other optimizations are using a single memory alocation for
    temporary data storage, giving generic_filter a C function instead of a Python function, and is
    multi-threaded.
    """
    # Check arguments
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2: raise ValueError("Invalid im")
    cdef intp filt_width_1 = filt_width - 1
    cdef intp n, H = PyArray_DIM(im, 0) - filt_width_1, W = PyArray_DIM(im, 1) - filt_width_1
    cdef intp tmp_n = HoG_init(filt_width, filt_width, &n)
    out = __get_out(out, n, H, W)
    nthreads = __get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows

    # Setup variables to be given to generic_filter
    cdef bint success = True
    cdef bint* success_p = &success # in OpenMP parallel blocks we cannot directly assign to success, we need a pointer
    cdef double[:,:] im_mv
    if PyArray_ISWRITEABLE(im): im_mv = im
    else:
        # HACK - we never write to im_mv but Cython Memoryviews do not support read-only memory at all
        PyArray_ENABLEFLAGS(im, NPY_ARRAY_WRITEABLE)
        im_mv = im
        PyArray_CLEARFLAGS(im, NPY_ARRAY_WRITEABLE)
    cdef double[:,:,::view.contiguous] out_mv = out
    
    # Temporary storage (for each thread)
    cdef DOUBLE_PTR_AR tmp = <DOUBLE_PTR_AR>malloc(nthreads * tmp_n * sizeof(double))
    if tmp is NULL: raise MemoryError()
    
    # Variables used in parallel block
    cdef intp i, a, b
    cdef double inc
    
    with nogil:
        if nthreads == 1:
            success = __generic_filter(im_mv, <filter_func>&HoG_run, tmp, filt_width, out_mv)
        else:
            # This uses OpenMP to do the multi-processing
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = H / <double>nthreads # a floating point number, use the floor of adding it together
                i = threadid()
                a = <intp>floor(inc*i)
                b = H if i == nthreads - 1 else (<intp>floor(inc*(i+1)))
                if not __generic_filter(im_mv[a:b+filt_width_1,:], <filter_func>&HoG_run, tmp + i*tmp_n, filt_width, out_mv[:,a:b,:]):
                    success_p[0] = False

        ## This can be done without OpenMP as well but instead with Python threads, with very little penalty
        #from threading import Thread
        #def __thread(intp i):
        #    cdef double inc = H / <double>nthreads
        #    cdef intp a = floor(inc*i), b = H if i == nthreads - 1 else floor(inc*(i+1))
        #    cdef double[:, :] im = im_mv[a:b+filt_width_1,:]
        #    cdef double[:,:,::view.contiguous] out = out_mv[:,a:b,:]
        #    with nogil:
        #        if not generic_filter_x(im, <filter_func>&HoG_run, tmp + i*tmp_n, filt_width, out):
        #            success = False
        #threads = [None] * nthreads
        #for i in xrange(nthreads):
        #    threads[i] = t = Thread(target=__thread, name="HOG-"+str(i), args=(i,))
        #    t.start()
        #for t in threads: t.join()
        
    free(tmp)
   
    if not success: raise MemoryError()
    return out

### Older version of ConstructHOG that used SciPy's generic_filter (which is not multi-threaded)
##cdef struct __wrapper_data:
##    intp filter_width # width/height of the filter
##    intp out_size # the amount of data generated per HoG run
##    intp out_pos  # the current linear pixel position
##    double** out  # an array of pointers to where the outputs are stored
##                  # accessed like out[i][j] where i is in [0,out_size) and j is out_pos
##                  # the outer array needs to be freed, the inner arrays are pointers into a NumPy array
##    double* out_buf # output buffer for run, size of out_size, must be freed
##    double* tmp_buf # temporary buffer given to run, must be freed
##
##cdef int __wrapper(const double* buf, intp filter_size, double* retval, __wrapper_data* data) nogil:
##    #assert filter_size == data.filter_width * data.filter_width
##    HoG_run(buf, data.filter_width, data.filter_width, data.out_buf, data.tmp_buf)
##    cdef intp i
##    for i in xrange(data.out_size):
##        data.out[i][data.out_pos] = data.out_buf[i]
##    data.out_pos += 1
##    #retval[0] = 0.0 # already set to 0 by default
##    return 1 # return zero if it has failed, if returning 0 can also set a Python error
##
##IF USE_PYCAPSULES:
##    cdef void __wrapper_destructor(object capsule):
##        free(PyCapsule_GetContext(capsule))
##
##    def __wrapper_func(__wrapper_data* data):
##        cdef object capsule = PyCapsule_New(<void*>&__wrapper, "HoG-Filter-Wrapper", &__wrapper_destructor)
##        PyCapsule_SetContext(capsule, data)
##        return capsule
##    
##ELSE:
##    cdef void __wrapper_destructor(void* func, void* _data):
##        cdef __wrapper_data* data = <__wrapper_data*>_data
##        free(data.tmp_buf)
##        free(data.out_buf)
##        free(data.out)
##        free(data)
##
##    cdef object __wrapper_func(__wrapper_data* data):
##        return PyCObject_FromVoidPtrAndDesc(<void*>&__wrapper, data, &__wrapper_destructor)
##        
##def ConstructHOG(ndarray im, intp filt_width, ndarray[npy_double, ndim=3] out=None):
##    """
##    The entire ConstructHOG function in Cython. Uses scipy.ndimge.filters.generic_filter for calling
##    the HoG function. Some other optimizations are using a single memory alocation for temporary
##    data storage and a giving generic_filter a C function instead of a Python function.
##    """
##    from scipy.ndimage.filters import generic_filter
##
##    cdef __wrapper_data* data = <__wrapper_data*>malloc(sizeof(__wrapper_data))
##    if data is NULL: raise MemoryError()
##    data.filter_width = filt_width
##    data.out_pos = 0
##    data.out = NULL
##    data.out_buf = NULL
##    data.tmp_buf = NULL
##    cdef object func = __wrapper_func(data)
##
##    cdef intp n, i, H = PyArray_DIM(im, 0), W = PyArray_DIM(im, 1)
##    data.tmp_buf = <double*>malloc(HoG_init(filt_width, filt_width, &n) * sizeof(double))
##    data.out_buf = <double*>malloc(n * sizeof(double))
##    data.out     = <double**>malloc(n * sizeof(double*))
##    if data.tmp_buf is NULL or data.out_buf is NULL or data.out is NULL: raise MemoryError()
##    data.out_size = n
##
##    cdef intp dims[3]
##    if out is None:
##        dims[0] = n; dims[1] = H; dims[2] = W
##        out = PyArray_EMPTY(3, dims, NPY_DOUBLE, False)
##    elif PyArray_DIM(out, 0) != n or PyArray_DIM(out, 1) != H or PyArray_DIM(out, 2) != W or \
##         PyArray_STRIDE(out, 1) != PyArray_DIM(out, 2)*sizeof(double) or PyArray_STRIDE(out, 2) != sizeof(double):
##        # Out must be a n x h x w matrix with the last two axes being C contiguous
##        raise ValueError('Invalid output array')
##    #cdef double* out_p = <double*>PyArray_DATA(out)
##    #cdef intp npix = H*W
##    for i in xrange(data.out_size): data.out[i] = <double*>PyArray_GETPTR1(out, i)
##
##    # Make a dummy output array where every value in the array maps to a single value
##    # This means it doesn't require allocation, doesn't really require memory, and hopefully is
##    # slightly faster because we don't need to be jumping around in memory ever
##    # We can do this because we don't need the output generated by generic_filter.
##    cdef double out_dummy_val
##    cdef ndarray out_dummy = PyArray_SimpleNewFromData(2, PyArray_SHAPE(im), NPY_DOUBLE, &out_dummy_val)
##    PyArray_STRIDES(out_dummy)[0] = 0
##    PyArray_STRIDES(out_dummy)[1] = 0
##
##    generic_filter(im, func, size=(filt_width,filt_width), output=out_dummy) #, mode='constant')
##    
##    return out


########## SIFT Calculator Pieces ##########

def sift_orientations(double[:,::1] im_mag not None, double[:,::1] im_cos not None, double[:,::1] im_sin not None, int num_angles, int nthreads=1):
    # CHANGED: this was extracted from dense_sift for speed increase and multi-threading
    cdef intp a, i, j, H = im_cos.shape[0], W = im_cos.shape[1]
    cdef double x, angle_cos, angle_sin, angle_step = 2*M_PI/num_angles
    
    # make orientation images
    # for each histogram angle
    cdef intp[3] dims
    dims[0] = num_angles; dims[1] = H; dims[2] = W
    cdef ndarray out = PyArray_EMPTY(3, dims, NPY_DOUBLE, False) # INTERMEDIATE: 8 * im.shape
    cdef double[:,:,::1] im_orientation = out
    for a in prange(num_angles, nogil=True, schedule='static', num_threads=nthreads):
        # compute each orientation channel
        angle_cos = cos(a*angle_step)
        angle_sin = sin(a*angle_step)
        for i in xrange(H):
            for j in xrange(W):
                x = im_cos[i,j] * angle_cos + im_sin[i,j] * angle_sin
                x = x * (x > 0)
                #x **= __sift.alpha # yes, power is REALLY slow, better to do it by hand...
                x = x * x * x # x^3
                x = x * x * x # x^9
                # weight by magnitude
                im_orientation[a,i,j] = x * im_mag[i,j]
    return out

def sift_neighborhoods(double[:,:,:,::1] out not None, double[:,:,::1] im_orientation not None, int num_bins, int step, int nthreads=1):
    # CHANGED: this was extracted from dense_sift for speed increase and multi-threading
    cdef intp i, j, x, y, H = out.shape[2], W = out.shape[3]
    for i in prange(num_bins, nogil=True, schedule='static', num_threads=nthreads):
        x = i*step
        for j in xrange(num_bins):
            y = j*step
            out[i*num_bins+j,:,:,:] = im_orientation[:, y:y+H, x:x+W]

def sift_normalize(double[:,::1] sift_arr not None, int nthreads=1):
    """normalize SIFT descriptors (after Lowe)"""
    # CHANGED: this operates on sift_arr in place and operates multi-threaded
    # OPT: this would work better with the C-order transposed array... but that would likely need to copy the data
    # However, that may not help, I did try copying columns of sift_arr to a temporary as necessary
    # and moving back at the end and it hurt the time.

    cdef intp i, j, nfeat = sift_arr.shape[0], npix = sift_arr.shape[1]
    cdef double norm

    # find indices of descriptors to be normalized (those whose norm is larger than 1)
    for j in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
    # or if we plan on having a lot of things hit that 'continue'
    #for j in prange(npix, nogil=True, schedule='guided', num_threads=nthreads):
        norm = 0
        for i in xrange(nfeat): norm += sift_arr[i,j] * sift_arr[i,j]
        if norm <= 1: continue
        norm = 1/sqrt(norm)
        for i in xrange(nfeat): 
            sift_arr[i,j] *= norm
            if sift_arr[i,j] > 0.2: sift_arr[i,j] = 0.2 # suppress large gradients
        # finally, renormalize to unit length
        norm = 0
        for i in xrange(nfeat): norm += sift_arr[i,j] * sift_arr[i,j]
        norm = 1/sqrt(norm)
        for i in xrange(nfeat): sift_arr[i,j] *= norm
