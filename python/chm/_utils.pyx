#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
Cython code for util. At the moment this is just fast/parallelized copy and hypot functions.

Other future functions that might go here are a parallelized version of numpy.pad and optimized
versions of MyMaxPooling and/or im2double.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters/filters.pxi"

ctypedef double* dbl_p

from libc.math cimport sqrt, hypot
from cython.parallel cimport parallel

def par_copy(dst, src, int nthreads):
    """
    Equivilent to numpy.copyto(dst, src, 'unsafe', None) but is parallelized with the given number
    of threads.
    """
    if dst.shape != src.shape: raise ValueError('dst and src must be same shape')
    nthreads = get_nthreads(nthreads, dst.size // 100000000)
    cdef intp N = dst.shape[0]
    cdef Range r
    if nthreads == 1: PyArray_CopyInto(dst, src)
    else:
        with nogil, parallel(num_threads=nthreads):
            r = get_thread_range(N)
            with gil: PyArray_CopyInto(dst[r.start:r.stop], src[r.start:r.stop])

def par_copy_any(dst, src, int nthreads):
    """
    Equivilent to numpy.copyto(dst, src, 'unsafe', None) but is parallelized with the given number
    of threads and works as long as dst and src are the same size (they can be different shapes).
    This is twice as slow as _util.par_copy, so if they are the same shape that should be used.
    """
    if dst.size != src.size: raise ValueError('dst and src must be same size')
    nthreads = get_nthreads(nthreads, dst.size // 50000000)
    cdef intp N1 = dst.shape[0], N2 = src.shape[0]
    cdef Range r1, r2
    if nthreads == 1: PyArray_CopyAnyInto(dst, src)
    else:
        with nogil, parallel(num_threads=nthreads):
            r1 = get_thread_range(N1); r2 = get_thread_range(N2)
            with gil: PyArray_CopyAnyInto(dst[r1.start:r1.stop], src[r2.start:r2.stop])

def par_hypot(ndarray x, ndarray y, ndarray out=None, int nthreads=1, bint precise=False):
    """
    Equivilent to doing sqrt(x*x + y*y) elementwise but slightly faster and parallelized.
    
    If precise is True (default it False), uses the C `hypot` function instead to protect against
    intermediate underflow and overflow situations at the potential cost of time.
    
    This requires the arrays to be 2D doubles with the last axis being contiguous.
    """
    # Check inputs
    cdef intp H = PyArray_DIM(x,0), W = PyArray_DIM(x,1)
    if not PyArray_ISBEHAVED_RO(x) or not PyArray_ISBEHAVED_RO(y) or \
       PyArray_TYPE(x) != NPY_DOUBLE or PyArray_TYPE(y) != NPY_DOUBLE or \
       PyArray_NDIM(x) != 2 or PyArray_NDIM(y) != 2 or H != PyArray_DIM(y, 0) or W != PyArray_DIM(y, 1) or \
       PyArray_STRIDE(x, 1) != sizeof(double) or  PyArray_STRIDE(y, 1) != sizeof(double):
        raise ValueError('Invalid input arrays')
    
    # Check output
    if out is None:
        out = PyArray_EMPTY(2, [H, W], NPY_DOUBLE, False)
    elif not PyArray_ISBEHAVED(out) or PyArray_TYPE(out) != NPY_DOUBLE or PyArray_NDIM(out) != 2 or \
         PyArray_DIM(out, 0) != H or PyArray_DIM(out, 1) != W or PyArray_STRIDE(out, 1) != sizeof(double):
        raise ValueError('Invalid output array')
    
    # Check nthreads
    nthreads = get_nthreads(nthreads, H // 64)
    
    # Get pointers
    cdef intp x_stride = PyArray_STRIDE(x, 0)//sizeof(double), y_stride = PyArray_STRIDE(y, 0)//sizeof(double)
    cdef intp out_stride = PyArray_STRIDE(out, 0)//sizeof(double)
    cdef dbl_p X = <dbl_p>PyArray_DATA(x), Y = <dbl_p>PyArray_DATA(y), OUT = <dbl_p>PyArray_DATA(out)
    
    # Run the hypotenuse calculator
    cdef Range r
    cdef hypot_fp hyp = <hypot_fp>hypot2 if precise else <hypot_fp>hypot1
    with nogil:
        if nthreads == 1: hyp(X, Y, OUT, H, W, x_stride, y_stride, out_stride)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(H)
                hyp(X+r.start*x_stride, Y+r.start*y_stride, OUT+r.start*out_stride, r.stop-r.start,
                    W, x_stride, y_stride, out_stride)
                
    # Done!
    return out

ctypedef void (*hypot_fp)(dbl_p x, dbl_p y, dbl_p out, intp H, intp W, intp x_stride, intp y_stride, intp out_stride) nogil
cdef void hypot1(dbl_p x, dbl_p y, dbl_p out, intp H, intp W, intp x_stride, intp y_stride, intp out_stride) nogil:
    """
    Calculates out = sqrt(x*x + y*y) for each value in x, y, and out. The arrays must all be HxW.
    The strides between rows of each row are given x_stride, y_stride, and out_stride.
    """
    cdef intp i, j
    for i in xrange(H):
        for j in xrange(W): out[j] = sqrt(x[j]*x[j] + y[j]*y[j])
        x += x_stride
        y += y_stride
        out += out_stride
cdef void hypot2(dbl_p x, dbl_p y, dbl_p out, intp H, intp W, intp x_stride, intp y_stride, intp out_stride) nogil:
    """
    Calculates out = hypot(x, y) for each value in x, y, and out (where hypot is the C hypot
    function. The arrays must all be HxW. The strides between rows of each row are given x_stride,
    y_stride, and out_stride.
    """
    cdef intp i, j
    for i in xrange(H):
        for j in xrange(W): out[j] = hypot(x[j], y[j])
        x += x_stride
        y += y_stride
        out += out_stride
