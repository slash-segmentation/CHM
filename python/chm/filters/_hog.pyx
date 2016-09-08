#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
HOG filter written in Cython and C. This was originally in MEX (MATLAB's Cython-like system) and
C++. Improved speed, added multi-threading, and increased accuracy.

Most of the code is in a separate C++ file.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters.pxi"

from libc.stdlib cimport malloc, free
from libc.math cimport ceil, floor
from cython.parallel cimport parallel
from openmp cimport omp_get_num_threads, omp_get_thread_num
from cython.view cimport contiguous

##### MATLAB Compatible Version #####
cdef extern from "HOG.h" nogil:
    cdef intp HOG_init(const intp w, const intp h, const intp *n)
    cdef void HOG_run(const double *pixels, const intp w, const intp h, double *out, double *H, const intp padding)
    cdef intp _HOG "HOG" (const double *pixels, const intp w, const intp h, double *out, const intp n)

def HOG(ndarray[npy_double, ndim=2, mode='c'] pixels not None, ndarray[npy_double, ndim=1, mode='c'] out not None):
    """
    Implements the HOG filter for a single block. Does not support changing the parameters directly
    (they are compiled in). Additionally RGB images are not supported. Instead of allocating the
    memory for the results, you must pass in the output array. The destination array must be a 1-D
    double array and have at least as many elements as are needed for the output. This function
    returns the number of elements stored in the array.
    """
    # NOTE: a much faster method is to bypass this and use "hog_entire(...)"
    if not PyArray_ISBEHAVED_RO(pixels): raise ValueError("Invalid im")
    if not PyArray_ISBEHAVED(out): raise ValueError("Invalid out")
    cdef intp n
    with nogil: n = _HOG(<DOUBLE_PTR_CAR>PyArray_DATA(pixels), PyArray_DIM(pixels, 1), PyArray_DIM(pixels, 0),
                         <DOUBLE_PTR_AR>PyArray_DATA(out), PyArray_DIM(out, 0))
    if n == -1: raise ValueError("Output array too small")
    if n == -2: raise MemoryError()
    return n

##### Highly optimized version for entire image #####
ctypedef void (*filter_func)(DOUBLE_PTR_CAR, const intp, const intp, DOUBLE_PTR_AR, void*, const intp) nogil

cdef bint generic_filter(double[:, :] input,
                         filter_func function, void* data, intp filter_size, intp pad_size,
                         double[:, :, ::contiguous] output) nogil:
    """
    Similar to SciPy's scipy.ndimage.filters.generic_filter with the following new features:
     * The output has one additional dimension over the input and the filter generates an array of
       data for each pixel instead of a scalar
     * The GIL is no longer required, meaning it can be used multi-threaded
     * Instead of extending the edges of the image only the 'valid' pixels are processed
     * Automatic padding added (this could be done by modifying the filter function itself though)

    To simplify the code, the following features were dropped (but could be re-added if necessary):
     * only accepts 2D double matrices as input
     * the last N axes of output must be C-order
     * the footprint is always a square and origin is always zeros
    """
    cdef intp i, j, x
    cdef intp f_rad = filter_size // 2 + pad_size
    cdef intp stride0 = input.strides[0], stride1 = input.strides[1], out_stride = output.strides[0]
    cdef intp full_size = filter_size + pad_size*2
    cdef intp H = input.shape[0] - full_size + 1, W = input.shape[1] - full_size + 1
    cdef intp in_sz = (filter_size + pad_size*2) * (filter_size + pad_size*2), out_sz = output.shape[0]
    
    # Allocate buffers
    cdef INTP_PTR_AR off = <INTP_PTR_AR>malloc(in_sz * sizeof(intp))
    cdef DOUBLE_PTR_AR in_buf = <DOUBLE_PTR_AR>malloc(in_sz * sizeof(double))
    cdef DOUBLE_PTR_AR out_buf = <DOUBLE_PTR_AR>malloc(out_sz * sizeof(double))
    if off is NULL or in_buf is NULL or out_buf is NULL: free(off); free(in_buf); free(out_buf); return False

    # Calculate the offsets
    for i in xrange(full_size):
        for j in xrange(full_size):
            off[i*full_size + j] = (i - f_rad) * stride0 + (j - f_rad) * stride1

    # Get memory pointers
    cdef CHAR_PTR_CA8R pi_row = (<CHAR_PTR_CA8R>&input[0,0]) + f_rad * (stride0 + stride1), pi
    cdef CHAR_PTR_A8R po = <CHAR_PTR_A8R>&output[0,0,0]

    # Process each pixel
    for i in xrange(H):
        pi = pi_row
        for j in xrange(W):
            for x in xrange(in_sz): in_buf[x] = (<DOUBLE_PTR_CAR>(pi + off[x]))[0]
            pi += stride1

            function(in_buf, filter_size, filter_size, out_buf, data, pad_size)

            for x in xrange(out_sz): (<DOUBLE_PTR_AR>(po + x*out_stride))[0] = out_buf[x]
            po += sizeof(double)
        pi_row += stride0

    free(off)
    free(in_buf)
    free(out_buf)

    return True

def hog_entire(ndarray im not None, int filt_width=15, bint compat=True, ndarray out=None, int nthreads=1):
    """
    The entire HOG filter in Cython. Uses a modified scipy.ndimge.filters.generic_filter for
    calling the HOG function. Some other optimizations are using a single memory alocation for
    temporary data storage, giving generic_filter a C function instead of a Python function, and is
    multi-threaded. This also supports running in 'non-compat' which pads the image with image data
    instead of 0s.
    """
    # Check arguments
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2: raise ValueError("Invalid im")
    cdef intp fw_1 = filt_width - 1, pad = 0 if compat else 1
    cdef intp n, H = PyArray_DIM(im, 0) - fw_1 - 2*pad, W = PyArray_DIM(im, 1) - fw_1 - 2*pad
    cdef intp tmp_n = HOG_init(filt_width, filt_width, &n)
    out = get_out(out, n, H, W)
    nthreads = get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows

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
    cdef double[:,:,::contiguous] out_mv = out
    
    # Temporary storage (for each thread)
    cdef DOUBLE_PTR_AR tmp = <DOUBLE_PTR_AR>malloc(nthreads * tmp_n * sizeof(double))
    if tmp is NULL: raise MemoryError()
    
    cdef intp a, b
    with nogil:
        if nthreads == 1:
            success = generic_filter(im_mv, <filter_func>&HOG_run, tmp, filt_width, pad, out_mv)
        else:
            # This uses OpenMP to do the multi-processing
            with parallel(num_threads=nthreads):
                a = get_range(H, &b);
                if not generic_filter(im_mv[a:b+fw_1,:], <filter_func>&HOG_run,
                                      tmp+omp_get_thread_num()*tmp_n,
                                      filt_width, pad, out_mv[:,a:b,:]):
                    success_p[0] = False

        ## This can be done without OpenMP as well but instead with Python threads, with very little penalty
        #from threading import Thread
        #def thread(intp i):
        #    cdef intp b, a = get_range(H, &b)
        #    cdef double[:, :] im = im_mv[a:b+fw_1,:]
        #    cdef double[:,:,::contiguous] out = out_mv[:,a:b,:]
        #    with nogil:
        #        if not generic_filter(im, <filter_func>&HOG_run, tmp+omp_get_thread_num()*tmp_n,
        #                              filt_width, pad, out):
        #            success = False
        #threads = [None] * nthreads
        #for i in xrange(nthreads):
        #    threads[i] = t = Thread(target=thread, name="HOG-"+str(i), args=(i,))
        #    t.start()
        #for t in threads: t.join()
        
    free(tmp)
   
    if not success: raise MemoryError()
    return out
