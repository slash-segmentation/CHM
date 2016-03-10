#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
#distutils: language=c++

"""
The core imresize functions in Cython.

Jeffrey Bush, 2015, NCMIR, UCSD
"""

# TODO: there seems to be a Cython bug in that sometimes a cdef function statement with an argument
# of intp or Py_intptr_t becomes int! This then causes problems on 64-bit machines where
# sizeof(int) != sizeof(intp), and there is then a compile error intp or intp* and when trying to
# use it! Temporary workaround is to use a void* pointer.

from __future__ import division

include "npy_helper.pxi"
include "fused.pxi"

def imresize(ndarray im, ndarray weights, ndarray indices):
    """
    Internal function for imresize written in Cython. The image must be aligned, solid, C or
    Fortran contiguous, 2-dimensional, and an integral or floating-point type (although float16 is
    not supported). The weights and indices must be 2D, the same size, and R x N sized (where R is
    the size of the new first dimension of the image) and C contiguous. They are of type double
    and intp. Out of all of these requirements, very few are actual requirements but instead will
    prevent copying of data.
    """
    cdef bint fortran = PyArray_ISFARRAY_RO(im)
    if not fortran and not PyArray_ISCARRAY_RO(im) or PyArray_NDIM(im) != 2: raise ValueError('Invalid image given to imresize core')
    
    cdef intp dims[2]
    dims[0] = PyArray_DIM(weights, 0)
    dims[1] = PyArray_DIM(im, 1)
    cdef ndarray out = PyArray_EMPTY(2, dims, PyArray_TYPE(im), fortran)
    if fortran: imresize_f[PyArray_TYPE(im)](im, out, weights, indices)
    else:       imresize_c[PyArray_TYPE(im)](im, out, weights, indices)
    return out

@fused
def imresize_f(ndarray[npy_number_basic, ndim=2, mode='fortran'] im, ndarray out, ndarray weights, ndarray indices):
    """Fortran specialization wrapper, just calls _imresize_f (after adjusting weights and indices)"""
    weights = PyArray_Transpose(weights, NULL)
    indices = PyArray_Transpose(indices, NULL)
    weights = PyArray_CheckFromAny(weights, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_FARRAY_RO|NPY_ARRAY_NOTSWAPPED, NULL)
    indices = PyArray_CheckFromAny(indices, PyArray_DescrFromType(NPY_INTP),   2, 2, NPY_ARRAY_FARRAY_RO|NPY_ARRAY_NOTSWAPPED, NULL)
    with nogil:
        _imresize_f[npy_number_basic](<npy_number_basic*>PyArray_DATA(im), <npy_number_basic*>PyArray_DATA(out),
                                      <double*>PyArray_DATA(weights), <void*>PyArray_DATA(indices),
                                      PyArray_DIM(im, 0), PyArray_DIM(im, 1), PyArray_DIM(weights, 1), PyArray_DIM(weights, 0))

cdef void _imresize_f(npy_number_basic* im, npy_number_basic* out, double* weights, void* indices, intp Rin, intp C, intp Rout, intp N) nogil:
    cdef npy_number_basic* im_p
    cdef npy_number_basic* out_p
    cdef double* weights_p
    cdef intp*   indices_p
    cdef double value
    cdef intp r, c, n
    for c in xrange(C):
        im_p  = im +c*Rin  # im_p  = im[:,c]
        out_p = out+c*Rout # out_p = out[:,c]
        for r in xrange(Rout):
            value = 0.0
            weights_p = weights+r*N # weights[:,r]
            indices_p = (<intp*>indices)+r*N # indices[:,r]
            for n in xrange(N):
                value += weights_p[n] * im_p[indices_p[n]]
            out_p[r] = cast_with_clip(value, <npy_number_basic>0)

@fused
def imresize_c(ndarray[npy_number_basic, ndim=2, mode='c'] im, ndarray out, ndarray weights, ndarray indices):
    """C specialization wrapper, just calls _imresize_c (after adjusting weights and indices)"""
    weights = PyArray_CheckFromAny(weights, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_NOTSWAPPED, NULL)
    indices = PyArray_CheckFromAny(indices, PyArray_DescrFromType(NPY_INTP),   2, 2, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_NOTSWAPPED, NULL)
    with nogil:
        _imresize_c[npy_number_basic](<npy_number_basic*>PyArray_DATA(im), <npy_number_basic*>PyArray_DATA(out),
                                      <double*>PyArray_DATA(weights), <void*>PyArray_DATA(indices),
                                      PyArray_DIM(im, 0), PyArray_DIM(im, 1), PyArray_DIM(weights, 0), PyArray_DIM(weights, 1))

cdef void _imresize_c(npy_number_basic* im, npy_number_basic* out, double* weights, void* indices, intp Rin, intp C, intp Rout, intp N) nogil:
    #cdef npy_number_basic* im_p
    cdef npy_number_basic* out_p
    cdef double* weights_p
    cdef intp*   indices_p
    cdef double value
    cdef intp r, c, n
    for r in xrange(Rout):
        #im_p  = im +r*C # im_p  = im[r,:]
        out_p = out+r*C # out_p = out[r,:]
        weights_p = weights+r*N # weights[r,:]
        indices_p = (<intp*>indices)+r*N # indices[r,:]
        for c in xrange(C):
            value = 0.0
            for n in xrange(N):
                value += weights_p[n] * im[indices_p[n]*C+c]
            out_p[c] = cast_with_clip(value, <npy_number_basic>0)


########## Fast imresize ##########

def imresize_fast(im):
    """
    Internal function for imresize_fast written in Cython. The image has the same requirements as
    per imresize.
    """
    cdef bint fortran = PyArray_ISFARRAY_RO(im)
    if not fortran and not PyArray_ISCARRAY_RO(im) or PyArray_NDIM(im) != 2: raise ValueError('Invalid image given to imresize core')

    cdef intp dims[2]
    dims[0] = (PyArray_DIM(im, 0) + 1) // 2
    dims[1] = PyArray_DIM(im, 1)
    cdef ndarray out = PyArray_EMPTY(2, dims, PyArray_TYPE(im), fortran)
    if fortran: imresize_fast_f[PyArray_TYPE(im)](im, out)
    else:       imresize_fast_c[PyArray_TYPE(im)](im, out)
    return out

cdef double* bicubic_weights = [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875]

@fused
def imresize_fast_f(ndarray[npy_number_basic, ndim=2, mode='fortran'] im, ndarray out):
    """Fortran specialization wrapper, just calls _imresize_fast_f"""
    with nogil:
        _imresize_fast_f[npy_number_basic](<npy_number_basic*>PyArray_DATA(im),
                                           <npy_number_basic*>PyArray_DATA(out),
                                           PyArray_DIM(im, 0), PyArray_DIM(im, 1))

# OPT: one further optimization here would be to separate out the rows that need clipping (first and last 2 rows)

cdef void _imresize_fast_f(npy_number_basic* im, npy_number_basic* out, intp Rin, intp C) nogil:
    cdef intp Rout = (Rin  + 1) // 2
    cdef npy_number_basic* im_p
    cdef npy_number_basic* out_p
    cdef double value
    cdef intp r, c, n, i, i_start
    for c in xrange(C):
        im_p  = im +c*Rin  # im_p  = im[:,c]
        out_p = out+c*Rout # out_p = out[:,c]
        for r in xrange(Rout):
            value = 0.0
            i_start = -3+2*r
            for n in xrange(8):
                i = i_start + n
                if i < 0: i = 0
                elif i >= Rin: i = Rin - 1
                value += bicubic_weights[n] * im_p[i]
            out_p[r] = cast_with_clip(value, <npy_number_basic>0)

@fused
def imresize_fast_c(ndarray[npy_number_basic, ndim=2, mode='c'] im, ndarray out):
    """C specialization wrapper, just calls _imresize_fast_c"""
    with nogil:
        _imresize_fast_c[npy_number_basic](<npy_number_basic*>PyArray_DATA(im),
                                           <npy_number_basic*>PyArray_DATA(out),
                                           PyArray_DIM(im, 0), PyArray_DIM(im, 1))

cdef void _imresize_fast_c(npy_number_basic* im, npy_number_basic* out, intp Rin, intp C) nogil:
    cdef intp Rout = (Rin  + 1) // 2
    #cdef npy_number_basic* im_p
    cdef npy_number_basic* out_p
    cdef double value
    cdef intp r, c, n, n2, i, i_start
    for r in xrange(Rout):
        #im_p  = im +r*C # im_p  = im[r,:]
        out_p = out+r*C # out_p = out[r,:]
        for c in xrange(C):
            value = 0.0
            i_start = -3+2*r
            for n in xrange(8):
                i = i_start + n
                if i < 0: i = 0
                elif i >= Rin: i = Rin - 1
                value += bicubic_weights[n] * im[i*C+c]
            out_p[c] = cast_with_clip(value, <npy_number_basic>0)
