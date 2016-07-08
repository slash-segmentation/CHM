"""
Useful Cython function for filters.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from pysegtools.general.cython.npy_helper cimport *
import_array()

from openmp cimport omp_get_max_threads

########## Utilities ##########

cdef inline ndarray get_out(ndarray out, intp N, intp H, intp W):
    """
    Checks the output array to make sure it is an NxHxW double array that is behaved with the last
    two strides being element-sized strides.
    
    If out is None then a new array is created and returned. If it is not None and doesn't meet the
    requirements, a ValueError is raised.
    """
    cdef intp[3] dims
    if out is None:
        dims[0] = N; dims[1] = H; dims[2] = W
        return PyArray_EMPTY(3, dims, NPY_DOUBLE, False)
    elif not PyArray_ISBEHAVED(out) or PyArray_TYPE(out) != NPY_DOUBLE or PyArray_NDIM(out) != 3 or \
         PyArray_DIM(out, 0) != N or PyArray_DIM(out, 1) != H or PyArray_DIM(out, 2) != W or \
         PyArray_STRIDE(out, 1) != <intp>(W*sizeof(double)) or PyArray_STRIDE(out, 2) != sizeof(double):
        raise ValueError('Invalid output array')
    return out

cdef inline int get_nthreads(int nthreads, intp max_threads) nogil:
    """
    Gets the number of threads to run on, given the argument input, the max that should be run
    (to make sure each thread does sufficient work), and the number of threads given by OpenMP.
    
    Equivilent to max(min(nthreads, max_threads, omp_get_max_threads()), 1)
    """
    if nthreads > max_threads: nthreads = <int>max_threads
    if nthreads > omp_get_max_threads(): nthreads = omp_get_max_threads()
    return 1 if nthreads < 1 else nthreads
