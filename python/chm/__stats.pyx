#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: language=c++

"""
Optimized stats functions that don't copying data excessively and are multi-threaded. Includes:
    
    cy_min_max(_0/_1) - calculates min and max of a 1D or 2D array along a particular axis
    cy_mean_stdev(_0/_1) - calculates mean and std deviation of a 1D or 2D array along an axis
    cy_percentile(_0/_1) - gets the values of percentiles of a 1D or 2D array along an axis

Typically these have the greatest speed benefit over the similar NumPy functions when the data is
very large and particularily if it is in a memory-mapped file. Even when using a single thread they
prove to by a few times faster than the NumPy versions for large data sets. For example, the
cy_min_max calculates min and max faster than NumPy calculates just min.

In the stats module there are wrappers for these functions that choose which version to call or
fall back to the NumPy varieties if needed.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters/filters.pxi"

from libc.stdlib cimport malloc, free
from libc.string cimport memset, strerror
from libc.math cimport floor, sqrt, NAN, nextafter, INFINITY
from libc.errno cimport errno, ENOMEM, ERANGE
from cython.parallel cimport parallel, prange
from cython.view cimport array

__all__ = ['cy_min_max', 'cy_min_max_0', 'cy_min_max_1',
           'cy_mean_stdev', 'cy_mean_stdev_0', 'cy_mean_stdev_1',
           'cy_percentile', 'cy_percentile_0', 'cy_percentile_1']

########## 1D Stats Calculator ##########
def cy_min_max(double[:] x, int nthreads=1):
    """
    Calculates the min and max. Very fast with no data copying and parallelized (although default
    is 1 thread). Returns the min and max.
    """
    cdef intp i, N = x.shape[0]
    if N == 0: return (NAN,NAN)
    nthreads = get_nthreads(nthreads, N // 5000)
    cdef Range r
    cdef double min, max
    cdef double[::1] mins, maxs
    if nthreads == 1:
        with nogil: __min_max(x, 0, N, &min, &max)
    else:
        mins = array(shape=(nthreads,), itemsize=sizeof(double), format="d")
        maxs = array(shape=(nthreads,), itemsize=sizeof(double), format="d")
        with nogil:
            memset(&mins[0], 0xFF, nthreads*sizeof(double))
            memset(&maxs[0], 0xFF, nthreads*sizeof(double))
            with parallel(num_threads=nthreads):
                r = get_thread_range(N)
                i = omp_get_thread_num()
                __min_max(x, r.start, r.stop, &mins[i], &maxs[i])
            min = mins[0]; max = maxs[0]
            for i in xrange(1, nthreads):
                if mins[i] < min: min = mins[i]
                if maxs[i] > max: max = maxs[i]
    return (min, max)

cdef inline void __min_max(double[:] x, intp L, intp R, double* min, double* max) nogil:
    cdef double mn = x[L], mx = x[L]
    for i in xrange(L+1, R):
        if   x[i] < mn: mn = x[i]
        elif x[i] > mx: mx = x[i]
    min[0] = mn; max[0] = mx
    
def cy_mean_stdev(double[:] x, double ddof=0.0, int nthreads=1):
    """
    Calculates the mean and standard deviation. Very fast with no data copying and parallelized
    (although default is 1 thread). Returns the mean and standard deviation.
    """
    cdef intp i, N = x.shape[0]
    if N == 0: return (NAN,NAN)
    if N == 1: return (x[0], 0.0 if ddof != 1.0 else NAN)
    nthreads = get_nthreads(nthreads, N // 5000)
    cdef double mean = 0, std = 0
    with nogil:
        if nthreads == 1:
            for i in xrange(N): mean += x[i]
            mean /= N
            for i in xrange(N): std += (x[i]-mean)*(x[i]-mean)
        else:
            for i in prange(N, num_threads=nthreads): mean += x[i]
            mean /= N
            for i in prange(N, num_threads=nthreads): std += (x[i]-mean)*(x[i]-mean)
        std = sqrt(std / (N - ddof))
    return (mean, std)


########## 2D Stats Calculator: Axis 0 ##########
def cy_min_max_0(double[:,::1] x, int nthreads=1):
    """
    Calculates the min and max for an NxM 2D C-array across axis 0. Very fast with no copying and
    parallelized (although default is 1 thread and can use at most M threads). Returns a 2xM NumPy
    array.
    """
    from numpy import empty, full
    cdef intp N = x.shape[0], M = x.shape[1]
    if M == 0: return empty((2,0))
    if N == 0: return full((2,M), NAN)
    out = empty((2,M))
    nthreads = get_nthreads(nthreads, M)
    cdef Range r
    cdef double[::1] mins = out[0], maxs = out[1]
    with nogil:
        if nthreads == 1: __min_max_0(x, 0, M, mins, maxs)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(M)
                __min_max_0(x, r.start, r.stop, mins, maxs)
    return out

cdef inline void __min_max_0(double[:,::1] x, intp L, intp R, double[::1] mins, double[::1] maxs) nogil:
    cdef intp i, j, N = x.shape[0]
    cdef double min, max
    for j in xrange(L, R):
        min = max = x[0,j]
        for i in xrange(1, N):
            if   x[i,j] < min: min = x[i,j]
            elif x[i,j] > max: max = x[i,j]
        mins[j] = min
        maxs[j] = max


def cy_mean_stdev_0(double[:,::1] x, double ddof=0.0, int nthreads=1):
    """
    Calculates the mean and std dev for an NxM 2D C-array across axis 0. Very fast with no data
    copying and parallelized (although default is 1 thread and can use at most M threads). Returns
    a 2xM NumPy array.
    """
    from numpy import empty, full
    cdef intp N = x.shape[0], M = x.shape[1]
    if M == 0: return empty((2,0))
    if N == 0: return full((2,M), NAN)
    out = empty((2,M))
    if N == 1:
        out[0] = x[0,:]
        out[1] = 0.0 if ddof != 1.0 else NAN
        return out
    nthreads = get_nthreads(nthreads, M)
    cdef Range r
    cdef double[::1] means = out[0], stds = out[1]
    with nogil:
        if nthreads == 1:
            __mean_stdev_0(x, 0, M, means, stds, ddof)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(M)
                __mean_stdev_0(x, r.start, r.stop, means, stds, ddof)
    return out

cdef inline void __mean_stdev_0(double[:,::1] x, intp L, intp R, double[::1] means, double[::1] stds, double ddof) nogil:
    cdef intp i, j, N = x.shape[0]
    cdef double mean, std, d
    for j in xrange(L, R):
        mean = x[0,j]
        for i in xrange(1, N): mean += x[i,j]
        mean /= N
        d = x[0,j]-mean; std = d*d
        for i in xrange(1, N): d = x[i,j]-mean; std += d*d
        means[j] = mean
        stds[j] = sqrt(std / (N-ddof))


########## 2D Stats Calculator: Axis 1 ##########
def cy_min_max_1(double[:,::1] x, int nthreads=1):
    """
    Calculates the min and max for an NxM 2D C-array across axis 1. Very fast with no copying and
    parallelized (although default is 1 thread and can use at most N threads). Returns a 2xM NumPy
    array.
    """
    from numpy import empty, full
    cdef intp N = x.shape[0], M = x.shape[1]
    if N == 0: return empty((2,0))
    if M == 0: return full((2,N), NAN)
    out = empty((2,N))
    nthreads = get_nthreads(nthreads, N)
    cdef Range r
    cdef double[::1] mins = out[0], maxs = out[1]
    with nogil:
        if nthreads == 1: __min_max_1(x, 0, N, mins, maxs)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(N)
                __min_max_1(x, r.start, r.stop, mins, maxs)
    return out

cdef inline void __min_max_1(double[:,::1] x, intp L, intp R, double[::1] mins, double[::1] maxs) nogil:
    cdef intp i, j, M = x.shape[1]
    cdef double min, max
    for i in xrange(L, R):
        min = max = x[i,0]
        for j in xrange(1, M):
            if x[i,j] < min: min = x[i,j]
            if x[i,j] > max: max = x[i,j]
        mins[i] = min
        maxs[i] = max
        
def cy_mean_stdev_1(double[:,::1] x, double ddof=0.0, int nthreads=1):
    """
    Calculates the mean and std dev for an NxM 2D C-array across axis 1. Very fast with no data
    copying and parallelized (although default is 1 thread and can use at most N threads). Returns
    a 2xM NumPy array.
    """
    from numpy import empty, full
    cdef intp N = x.shape[0], M = x.shape[1]
    if N == 0: return empty((2,0))
    if M == 0: return full((2,N), NAN)
    out = empty((2,N))
    if M == 1:
        out[0] = x[:,0]
        out[1] = 0.0 if ddof != 1.0 else NAN
        return out
    nthreads = get_nthreads(nthreads, N)
    cdef Range r
    cdef double[::1] means = out[0], stds = out[1]
    with nogil:
        if nthreads == 1:
            __mean_stdev_1(x, 0, N, means, stds, ddof)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(N)
                __mean_stdev_1(x, r.start, r.stop, means, stds, ddof)
    return out

cdef inline void __mean_stdev_1(double[:,::1] x, intp L, intp R, double[::1] means, double[::1] stds, double ddof) nogil:
    cdef intp i, j, M = x.shape[1]
    cdef double mean, std, d
    for i in xrange(L, R):
        mean = x[i,0]
        for j in xrange(1, M):
            mean += x[i,j]
        mean /= M
        d = x[i,0]-mean; std = d*d
        for j in xrange(1, M): d = x[i,j]-mean; std += d*d
        means[i] = mean
        stds[i] = sqrt(std / (M-ddof))


########## Percentile Calculator ##########
DEF SIZEOF_DBL=8 # cannot do sizeof(double) for Cython compile-time constants...
DEF COPY_SIZE=1024*1024//SIZEOF_DBL # copy at most 1 MB of data

def cy_percentile(double[:] x, q, bint overwrite=False, int nthreads=1):
    """
    Finds the value in x for the given percentile(s) in q. For very large data this will calculate
    the approximate value using a histogram of the data. This is repeated until the remaining data
    is less than 1MB when it switches to using the quickselect algorithm with a copy of the data
    (since quickselect modifies the data). If overwrite is True then this always uses quickselect
    from the beginning.
    
    The argument q can be given as a single value or a sequence of values. Several computations are
    preserved between the seperate percentiles if given many at once so it is significantly more
    efficient. For example, running 3 percentiles at the same time is ~25-35% faster than doing
    each separately, particularily if selecting with a histogram is needed and multithreaded.
    
    The number of threads is only used during histogramming of the data and not during quickselect.
    It is used to get the min and max of the data (using cy_min_max) and if there is more than one
    percentile to calculate then each percentile will use a separate thread.
    """
    from numpy import empty
    
    # Convert q to k
    cdef bint scalar
    cdef intp[::1] ks
    cdef double[::1] fs
    ks,fs,scalar = q2k(q, x.shape[0])
    cdef NK = ks.shape[0]
    
    # Calculate percentile and return output
    cdef double[::1] out = empty((NK,))
    if NK != 0: percentile(x, ks, fs, out, 1, overwrite)
    return out[0] if scalar else out.base

def cy_percentile_0(double[:,::1] x, q, bint overwrite=False, int nthreads=1):
    """
    Runs cy_percentile along with axis=0 returning a 1D or 2D array of values. This divides up the
    columns among the threads. Any leftover threads get used by cy_percentile.
    """
    return percentile_axis([x[:,i] for i in xrange(x.shape[1])], q, nthreads, overwrite)

def cy_percentile_1(double[:,::1] x, q, bint overwrite=False, int nthreads=1):
    """
    Runs cy_percentile along with axis=1 returning a 1D or 2D array of values. This divides up the
    rows among the threads. Any leftover threads get used by cy_percentile.
    """
    return percentile_axis([x[i,:] for i in xrange(x.shape[0])], q, nthreads, overwrite)

cdef percentile_axis(list xs, q, int nthreads, bint overwrite):
    """Internal function for cy_precentile_0 axis cy_precentile_1."""
    from numpy import empty
    cdef intp N = len(xs), i
    cdef int[:] nts
    
    # Convert q to k
    cdef bint scalar
    cdef intp[::1] ks
    cdef double[::1] fs
    ks,fs,scalar = q2k(q, 1 if N == 0 else xs[0].shape[0])
    cdef NK = ks.shape[0]
    
    # Calculate percentiles and return output
    cdef double[:,::1] out = empty((NK, N))
    if NK != 0 and N != 0:
        nthreads = get_nthreads(nthreads, NK*N)
        if N == 1: percentile(xs[0], ks, fs, out[:,0], nthreads, overwrite)
        elif nthreads == 1:
            for i in xrange(N): percentile(xs[i], ks, fs, out[:,i], 1, overwrite)
        elif nthreads <= N:
            for i in prange(N, num_threads=N, nogil=True, schedule='static'):
                with gil: percentile(xs[i], ks, fs, out[:,i], 1, overwrite)
        else: # nthreads > N
            nts = divy_threads(nthreads, N)
            for i in prange(N, num_threads=N, nogil=True, schedule='static'):
                with gil: percentile(xs[i], ks, fs, out[:,i], nts[i], overwrite)
    return out.base[0,:] if scalar else out.base

cdef inline int[:] divy_threads(int nthreads, intp n):
    """Divies up nthreads into n groups as equally as possible"""
    cdef int base = nthreads // n, rem = nthreads % n, i
    cdef int[::1] nts = array(shape=(n,), itemsize=sizeof(int), format="i")
    for i in xrange(n): nts[i] = base + (i<rem)
    return nts

cdef tuple q2k(qs, intp N):
    """Converts a single or list of percentiles (0.0-1.0) to indices and fractions."""
    from numpy import empty, intp as npy_intp
    from collections import Sequence
    cdef bint scalar = not isinstance(qs, Sequence)
    if scalar: qs = [qs]
    cdef intp NK = len(qs), i
    cdef intp[::1] ks = empty((NK,), npy_intp)
    cdef double[::1] fs = empty((NK,))
    cdef double q,k
    for i,q in enumerate(qs):
        if q < 0.0 or q > 1.0: raise ValueError('q')
        k = q*(N-1)
        ks[i] = <intp>floor(k)
        fs[i] = k-ks[i]
    return ks, fs, scalar

cdef int percentile(double[:] xs, intp[::1] ks, double[::1] fs, double[:] out, int nthreads, bint overwrite) except -1:
    """Core internal function for cy_precentile, cy_precentile_0, cy_precentile_1."""
    cdef intp N = xs.shape[0], NK = ks.shape[0], i
    cdef bint contiguous = xs.strides[0] == sizeof(double)
    cdef double[::1] x = xs if contiguous else None
    overwrite = overwrite and contiguous

    # Check if we go straight into quickselect
    if overwrite or N <= COPY_SIZE:
        if not overwrite: x = xs.copy()
        with nogil:
            for i in xrange(NK): out[i] = quickselect_weighted(&x[0], N, ks[i], fs[i])
        return 0
    
    # Histogram the data to reduce the values we have to search
    cdef intp nbins = <intp>sqrt(N)
    # Get the min and max of the range
    cdef double bs_inv, min, max
    min,max = cy_min_max(xs, nthreads)
    max = nextafter(max, INFINITY) # we need to make the max value *exclusive*
    # Allocate the bins
    cdef intp* bins = <intp*>malloc((nbins+1)*sizeof(intp))
    if bins is NULL: raise MemoryError()
    # Adjust the number of threads
    nthreads = get_nthreads(nthreads, NK)
    cdef int err = 0
    cdef int* p_err = &err
    with nogil:
        # Calculate the counts in the bins
        bs_inv = nbins/(max-min)
        memset(bins, 0, (nbins+1)*sizeof(intp))
        if contiguous:
            for i in xrange(N): bins[<intp>((x[i]-min)*bs_inv)] += 1
        else:
            for i in xrange(N): bins[<intp>((xs[i]-min)*bs_inv)] += 1
        bins[nbins-1] += bins[nbins] # catch the overflow values without using ifs in the loop
        # Check the bins and calculate the values at the percentiles
        if nthreads == 1:
            for i in xrange(NK):
                clear_errno()
                out[i] = percentile_recurse(xs, bins, nbins, min, (max-min)/nbins, ks[i], fs[i], i==NK-1)
                if out[i] == -INFINITY and errno != 0: err = errno; break
        else:
            for i in prange(NK, num_threads=nthreads):
                clear_errno()
                out[i] = percentile_recurse(xs, bins, nbins, min, (max-min)/nbins, ks[i], fs[i], False)
                if out[i] == -INFINITY and errno != 0: p_err[0] = errno; break
            free(bins)
            
    # Check for errors
    if err != 0:
        if   err == ENOMEM: raise MemoryError()
        elif err == ERANGE: raise ValueError()
        raise OSError(err, strerror(err))
    return 0

cdef inline void clear_errno() nogil:
    global errno
    errno = 0
    
cdef double percentile_recurse(double[:] x, intp* bins, intp nbins, double min, double bin_sz, intp k, double f, bint free_bins) nogil:
    """
    Finds the bin that contains the k-th smallest value and proceedes with quickselect or another
    histogram run. If the bin that contains the value would still require more than 1MB of data to
    copy, then it recurses. Otherwise we copy the values out and perform quickselect. If free_bins
    is True then the bins pointer is given to free once it is no longer needed.
    
    Some minor optimizations that could be done if recursing:
        Record the minimum and maximum found in each bin and use those values to recurse with
            (which would make bin_size smaller and thus more granular)
        'Trim' the x data to automatically skip leading and trailing values not in the recursed bin
            (gets tricky since k would also have to be updated as well)
    However we are unlikely to recurse that much so these would likely be wasted.
    """
    cdef intp i, j = 0, cnt = 0, N = x.shape[0], Nb
    cdef double max, v
    cdef double* data
    for i in xrange(nbins):
        cnt += bins[i]
        if cnt > k:
            # Found the bin!
            max = (i+1)*bin_sz+min; min += i*bin_sz # min/max of the bin
            Nb = bins[i]; k -= cnt-Nb               # size of bin and k within the bin
            if free_bins: free(bins)
            if min == max: return min # found the value itself!
            elif Nb <= COPY_SIZE:
                # Data has shrunk enough to be copied and sent to quickselect
                data = <double*>malloc(Nb*sizeof(double))
                if data is NULL: errno = ENOMEM; return -INFINITY
                for i in xrange(N):
                    if min <= x[i] < max: data[j] = x[i]; j += 1
                v = quickselect_weighted(data, Nb, k, f)
                free(data)
                return v
            # Data is still too large to be copied, histogram again
            return histselect(x, min, max, Nb, k, f)
    if free_bins: free(bins)
    errno = ERANGE
    return -INFINITY

cdef double histselect(double[:] x, double min, double max, intp N, intp k, double f) nogil:
    """Histograms the data recursively to find the k-th smallest value."""
    cdef intp i, nbins = <intp>sqrt(N)
    cdef intp* bins = <intp*>malloc((nbins+1)*sizeof(intp))
    if bins is NULL: errno = ENOMEM; return -INFINITY
    memset(bins, 0, (nbins+1)*sizeof(intp))
    cdef double bs_inv = nbins/(max-min)
    for i in xrange(x.shape[0]):
        if min <= x[i] < max: bins[<intp>((x[i]-min)*bs_inv)] += 1
    bins[nbins-1] += bins[nbins] # catch the overflow values without using ifs in the loop
    return percentile_recurse(x, bins, nbins, min, (max-min)/nbins, k, f, True)

cdef inline double quickselect_weighted(double* x, intp N, intp k, double f) nogil:
    """
    Finds the k-th smallest and (k+1)-th smallest values in x and weights them according to f.
    Note that if f is 0 or k == N-1 then only the k-th smallest is returned.
    """
    if f == 0.0 or k == N-1: return quickselect(x, N, k)
    return (1-f)*quickselect(x, N, k) + f*quickselect(x, N, k+1)
    
cdef double quickselect(double* x, intp N, intp k) nogil:
    """
    Finds the k-th smallest value in x using the quickselect algorithm. This will change values in
    the array x as it needs to so if overwriting is not desired make sure to copy the data first.
    """
    cdef intp L = 0, R = N-1, i, j
    cdef double p
    while L < R:
        i = L; j = R
        p = x[k]
        # do the left and right scan until the pointers cross
        while i <= j:
            # scan from the left then scan from the right
            while x[i] < p: i += 1
            while p < x[j]: j -= 1
            # now swap values if they are in the wrong part
            if i <= j: swap(x, i, j); i += 1; j -= 1
        # update the scan range
        if j < k: L = i
        if k < i: R = j
    return x[k]
    
cdef inline void swap(double* x, intp i, intp j) nogil:
    cdef double tmp = x[i]
    x[i] = x[j]
    x[j] = tmp
