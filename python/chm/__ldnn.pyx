#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""
CHM LDNN functions written in Cython. These includes parts of kmeansML, distSqr,
UpdateDiscriminants, and UpdateDiscriminants_SB (last two are renamed though).

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from pysegtools.general.cython.npy_helper cimport *
import_array()

from libc.math cimport exp, sqrt
ctypedef double* dbl_p

#################### General ####################

def downsample(intp min, double[:,:] X, char[::1] Y, intp downsample=10):
    """
    The full data set is in X and Y specifies which rows in X to use. If the number of Trues in Y is
    small, all of those rows in X are used. If it is large then most samples are skipped according
    to `downsample`.
    
    This code is equivilent to:
        from numpy import flatnonzero, empty
        n_trues = Y.sum()
        if n_trues < min:
            raise ValueError('Not enough data - either increase the data size or lower the number of levels (%d < %d)' % (n_trues,k))
        if n_trues <= min*downsample:
            out = empty((n_trues, X.shape[0]))
            if X.flags.c_contiguous: X.compress(Y, 1, out.T)
            else:                    X.T.compress(Y, 0, out)
        else:
            out = empty((((n_trues+downsample-1)//downsample), X.shape[0]))
            idx = flatnonzero(Y)[::downsample]
            if X.flags.c_contiguous: X.take(idx, 1, out.T)
            else:                    X.T.take(idx, 0, out)
        return out

    However this is ~1.25x (for C ordered X) to ~2.25x (for F ordered X) faster and uses no temporary
    memory allocations. This is ~6x faster when X is Fortran-ordered but even on C-ordered data this
    is significantly faster than copying the X data to a new array with Fortran ordering.

    Inputs:
        min         minimum number of samples to keep
        X           m-by-n double matrix
        Y           n length boolean vector (given as chars since Cython does not understand boolean type)
        downsample  the amount of downsampling to do, defaults to 10

    Returns:
        X'          n*-by-m C-ordered matrix (where n* is the selected/downsampled n)
    """
    cdef intp m = X.shape[0], n = X.shape[1], n_trues = 0, skip = downsample, i, j = 0
    
    assert(Y.shape[0] == n)

    # Count the number of trues: n_trues = Y.sum()
    for i in xrange(n): n_trues += <bint>Y[i] # get the number of True values
    if n_trues < min:
        raise ValueError('Not enough data - either increase the data size or lower the number of levels (%d < %d)' % (n_trues,min))
    
    # X.take(flatnonzero(Y), 1)  (possibly downsampled)
    cdef bint ds = n_trues > min*downsample
    cdef intp n_out = (n_trues+downsample-1)//downsample if ds else n_trues
    cdef double[:,::1] data = PyArray_EMPTY(2, [n_out, m], NPY_DOUBLE, False)
    if ds:
        for i in xrange(n):
            if Y[i]:
                if skip == downsample: data[j,:] = X[:,i]; j += 1; skip = 0
                skip += 1
    else:
        for i in xrange(n):
            if Y[i]: data[j,:] = X[:,i]; j += 1
    return data.base

def stddev(double[:,::1] X):
    """
    Calculates the standard deviation of an n-by-m 2D C-array across axis 0. Equivalent to:
        np.std(X, 0)
    except it is significantly faster (~4x in testing) and uses O(m) extra memory unlike
    the Numpy function which uses O(n*m) memory.
    """
    # I also tested a parallel version of this and with 2 threads it was ~15% faster and
    # barely benefited from a third or more thread. It made the code significantly more
    # complex and even for very large datasets that is ~10ms right before a process that
    # is going to take minutes so it isn't worth it.
    cdef intp N = X.shape[0], M = X.shape[1], i, j
    cdef double[::1] stds = PyArray_ZEROS(1, &M, NPY_DOUBLE, False)
    cdef double[::1] means = PyArray_ZEROS(1, &M, NPY_DOUBLE, False)
    with nogil:
        for i in xrange(N):
            for j in xrange(M): means[j] += X[i,j]
        for j in xrange(M): means[j] = means[j] / N
        for i in xrange(N):
            for j in xrange(M): stds[j] += (X[i,j]-means[j])*(X[i,j]-means[j])
        for j in xrange(M): stds[j] = sqrt(stds[j]/N)
    return stds.base

def run_kmeans(intp k, X, Y, bint whiten=False, bint scipy=False):
    """
    Downsample, possibly 'whiten', and run k-means on the data.

    Inputs:
        X       m-by-n matrix where m is the number of features and n is the number of samples
        Y       n-length array of bool labels
        
    Parameters:
        k       number of clusters to make
        whiten  scale each feature vector in X to have a variance of 1 before running k-means

    Returns:
        means   k-by-m matrix, cluster means/centroids
    """
    # NOTE: This could be easily outside of Cython but it calls 3 Cython functions in a row and that is about it...
    from numpy import int8
    
    assert(X.ndim == 2 and Y.ndim == 1 and X.shape[1] == Y.shape[0] and Y.dtype == bool and X.dtype == float)
    
    # Downsample the data (always creates a copy)
    data = downsample(k, X, Y.view(int8))
    
    # 'Whiten' the data (make variance of each feature equal to 1)
    if whiten:
        sd = stddev(data)
        sd[sd == 0] = 1 # avoid divide-by-0, turning it into no-scaling (should only ever be 0 for the column of all 1s anyways)
        data /= sd
    
    # Calculate clusters using kmeans
    if scipy:
        from scipy.cluster.vq import kmeans
        clusters = kmeans(data, k)[0]
    else:
        clusters = kmeansML(k, data)

    # Un-whiten the clusters
    if whiten: clusters *= sd

    return clusters


#################### Shuffling ####################

#from libc.stdlib cimport rand, srand, RAND_MAX
cdef extern from 'randomkit.h' nogil:
    ctypedef struct rk_state:
        pass
    ctypedef enum rk_error:
        pass
    int rk_randomseed(rk_state *state) nogil
    unsigned long rk_interval(unsigned long max, rk_state *state) nogil

cdef rk_state rng
cdef bint rng_inited = False

cdef void init_rng() nogil:
    global rng_inited
    rk_randomseed(&rng)
    rng_inited = True

cpdef inline void shuffle(intp[::1] arr, intp sub = -1) nogil:
    """
    Fisher–Yates In-Place Shuffle. It is optimal efficiency and unbiased (assuming the RNG is
    unbiased). This uses the random-kit RNG bundled with Numpy. If sub is given only that many
    elements are shuffled at the END of the array (they are shuffled with the entire array
    though).
    """
    if not rng_inited: init_rng()
    cdef intp i, j, t, n = arr.shape[0], end = 0 if sub == -1 else n-sub-1
    for i in xrange(n-1, end, -1):
        j = rk_interval(i, &rng)
        t = arr[j]; arr[j] = arr[i]; arr[i] = t
    
    # The original implementation used rand()/RAND_MAX and was seriously flawed. On some systems
    # any training set with more than 32767 samples (a single 180x180 image would get past that) it
    # would do a divide-by-0 almost every time. Also, the results would be skewed because rand()
    # was not very robust and the method used for scaling the output was also not great.
    #for i in xrange(n-1):
    #   j = i + rand() / (RAND_MAX / (n - i) + 1)
    #   t = arr[j]; arr[j] = arr[i]; arr[i] = t


#################### K-Means ML ####################

# This requires SciPy 0.16.0 released in Aug 2015
from scipy.linalg.cython_blas cimport dgemm

DEF KMEANS_ML_MAX_ITER      = 100
DEF KMEANS_ML_ETOL          = 0.0
DEF KMEANS_ML_DTOL          = 0.0

DEF KMEANS_ML_RATE          = 3
DEF KMEANS_ML_MIN_N         = 50
DEF KMEANS_ML_MAX_RETRIES   = 3

class ClusteringWarning(UserWarning): pass

cpdef kmeansML(intp k, double[:,::1] data):
    """
    Mulit-level K-Means. Tries very hard to always return k clusters. The parameters are now
    compile-time constants and no longer returns membership or RMS error, however these could be
    added back fairly easily (they are available in this function, just not returned).

    Parameter values:
        maxiter 100
        dtol    0.0
        etol    0.0

    Inputs:
        k       number of clusters
        data    n-by-d matrix where n is the number of samples and d is the features per sample

    Returns:
        means   k-by-d matrix, cluster means/centroids

    Originally by David R. Martin <dmartin@eecs.berkeley.edu> - October 2002

    Jeffrey Bush, 2015-2017, NCMIR, UCSD
    Converted into Python/Cython and optimized greatly
    """
    cdef intp n = data.shape[0], d = data.shape[1]
    cdef ndarray means      = PyArray_EMPTY(2, [k, d], NPY_DOUBLE, False)
    cdef ndarray means_next = PyArray_EMPTY(2, [k, d], NPY_DOUBLE, False)
    cdef ndarray membership = PyArray_EMPTY(1, &n, NPY_INTP, False)
    cdef ndarray counts     = PyArray_EMPTY(1, &k, NPY_INTP, False)
    cdef ndarray temp       = PyArray_EMPTY(2, [k, n], NPY_DOUBLE, False)
    cdef double rms2 = __kmeansML(k, data, means, means_next, membership, counts, temp)
    return means

cdef double __kmeansML(intp k, double[:,::1] data, double[:,::1] means, double[:,::1] means_next,
                       intp[::1] membership, intp[::1] counts, double[:,::1] temp, int retry=1) except -1.0:
    """
    Mulit-level K-Means core. Originally the private function kmeansInternal.

    Inputs:
        k           number of clusters
        data        n-by-d matrix where n is the number of samples and d is the features per sample
        retry       the retry count, should be one except for recursive calls
        
    Outputs:
        means       k-by-d matrix, cluster means/centroids
        means_next  k-by-d matrix used as a temporary
        membership  n length vector giving which samples belong to which clusters
        counts      k length vector giving the size of each cluster
        temp        k-by-n matrix used as a temporary
        <return>    RMS^2 error
    """
    cdef intp n = data.shape[0], d = data.shape[1], i, j
    cdef bint has_empty = False, converged
    cdef double S, max_sum
    cdef double[:,::1] means_orig = means, means_next_orig = means_next, means_tmp

    # Compute initial means
    cdef intp coarseN = (n+KMEANS_ML_RATE//2)//KMEANS_ML_RATE
    if coarseN < KMEANS_ML_MIN_N or coarseN < k:
        # pick random points for means
        random_subset(data, k, means)
    else:
        # recurse on random subsample to get means - O(coarseN) allocation
        __kmeansML(k, random_subset(data, coarseN), means, means_next, membership, counts, temp, 0)
    
    import numpy as np
    
    # Iterate
    with nogil:
        rms2 = km_update(data, means, means_next, membership, counts, temp)
        for _ in xrange(KMEANS_ML_MAX_ITER - 1):
            # Save last state
            prevRms2 = rms2
            means_tmp = means_next; means_next = means; means = means_tmp

            # Compute cluster membership, RMS^2 error, new means, and cluster counts
            rms2 = km_update(data, means, means_next, membership, counts, temp)
            if rms2 > prevRms2:
                with gil: raise RuntimeError('rms should always decrease')

            # Check for convergence
            IF KMEANS_ML_ETOL==0.0:
                converged = prevRms2 == rms2
            ELSE:
                # 2*(rmsPrev-rms)/(rmsPrev+rms) <= etol
                converged = sqrt(prevRms2/rms2) <= (2 + KMEANS_ML_ETOL) / (2 - KMEANS_ML_ETOL)
            if converged:
                max_sum = 0.0
                for i in xrange(k):
                    S = 0.0
                    for j in xrange(d): S += (means[i,j] - means_next[i,j]) * (means[i,j] - means_next[i,j])
                    if S > max_sum: max_sum = S
                if max_sum <= KMEANS_ML_DTOL*KMEANS_ML_DTOL: break

        for i in xrange(k):
            if counts[i] == 0:
                has_empty = True
                break

    if has_empty:
        # If there is an empty cluster, then re-run kmeans.
        # Retry a fixed number of times
        from warnings import warn
        if retry < KMEANS_ML_MAX_RETRIES:
            warn.warn('Re-running kmeans due to empty cluster.', ClusteringWarning)
            return __kmeansML(k, data, means_orig, means_next_orig, membership, counts, temp, retry+1)
        else:
            warn.warn('There is an empty cluster.', ClusteringWarning)

    # At this point the means data is in means_next
    # We need to make sure that refers to the means_orig array or copy the data over
    elif &means_orig[0,0] != &means_next[0,0]: means_orig[:,:] = means_next[:,:]

    return rms2

cdef double[:,::1] random_subset(double[:,::1] data, Py_ssize_t n, double[:,::1] out = None):
    """Takes a random n rows from the data array."""
    cdef intp total = data.shape[0], m = data.shape[1], i
    assert(total >= n)
    if out is None: out = PyArray_EMPTY(2, [n, m], NPY_DOUBLE, False)
    cdef intp[::1] inds = PyArray_Arange(0, total, 1, NPY_INTP)
    shuffle(inds, n) # NOTE: shuffles n elements at the END of the array
    for i in xrange(n): out[i,:] = data[inds[total-n+i],:]
    return out
    
cdef double km_update(double[:,::1] data, double[:,::1] means, double[:,::1] means_next,
                      intp[:] membership, intp[::1] counts, double[:,::1] dist) nogil:
    """
    K-Means updating. Combines the original private functions computeMembership and computeMeans.
    Now returns RMS^2 instead of RMS.

    Inputs:
        data        n-by-d C matrix, sample features
        means       k-by-d C matrix, current means

    Outputs:
        means_next  k-by-d C matrix, updated means
        membership  n length vector, filled in with which samples belong to which clusters
        counts      k length vector, filled in with size of each cluster
        dist        k-by-n matrix used as a temporary
        <return>    RMS^2 error

    Where:
        k           number of clusters
        d           number of features per sample
        n           number of samples
    """

    # CHANGED: returns the RMS^2 instead of RMS
    cdef intp n = data.shape[0], d = data.shape[1], k = means.shape[0], i, j, p
    cdef double x, min_sum = 0.0

    # Compute cluster membership and RMS error
    distSqr(data, means, dist)
    cdef double[::1] mins = dist[0,:] # first row of z starts out as mins
    membership[:] = 0 # all mins are in row 0
    for j in xrange(1, k): # go through all other rows and check for minimums
        for i in xrange(n):
            if dist[j,i] < mins[i]: mins[i] = dist[j,i]; membership[i] = j
    for i in xrange(n): min_sum += mins[i]

    # Compute new means and cluster counts
    means_next[:,:] = 0.0
    counts[:] = 0
    for i in xrange(n):
        j = membership[i]
        for p in xrange(d): means_next[j,p] += data[i,p]
        counts[j] += 1
    for j in xrange(k):
        x = 1.0 / max(1, counts[j])
        for p in xrange(d): means_next[j,p] *= x

    # Return RMS^2 error
    return min_sum / n

cdef void distSqr(double[:,::1] x, double[:,::1] y, double[:,::1] z) nogil:
    """
    Return matrix of all-pairs squared distances between the vectors in the columns of x and y.
    Equivilent to:
        # approximately:  x^2 + y^2 - 2*x@y.T   (where @ is matrix multiplication)
        z = x.dot(y.T).T
        z *= -2
        z += (x*x).sum(1)[None,:]
        z += (y*y).sum(1)[:,None]
        return z
        
    INPUTS
        x       n-by-k C matrix
        y       m-by-k C matrix

    OUTPUT
        z       m-by-n C matrix

    This is an optimized version written in Cython. It works just like the original but is faster
    and uses less memory. The matrix multiplication is done with BLAS (using dgemm to be able to do
    the multiply and addition all at the same time). The squares and sums are done with looping.

    The original uses O((n+m)*(k+1)) bytes of memory while this allocates no memory.
    """
    # NOTE: this is a near-commutative operation, e.g. distSqr(x, y, z) is the same as distSqr(y, x, z.T)
    cdef int n = x.shape[0], m = y.shape[0], d = x.shape[1], i, j, p
    cdef double alpha = -2.0, beta = 1.0, sum2
    cdef double[:] temp = z[m-1,:] # last row in z is used as a temporary
    # temp = (x*x).sum(1)
    for i in xrange(n):
        sum2 = 0.0
        for p in xrange(d): sum2 += x[i,p]*x[i,p]
        temp[i] = sum2
    # z = temp[None,:] + (y*y).sum(1)[:,None]
    for j in xrange(m):
        sum2 = 0.0
        for p in xrange(d): sum2 += y[j,p]*y[j,p]
        for i in xrange(n): z[j,i] = sum2 + temp[i]
    # z += -2*x.dot(y.T)
    dgemm("T", "N", &n, &m, &d, &alpha, &x[0,0], &d, &y[0,0], &d, &beta, &z[0,0], <int*>&z.shape[1])
    

#################### Gradient Descent ####################

def gradient(double[::1] f, double[:,::1] g, double[:,:,::1] s, double[::1,:] x, double[::1] y, double[:,:,::1] grads):
    """
    Calculates the gradient of the error using equation 12/13 from Seyedhosseini et al 2013 for a batch
    of samples. The summed output across all samples in the batch is saved to the grads matrix.
    """
    grads[:,:,:] = 0.0
    cdef double c1, c2, c3
    cdef intp N = s.shape[0], M = s.shape[1], P = s.shape[2], n = x.shape[0], i, j, k, p
    with nogil:
        for p in xrange(P):
            c1 = -2.0*(y[p]-f[p])*(1.0-f[p])
            for i in xrange(N):
                c2 = c1/(1.0-g[i,p])*g[i,p]
                for j in xrange(M):
                    c3 = c2*(1.0-s[i,j,p])
                    for k in xrange(n):
                        grads[i,j,k] += c3*x[k,p]

def descent(double[:,:,::1] grads, double[:,:,::1] prevs, double[:,:,::1] W, double rate, double momentum):
    """
    Updates the weights based on gradient descent using the given learning rate and momentum. The grads
    are the summed values instead of the averages. The rate and momentum values should be per-sample. 
    """
    cdef intp N = grads.shape[0], M = grads.shape[1], n = grads.shape[2], i, j, k
    with nogil:
        for i in xrange(N):
            for j in xrange(M):
                for k in xrange(n):
                    prevs[i,j,k] = grads[i,j,k] + momentum*prevs[i,j,k]
                    W[i,j,k] -= rate*prevs[i,j,k]

def descent_do(double[:,:,::1] grads, double[:,:,::1] prevs, double[:,:,::1] W,
               intp[::1] i_order, intp[::1] j_order, double rate, double momentum):
    """
    Updates the weights based on gradient descent with dropout using the given learning rate (which should
    be scaled based on the batch size) and momentum. The matrices prevs and W represent the entire dataset
    and the i_order and j_order arrays which regions in those matrices to use. The grads matrix represents
    just the non-dropped out data.
    """
    cdef intp Nd = grads.shape[0], Md = grads.shape[1], n = grads.shape[2], i, j, k
    cdef dbl_p W_ij, p_ij
    with nogil:
        for i in xrange(Nd):
            for j in xrange(Md):
                p_ij = &prevs[i_order[i],j_order[j],0]
                W_ij = &W[i_order[i],j_order[j],0]
                for k in xrange(n):
                    p_ij[k] = grads[i,j,k] + momentum*p_ij[k]
                    W_ij[k] -= rate*p_ij[k]

def gradient_descent_dropout(double[:,:] X, char[::1] Y, double[:,:,::1] W,
                             const intp niters, const double rate, const double momentum, disp=None):
    """
    This is an optimized version of gradient descent that always has dropout=0.5 and batchsz=1. See
    chm.ldnn.gradient_descent for more information about the other parameters.
    
    Slightly faster when X is Fortran-ordered but not by much (possibly about 15%). However that would mean
    that the entire X array, which is normally C-ordered, would have to be copied and stored in memory and
    it is huge.
    """

    # Matrix sizes
    cdef intp N = W.shape[0], M = W.shape[1], n = W.shape[2], P = Y.shape[0], N2 = N//2, M2 = M//2

    # Allocate memory
    cdef intp[::1] order   = PyArray_Arange(0, P, 1, NPY_INTP)
    cdef intp[::1] i_order = PyArray_Arange(0, N, 1, NPY_INTP)
    cdef intp[::1] j_order = PyArray_Arange(0, M, 1, NPY_INTP)
    cdef double[:,:,::1] prevs = PyArray_ZEROS(3, [N,M,n], NPY_DOUBLE, False)
    cdef double[:,::1] s = PyArray_ZEROS(2, [N2,M2], NPY_DOUBLE, False) # stores shuffled data
    cdef double[::1]   g = PyArray_ZEROS(1, &N2,     NPY_DOUBLE, False) # stores shuffled data
    cdef double[::1]   x = PyArray_EMPTY(1, &n,      NPY_DOUBLE, False) # a single row from X
    cdef ndarray total_error = PyArray_EMPTY(1, &niters, NPY_DOUBLE, False)
    
    # Variables
    cdef intp i, p
    cdef double totalerror
    
    for i in xrange(niters):
        with nogil:
            totalerror = 0.0
            shuffle(order)
            for p in xrange(P):
                shuffle(i_order)
                shuffle(j_order)
                x[:] = X[:,order[p]] # copying here greatly increases overall speed 
                totalerror += _grad_desc_do(x, 0.1 + 0.8*<bint>Y[order[p]], W, prevs, i_order[:N2], j_order[:M2], s, g, rate, momentum)
        total_error[i] = sqrt(totalerror/P)
        if disp is not None: disp('Iteration #%d error=%f' % (i+1,total_error[i]))
    return total_error

cdef double _grad_desc_do(const double[::1] x, const double y, double[:,:,::1] W, double[:,:,::1] prevs,
                          const intp[::1] i_order, const intp[::1] j_order, double[:,::1] s, double[::1] g,
                          const double rate, const double momentum) nogil:
    """A single step and sample for gradient descent dropout."""
    cdef intp N2 = i_order.shape[0], M2 = j_order.shape[0], n = x.shape[0], i, j, k
    
    # Calculate the sigmas, gs, and classifier (eqs 9 and 10 from Seyedhosseini et al 2013) 
    cdef double f = 1.0, g_i, s_ij
    cdef dbl_p W_ij, p_ij
    for i in xrange(N2):
        g_i = 1.0
        for j in xrange(M2):
            s_ij = 0.0
            W_ij = &W[i_order[i],j_order[j],0]
            for k in xrange(n): s_ij += W_ij[k]*x[k]
            s_ij = 1.0/(1.0+exp(-s_ij))
            s[i,j] = s_ij
            g_i *= s_ij
        g[i] = g_i
        f *= 1.0-g_i
    f = 1.0-f
    
    # Calculate gradient (eqs 12 and 13 from Seyedhosseini et al 2013)
    # and perform a gradient descent step
    cdef double yf = y-f, c1 = -2.0*(1.0-f)*yf, c2, c3
    for i in xrange(N2):
        c2 = c1/(1.0-g[i])*g[i]
        for j in xrange(M2):
            p_ij = &prevs[i_order[i],j_order[j],0]
            W_ij = &W[i_order[i],j_order[j],0]
            c3 = c2*(1.0-s[i,j])
            for k in xrange(n):
                p_ij[k] = x[k]*c3 + momentum*p_ij[k]
                W_ij[k] -= rate*p_ij[k]

    # Calculate error (part of eq 11 from Seyedhosseini et al 2013)
    return yf*yf
