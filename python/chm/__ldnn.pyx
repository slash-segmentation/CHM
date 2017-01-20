#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""
CHM LDNN functions written in Cython. These includes parts of kmeansML, distSqrt,
UpdateDiscriminants, and UpdateDiscriminants_SB (last two are renamed though).

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from pysegtools.general.cython.npy_helper cimport *
import_array()

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport exp, sqrt

ctypedef int* int_p
ctypedef double* dbl_p
ctypedef const double* c_dbl_p
ctypedef const npy_bool* c_bool_p

cdef __print(s):
    """Like print(...) but pre-pends the current timestamp and forces a flush."""
    import sys, datetime
    print('%s %s'%(str(datetime.datetime.utcnow())[:19], s))
    sys.stdout.flush()
    
def calc_kmeans(int k, ndarray X, ndarray Y, int downsample=10):
    """
    Calculates kmeans on the given data. The full data set is in X and Y specifies which rows in X
    to use. If the number of Trues in Y is small, all of those rows in X are used. If it is large
    then most samples are skipped according to `downsample`.
    
    In the end this is essentially:
        `kmeans(k, X.take(flatnonzero(Y), 1))`
    or:
        `kmeans(k, X.take(flatnonzero(Y)[::downsample], 1))`
    depending on the number of Trues in Y and the value of k. The X.take(...) is much, much, faster
    than the above code (~140x for the second method) and uses only one output temporary memory
    allocation.

    Inputs:
        k       number of clusters
        X       m-by-n double matrix, Fortran-ordered
        Y       n long boolean vector
        downsample  the amount of downsampling to do, defaults to 10

    Returns:
        means   m-by-k Fortran matrix, cluster means/centroids
    """
    cdef intp m = PyArray_DIM(X, 0), n = PyArray_DIM(X, 1), n_trues = 0, skip = downsample
    # Count the number of trues
    cdef c_bool_p y = <c_bool_p>PyArray_DATA(Y), end = y + n
    while y < end: n_trues += y[0]; y += 1 # get the number of True values
    y = <c_bool_p>PyArray_DATA(Y) # reset pointer to the beginning
    if n_trues < k:
        print('%d < %d'%(n_trues,k))
        raise ValueError('Not enough data - either increase the data size or lower the number of levels')
    # X.take(flatnonzero(Y), 1)  (possibly downsampled)
    cdef bint ds = n_trues > k*downsample
    cdef ndarray Z = PyArray_EMPTY(2, [m, (n_trues+downsample-1)//downsample if ds else n_trues], NPY_DOUBLE, True) 
    cdef dbl_p x = <dbl_p>PyArray_DATA(X)
    cdef dbl_p z = <dbl_p>PyArray_DATA(Z)
    if ds:
        while y < end:
            if y[0]:
                if skip == downsample: memcpy(z, x, m*sizeof(double)); z += m; skip = 0
                skip += 1
            y += 1; x += m
    else:
        while y < end:
            if y[0]: memcpy(z, x, m*sizeof(double)); z += m
            y += 1; x += m
    # Run k-means
    return kmeansML(k, Z)


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

cdef inline void shuffle(intp[::1] arr, intp sub = -1) nogil:
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

cpdef ndarray kmeansML(int k, double[::1,:] data_f):
    """
    Mulit-level K-Means. Tries very hard to always return k clusters. The parameters are now
    compile-time constants and no longer returns membership or RMS error, however thse could be
    added back fairly easily.

    Parameter values:
        maxiter 100
        dtol    0.0
        etol    0.0

    Inputs:
        k           number of clusters
        data        d-by-n matrix, sample features
                    algorithm will be the fastest if this is Fortran-ordered

    Returns:
        means       d-by-k Fortran matrix, cluster means/centroids

    Originally by David R. Martin <dmartin@eecs.berkeley.edu> - October 2002

    Jeffrey Bush, 2015-2016, NCMIR, UCSD
    Converted into Python/Cython and optimized greatly
    """
    cdef double[:,::1] data = data_f.T
    # data is n x d (after transpose above)
    cdef intp n = data.shape[0], d = data.shape[1]
    cdef ndarray means = PyArray_EMPTY(2, [k, d], NPY_DOUBLE, False)
    cdef int_p membership = <int_p>malloc(n*max(sizeof(double),sizeof(int)))
    cdef int_p counts     = <int_p>malloc(k*sizeof(int))
    cdef dbl_p temp       = <dbl_p>malloc(n*k*sizeof(double))
    cdef dbl_p means_next = <dbl_p>malloc(k*d*sizeof(double))
    try:
        if membership is NULL or counts is NULL or temp is NULL or means_next is NULL: raise MemoryError()
        __kmeansML(k, data, means, 1, membership, counts, temp, means_next)
    finally:
        free(membership)
        free(counts)
        free(temp)
        free(means_next)
    # means are k x d (before transpose below)
    return means.T

cdef int __kmeansML(int k, double[:,::1] data, double[:,::1] means, int retry,
                    int_p membership, int_p counts, dbl_p temp, dbl_p means_next) except -1:
    """
    Mulit-level K-Means core. Originally the private function kmeansInternal.

    Inputs:
        k           number of clusters
        data        d-by-n Fortran matrix, sample features
        retry       the retry count, shold be one except for recursive calls

    Outputs:
        means       d-by-k Fortran matrix, cluster means/centroids
        membership  n length vector giving which samples belong to which clusters
        counts      k length vector giving the size of each cluster
        temp        n-by-m matrix used as a temporary
        means_next  d-by-k matrix used as a temporary

    Returns 0 if successful.
    """
    cdef intp n = data.shape[0], d = data.shape[1], i, j
    cdef bint has_empty = False, converged
    cdef double S, max_sum
    cdef dbl_p data_p    = &data[0,0]
    cdef dbl_p means_cur = &means[0,0]
    cdef dbl_p means_next_orig = means_next
    cdef dbl_p means_cur_i, means_next_i, means_tmp

    # Compute initial means
    cdef int coarseN = (n+KMEANS_ML_RATE//2)//KMEANS_ML_RATE
    if coarseN < KMEANS_ML_MIN_N or coarseN < k:
        # pick random points for means
        random_subset(data, k, means)
    else:
        # recurse on random subsample to get means - O(coarseN) allocation
        __kmeansML(k, random_subset(data, coarseN), means, 0, membership, counts, temp, means_next)

    # Iterate
    with nogil:
        rms2 = km_update(k, d, n, data_p, means_next, means_cur, membership, counts, temp)
        for _ in xrange(KMEANS_ML_MAX_ITER - 1):
            # Save last state
            prevRms2 = rms2
            means_tmp = means_next; means_next = means_cur; means_cur = means_tmp

            # Compute cluster membership, RMS^2 error, new means, and cluster counts
            rms2 = km_update(k, d, n, data_p, means_next, means_cur, membership, counts, temp)
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
                    means_next_i = means_next+i*d
                    means_cur_i  = means_cur +i*d
                    S = 0.0
                    for j in xrange(d): S += (means_cur_i[j] - means_next_i[j]) * (means_cur_i[j] - means_next_i[j])
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
            warn('Re-running kmeans due to empty cluster.')
            __kmeansML(k, data, means, retry+1, membership, counts, temp, means_next_orig)
        else:
            warn('There is an empty cluster.')

    # The means data is in means_next right now, make sure that that corresponds to the means ndarray
    elif means_next == means_next_orig: memcpy(means_cur, means_next, k*d*sizeof(double))

    return 0

cdef double[:,::1] random_subset(double[:,::1] data, Py_ssize_t n, double[:,::1] out = None):
    """Takes a random n rows from the data array."""
    cdef intp total = data.shape[0], m = data.shape[1], i
    assert(total >= n)
    if out is None: out = PyArray_EMPTY(2, [n, m], NPY_DOUBLE, False)
    cdef intp[::1] inds = PyArray_Arange(0, total, 1, NPY_INTP)
    shuffle(inds, n) # NOTE: shuffles n elements at the END of the array
    cdef dbl_p out_p  = <dbl_p>PyArray_DATA(out)
    cdef dbl_p data_p = <dbl_p>PyArray_DATA(data)
    for i in xrange(n): memcpy(out_p+i*m, data_p+inds[total-n+i]*m, m*sizeof(double))
    return out
    
cdef double km_update(int k, int d, int n, dbl_p data, dbl_p means_next, dbl_p means_cur,
                      int_p membership, int_p counts, dbl_p z) nogil:
    """
    K-Means updating. Combines the original private functions computeMembership and computeMeans.
    Now returns RMS^2 instead of RMS.

    Inputs:
        k           number of clusters
        d           number of features per sample
        n           number of samples
        data        d-by-n Fortran matrix, sample features
        means_cur   d-by-k Fortran matrix, current means

    Outputs:
        means_next  d-by-k Fortran matrix, updated means
        membership  n length vector, filled in with which samples belong to which clusters
                    used as a temporary for n doubles as well
        counts      k length vector, filled in with size of each cluster
        z           n-by-m matrix used as a temporary
        <return>    RMS^2 error
    """

    # CHANGED: returns the RMS^2 instead of RMS
    cdef int i, j, p, min_j
    cdef double x, min_sum = 0.0
    cdef dbl_p z_j, data_i, means_j

    # Compute cluster membership and RMS error
    distSqr(k, d, n, data, means_cur, z, <dbl_p>membership) # z is Fortran ordered, membership is just used as temporary memory
    cdef dbl_p mins = z # first column of z starts out as mins
    memset(membership, 0, n*sizeof(int)) # all mins are in column 0
    for j in xrange(1, k): # go through all other columns and check for minimums
        z_j = z+j*n
        for i in xrange(n):
            if z_j[i] < mins[i]: mins[i] = z_j[i]; membership[i] = j
    for i in xrange(n): min_sum += mins[i]

    # Compute new means and cluster counts
    memset(means_next, 0, k*d*sizeof(double))
    memset(counts,     0, k*sizeof(int))
    for i in xrange(n):
        j = membership[i]
        data_i = data+i*d
        means_j = means_next+j*d
        for p in xrange(d): means_j[p] += data_i[p]
        counts[j] += 1
    for j in xrange(k):
        x = 1.0 / max(1, counts[j])
        means_j = means_next+j*d
        for p in xrange(d): means_j[p] *= x

    # Return RMS^2 error
    return min_sum / n

cdef void distSqr(int m, int d, int n, dbl_p x, dbl_p y, dbl_p z, dbl_p temp) nogil:
    """
    Return matrix of all-pairs squared distances between the vectors in the columns of x and y.

    INPUTS
        m,d,n   size of the matrices
        x       d-by-n Fortran matrix (or n-by-d C matrix)
        y       d-by-m Fortran matrix (or m-by-d C matrix)

    OUTPUTS
        z       n-by-m Fortran matrix (or m-by-n C matrix)
        temp    n length vector

    This is an optimized version written in Cython. It works just like the original but is faster
    and uses less memory. The matrix multiplication is done with BLAS (using dgemm to be able to do
    the multiply and addition all at the same time). The squares and sums are done with looping.
    This is also parallelizable.

    The original uses O(n*d+m*d+n+m) bytes of memory while this uses O(n) bytes of memory (which is
    actually passed in as an argument and reused).
    """
    # NOTE: this is a near-commutative operation, e.g. distSqr(x, y) is the same as distSqr(y, x).T
    cdef int i, j, p
    cdef double alpha = -2.0, beta = 1.0, sum2
    cdef dbl_p x_i, y_j, z_j
    for i in xrange(n):
        x_i = x+i*d; sum2 = 0
        for p in xrange(d): sum2 += x_i[p]*x_i[p]
        temp[i] = sum2
    for j in xrange(m):
        y_j = y+j*d; sum2 = 0
        for p in xrange(d): sum2 = sum2 + y_j[p]*y_j[p]
        z_j = z+j*n
        for i in xrange(n): z_j[i] = sum2 + temp[i]
    dgemm("T", "N", &n, &m, &d, &alpha, x, &d, y, &d, &beta, z, &n)


#################### Gradient Descent ####################

# Note: combining gradient and descent is actually slower...

def gradient(double[::1] f, double[:,::1] g, double[:,:,::1] s, double[::1,:] x, double[::1] y, double[:,:,::1] grads):
    """
    Calculates the gradient of the error using equation 12/13 from Seyedhosseini et al 2013.
    Difference from the paper is that the value is not multiplied by -2 (that is instead handled
    later). The output is saved to the grads matrix.
    """
    grads[:,:,:] = 0
    cdef double c1, c2, c3
    cdef intp N = s.shape[0], M = s.shape[1], P = s.shape[2], n = x.shape[0], i, j, k, p
    with nogil:
        for p in xrange(P):
            c1 = (y[p]-f[p])*(1.0-f[p])
            for i in xrange(N):
                c2 = c1/(1.0-g[i,p])*g[i,p]
                for j in xrange(M):
                    c3 = c2*(1.0-s[i,j,p])
                    for k in xrange(n):
                        grads[i,j,k] += c3*x[k,p]

def descent(double[:,:,::1] grads, double[:,:,::1] prevs, double[:,:,::1] w, double rate, double momentum):
    """
    Updates the weights based on gradient descent using the given learning rate and momentum. The
    gradients should be off by a factor of -2.
    """
    cdef intp N = grads.shape[0], M = grads.shape[1], n = grads.shape[2], i, j, k
    with nogil:
        for i in xrange(N):
            for j in xrange(M):
                for k in xrange(n):
                    prevs[i,j,k] = grads[i,j,k] + momentum*prevs[i,j,k]
                    w[i,j,k] += rate*prevs[i,j,k]

def gradient_descent_dropout(double[::1,:] X, char[::1] Y, double[:,:,::1] W,
                             const intp niters=15, const double rate=0.05, const double momentum=0.5, const bint disp=True):
    """See ldnn.gradient_descent_dropout for documentation."""

    # Matrix sizes
    cdef intp N = W.shape[0], M = W.shape[1], n = W.shape[2], P = Y.shape[0], N2 = N//2, M2 = M//2

    # Allocate memory
    cdef intp[::1] order   = PyArray_Arange(0, P, 1, NPY_INTP)
    cdef intp[::1] i_order = PyArray_Arange(0, N, 1, NPY_INTP)
    cdef intp[::1] j_order = PyArray_Arange(0, M, 1, NPY_INTP)
    cdef double[:,:,::1] prevs = PyArray_ZEROS(3, [N,M,n], NPY_DOUBLE, False)
    cdef double[:,::1] s = PyArray_ZEROS(2, [N2,M2], NPY_DOUBLE, False) # stores shuffled data
    cdef double[::1]   g = PyArray_ZEROS(1, &N2,     NPY_DOUBLE, False) # stores shuffled data
    cdef ndarray total_error = PyArray_EMPTY(1, &niters, NPY_DOUBLE, False)

    # Variables
    cdef intp i, p
    cdef double totalerror, y
    cdef double[::1] x
    
    for i in xrange(niters):
        with nogil:
            totalerror = 0.0
            shuffle(order)
            for p in xrange(P):
                shuffle(i_order)
                shuffle(j_order)
                totalerror += _grad_desc_do(X[:,order[p]], 0.1 + 0.8*Y[order[p]], W, prevs,
                                            i_order[:N2], j_order[:M2], s, g, rate, momentum)
        total_error[i] = sqrt(totalerror/P)
        if disp: __print('       Epoch #%d error=%f' % (i+1,total_error[i]))
    return total_error

cdef double _grad_desc_do(const double[::1] x, const double y, double[:,:,::1] W, double[:,:,::1] prevs,
                          const intp[::1] i_order, const intp[::1] j_order, double[:,::1] s, double[::1] g,
                          const double rate, const double momentum) nogil:
    """A single step and sample for gradient descent - dropout."""
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
    cdef double yf = y-f, c1 = (1.0-f)*yf, c2, c3
    for i in xrange(N2):
        c2 = c1/(1.0-g[i])*g[i]
        for j in xrange(M2):
            p_ij = &prevs[i_order[i],j_order[j],0]
            W_ij = &W[i_order[i],j_order[j],0]
            c3 = c2*(1.0-s[i,j])
            for k in xrange(n):
                p_ij[k] = x[k]*c3 + momentum*p_ij[k]
                W_ij[k] += rate*p_ij[k]

    # Calculate error (part of eq 11 from Seyedhosseini et al 2013)
    return yf*yf
