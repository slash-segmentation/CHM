#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""
CHM Training functions written in Cython. These include LearnAndOrNetMEX (called learn), kmeansML,
distSqrt, UpdateDiscriminants, and UpdateDiscriminants_SB. The only exposed function is learn.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
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


#################### Learn And-Or Net ####################
DEF CLUSTER_DOWNSAMPLE=10

DEF N_GROUP_SB          = 10
DEF N_DISC_PER_GROUP_SB = 20
DEF N_GROUP             = 24
DEF N_DISC_PER_GROUP    = 24

def learn(ndarray X, ndarray Y, bint sb, int maxepoch, int nthreads=1):
    """
    Learn And-Or Network for CHM
    
    X           M by N training set, M is number of features and N is number of samples
                this includes an extra feature that will be initialized with the centroids
    Y           N-long boolean array of training labels
    sb          if True, runs in SB mode (see below)
    maxepoch    the number of refinement passes to do on the discriminants
    
    In SB mode:
        epsilon is 0.01, number of groups is 10, and number of discriminants per group is 20
        only updates discriminants after every 100 samples
        
    Not in SB mode:
        epsilon is 0.05 and number of groups and discriminants per group is 24
        updates half of the discriminants within half of the groups for each sample
    
    In either mode momentum is 0.5
    """
    # CHANGED: removed all remaining validation and regularization code (most of it had already been removed)
    # CHANGED: all optional params are hard-coded in the C since they were on their way there anyways
    # CHANGED: only returns discriminants
    # CHANGED: assumes the X array includes the extra feature for the centroids
    # TODO: use nthreads
    if PyArray_FLAGS(X) & NPY_ARRAY_FARRAY    != NPY_ARRAY_FARRAY:    raise ValueError('Bad X')
    if PyArray_FLAGS(Y) & NPY_ARRAY_FARRAY_RO != NPY_ARRAY_FARRAY_RO: raise ValueError('Bad Y')
    # Note: the extra feature is set to all 1s at this point and won't effect kmeans
    fill_last_row_with_1(X)
    cdef Py_ssize_t n_feats = PyArray_DIM(X, 0), n_samps = PyArray_DIM(X, 1)
    
    __print('    Clustering...')
    cdef int n_group, n_dpg
    if sb: n_group = N_GROUP_SB; n_dpg = N_DISC_PER_GROUP_SB
    else:  n_group = N_GROUP;    n_dpg = N_DISC_PER_GROUP
    cdef ndarray pos = calc_kmeans(n_group, X,  Y)
    cdef ndarray neg = calc_kmeans(n_dpg,   X, ~Y)
    
    __print('    Number of training samples = %d'%n_samps)

    cdef int n_disc = n_group*n_dpg
    cdef PyArray_Dims shape
    cdef intp dims[2]
    dims[0] = n_feats; dims[1] = n_disc; shape.len = 2; shape.ptr = dims
    cdef ndarray discriminants = PyArray_Newshape(pos[:,:,None] - neg[:,None,:], &shape, NPY_CORDER)
    cdef ndarray centroids     = PyArray_Newshape(pos[:,:,None] + neg[:,None,:], &shape, NPY_CORDER)
    del pos, neg
    cdef ndarray disc_norm = norm(discriminants)
    discriminants /= disc_norm
    centroids *= discriminants
    centroids = PyArray_Sum(centroids, 0, NPY_DOUBLE, disc_norm)
    centroids *= -0.5
    fill_last_row_with_copy(discriminants, centroids)
    del centroids, disc_norm

    #from numpy.random import randn
    #discriminants = randn(n_feats+1, n_disc) #uncomment for random initialization

    __print('    Updating discriminants...')
    cdef c_dbl_p  x = <c_dbl_p>PyArray_DATA(X)
    cdef c_bool_p y = <c_bool_p>PyArray_DATA(Y)
    cdef dbl_p    disc = <dbl_p>PyArray_DATA(discriminants)
    cdef int retval
    with nogil:
        retval = UpdateDiscriminants_SB(x, y, n_feats, n_samps, maxepoch, disc) if sb else \
                 UpdateDiscriminants(x, y, n_feats, n_samps, maxepoch, disc)
    if retval < 0: raise MemoryError()

    return discriminants

cdef __print(s):
    """Like print(...) but pre-pends the current timestamp and forces a flush."""
    import sys, datetime
    print('%s %s'%(str(datetime.utcnow())[:19], s))
    sys.stdout.flush()
    
cdef void fill_last_row_with_1(ndarray X):
    """
    Fills the last row of a double matrix with 1.0, like X[-1,:] = 1.0. Works with srided arrays.
    """
    cdef intp nr = PyArray_DIM(X,0), nc = PyArray_DIM(X,1), sc = PyArray_STRIDE(X,1), i
    cdef dbl_p x = <dbl_p>(<char*>PyArray_DATA(X) + (nr-1)*PyArray_STRIDE(X,0))
    for i in xrange(nc): x[0] = 1.0; x = <dbl_p>(<char*>x+sc)

cdef void fill_last_row_with_copy(ndarray X, ndarray Y):
    """
    Fills the last row of a double matrix with the contents of another array, like X[-1,:] = Y.
    Works with srided arrays.
    """
    cdef intp nr = PyArray_DIM(X,0), nc = PyArray_DIM(X,1), i
    cdef intp sx = PyArray_STRIDE(X,1), sy = PyArray_STRIDE(Y,0)
    cdef dbl_p x = <dbl_p>(<char*>PyArray_DATA(X) + (nr-1)*PyArray_STRIDE(X,0))
    cdef dbl_p y = <dbl_p>PyArray_DATA(Y)
    for i in xrange(nc): x[0] = y[0]; x = <dbl_p>(<char*>x+sx); y = <dbl_p>(<char*>y+sy)
    
cdef ndarray calc_kmeans(int k, ndarray X, ndarray Y):
    """
    Calculates kmeans on the given data. The full data set is in X and Y specifies which rows in X
    to use. If the number of Trues in Y is small, all of those rows in X are used. If it is large
    then most samples are skipped according to CLUSTER_DOWNSAMPLE.
    
    In the end this is essentially:
        `kmeans(k, X.take(flatnonzero(Y), 1))`
    or:
        `kmeans(k, X.take(flatnonzero(Y)[::CLUSTER_DOWNSAMPLE], 1))`
    depending on the number of Trues in Y and the value of k. The X.take(...) is much, much, faster
    than the above code (~140x for the second method) and uses only one output temporary memory
    allocation.

    Inputs:
        k       number of clusters
        X       m-by-n double matrix, Fortran-ordered
        Y       n long boolean vector

    Returns:
        means   m-by-k Fortran matrix, cluster means/centroids
    """
    cdef intp m = PyArray_DIM(X, 0), n = PyArray_DIM(X, 1), n_trues = 0, skip = CLUSTER_DOWNSAMPLE
    cdef c_bool_p y = <c_bool_p>PyArray_DATA(Y), end = y + n
    while y < end: n_trues += y[0]; y += 1 # get the number of True values
    y = <c_bool_p>PyArray_DATA(Y) # reset pointer to the beginning
    if n_trues < k:
        print('%d < %d'%(n_trues,k))
        raise ValueError('Not enough data - either increase the data size or lower the number of levels')
    cdef bint downsample = n_trues > k*CLUSTER_DOWNSAMPLE
    cdef intp[2] dims
    dims[0] = m
    dims[1] = (n_trues+CLUSTER_DOWNSAMPLE-1)//CLUSTER_DOWNSAMPLE if downsample else n_trues
    cdef ndarray Z = PyArray_EMPTY(2, dims, NPY_DOUBLE, True) # X.take(flatnonzero(Y), 1)  (possibly downsampled)
    cdef dbl_p x = <dbl_p>PyArray_DATA(X)
    cdef dbl_p z = <dbl_p>PyArray_DATA(Z)
    if downsample:
        while y < end:
            if y[0]:
                if skip == CLUSTER_DOWNSAMPLE: memcpy(z, x, m*sizeof(double)); z += m; skip = 0
                skip += 1
            y += 1; x += m
    else:
        while y < end:
            if y[0]: memcpy(z, x, m*sizeof(double)); z += m
            y += 1; x += m
    return kmeansML(k, Z)

cdef ndarray norm(ndarray X):
    """Calculates sqrt(sum(X*X, axis=0)) with no intermediate memory. X must be Fortran-ordered."""
    cdef intp nr = PyArray_DIM(X,0), nc = PyArray_DIM(X,1), r, c
    cdef ndarray out = PyArray_ZEROS(1, &nc, NPY_DOUBLE, False)
    cdef dbl_p x = <dbl_p>PyArray_DATA(X)
    cdef dbl_p o = <dbl_p>PyArray_DATA(out)
    cdef double sum
    for c in xrange(nc):
        sum = 0
        for r in xrange(nr): sum += x[0]*x[0]; x += 1
        o[c] = sqrt(sum)
    return out


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

cdef inline void shuffle(intp *arr, const intp n, intp sub = -1) nogil:
    """
    Fisher–Yates In-Place Shuffle. It is optimal efficiency and unbiased (assuming the RNG is
    unbiased). This uses the random-kit RNG bundled with Numpy. If sub is given only that many
    elements are shuffled at the END of the array (they are shuffled with the entire array
    though).
    """
    if not rng_inited: init_rng()
    cdef intp i, j, t, end = 0 if sub == -1 else n-sub-1
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

cdef ndarray kmeansML(int k, ndarray data):
    """
    Mulit-level K-Means. Tries very hard to always return k clusters. The paramters are now
    compile-time constants and not longer returns membership or RMS error, however thse could be
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
    #data = PyArray_FROMANY(PyArray_Transpose(data, NULL), NPY_DOUBLE, 2, 2, NPY_ARRAY_CARRAY_RO)
    data = PyArray_Transpose(data, NULL)
    # data is n x d (after transpose above)
    cdef int n = PyArray_DIM(data, 0), d = PyArray_DIM(data, 1)
    cdef intp[2] dims
    dims[0] = k; dims[1] = d
    cdef ndarray means = PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
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
    return PyArray_Transpose(means, NULL)

cdef int __kmeansML(int k, ndarray data, ndarray means, int retry,
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
    cdef int n = PyArray_DIM(data, 0), d = PyArray_DIM(data, 1), i, j
    cdef bint has_empty = False, converged
    cdef double S, max_sum
    cdef dbl_p data_p    = <dbl_p>PyArray_DATA(data)
    cdef dbl_p means_cur = <dbl_p>PyArray_DATA(means)
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

cdef ndarray random_subset(ndarray data, Py_ssize_t n, ndarray out = None):
    """Takes a random n rows from the C-ordered data array."""
    cdef intp total = PyArray_DIM(data, 0), m = PyArray_DIM(data, 1), i
    cdef intp[2] dims
    assert total >= n
    if out is None:
        dims[0] = n; dims[1] = m
        out = PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
    cdef intp* inds = <intp*>malloc(total*sizeof(intp))
    if inds is NULL: raise MemoryError()
    for i in xrange(total): inds[i] = i
    shuffle(inds, total, n) # note: shuffles n elements at the END of the array
    inds += total-n
    cdef dbl_p out_p  = <dbl_p>PyArray_DATA(out)
    cdef dbl_p data_p = <dbl_p>PyArray_DATA(data)
    for i in xrange(n):
        memcpy(out_p+i*m, data_p+inds[i]*m, m*sizeof(double))
    inds -= total-n
    free(inds)
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


#################### Update Discriminants ####################

DEF EPSILON_SB  = 0.01
DEF MOMENTUM_SB = 0.5
DEF N_DISC_SB   = N_GROUP_SB*N_DISC_PER_GROUP_SB

DEF BSIZE = 100

cdef int UpdateDiscriminants_SB(c_dbl_p X, c_bool_p Y, const intp M, const intp N, const int maxepoch, double *discriminants) nogil:
    # CHANGED: from C to Cython
    # CHANGED: uses doubles instead of floats
    # CHANGED: all paramters except maxepoch are now compile-time constants
    # CHANGED: totalerror is no longer returned
    cdef intp* sample_order = <intp*>malloc(N*sizeof(intp))
    cdef double outputs[N_DISC_SB]
    cdef double outputsAND[N_GROUP_SB]
    cdef dbl_p prevupdates  = <dbl_p>malloc(N_DISC_SB*M*sizeof(double))
    cdef dbl_p updates      = <dbl_p>malloc(N_DISC_SB*M*sizeof(double))
    cdef double totalerror
    if sample_order is NULL or prevupdates is NULL or updates is NULL:
        free(sample_order)
        free(prevupdates)
        free(updates)
        return -1

    # e goes over epochs, i goes over N (samples), j goes over M (features)
    # g goes over N_GROUP_SB, d goes over N_DISC_PER_GROUP_SB
    # ii is the shuffled value of i
    cdef intp i, j, g, d, ii
    cdef int e

    # Temporary variables
    cdef double AND_c_muls, prod, sum, out, erro, mp, term2, update
    cdef c_dbl_p x # single sample from X
    cdef dbl_p outs, discs, updts, prevs

    for i in xrange(N): sample_order[i] = i
    memset(prevupdates, 0, N_DISC_SB*M*sizeof(double))
    memset(updates,     0, N_DISC_SB*M*sizeof(double))

    for e in xrange(maxepoch):
        shuffle(sample_order, N)
        totalerror = 0
        for i in xrange(N):
            ii = sample_order[i]
            x = X+ii*M

            AND_c_muls = 1
            outs = outputs
            discs = discriminants
            updts = updates
            for g in xrange(N_GROUP_SB):
                prod = 1
                for d in xrange(N_DISC_PER_GROUP_SB):
                    sum = 0
                    for j in xrange(M): sum += discs[0]*x[j]; discs += 1
                    out = 1/(1+exp(-sum))
                    outs[0] = out; outs += 1
                    prod *= out
                outputsAND[g] = prod
                AND_c_muls *= 1-prod
            erro = 0.1 + 0.8*Y[ii] - (1 - AND_c_muls)
            totalerror += erro*erro

            outs = outputs
            for g in xrange(N_GROUP_SB):
                mp = AND_c_muls/(1-outputsAND[g])*outputsAND[g]*erro
                for d in xrange(N_DISC_PER_GROUP_SB):
                    term2 = mp * (1 - outs[0]); outs += 1
                    for j in xrange(M): updts[0] += x[j]*term2; updts += 1

            if i%BSIZE == BSIZE-1:
                discs = discriminants
                updts = updates
                prevs = prevupdates
                for j in xrange(N_DISC_SB*M):
                    update = updts[0] + MOMENTUM_SB*prevs[0]
                    discs[0] += EPSILON_SB*update; discs += 1
                    prevs[0] = update;             prevs += 1
                    updts[0] = 0;                  updts += 1

        totalerror = sqrt(totalerror/N)
        with gil: __print('       Epoch #%d error=%f' % (e+1,totalerror))

    free(sample_order)
    free(prevupdates)
    free(updates)
    return 0

DEF EPSILON   = 0.05
DEF MOMENTUM  = 0.5
DEF N_GROUP_2 = (N_GROUP//2)
DEF N_DISC_PER_GROUP_2  = (N_DISC_PER_GROUP//2)
DEF N_DISC    = (N_GROUP*N_DISC_PER_GROUP)
DEF N_DISC_2  = (N_GROUP_2*N_DISC_PER_GROUP_2)

cdef int UpdateDiscriminants(c_dbl_p X, c_bool_p Y, const intp M, const intp N, const int maxepoch, dbl_p discriminants) nogil:
    # CHANGED: from C to Cython
    # CHANGED: all paramters except maxepoch are now compile-time constants
    # CHANGED: totalerror is no longer returned
    cdef intp *samp_order  = <intp*>malloc(N*sizeof(intp))
    cdef intp grp_order[N_GROUP]
    cdef intp disc_order[N_DISC_PER_GROUP]
    cdef double outputs[N_DISC_2]     # stores shuffled data
    cdef double outputsAND[N_GROUP_2] # stores shuffled data
    cdef dbl_p prevupdates = <dbl_p>malloc(N_DISC*M*sizeof(double))
    cdef double totalerror
    if samp_order is NULL or prevupdates is NULL:
        free(samp_order)
        free(prevupdates)
        return -1

    # e goes over epochs, i goes over N (samples), j goes over M (features)
    # g goes over N_GROUP_2, d goes over N_DISC_PER_GROUP_2
    # ii/gg/dd are the shuffled values of i/g/d
    # (gg is multiplied by N_DISC_PER_GROUP*M and dd is multiplied by M as well)
    cdef intp i, j, g, d, ii, gg, dd
    cdef int e

    # Temporary variables
    cdef double AND_c_muls, prod, sum, out, erro, mp, term2, update
    cdef c_dbl_p x # single sample from X
    cdef dbl_p outs, discs, prevs

    for i in xrange(N):                samp_order[i] = i
    for i in xrange(N_GROUP):          grp_order[i]  = i*M*N_DISC_PER_GROUP
    for i in xrange(N_DISC_PER_GROUP): disc_order[i] = i*M
    memset(prevupdates, 0, N_DISC*M*sizeof(double))

    for e in xrange(maxepoch):
        totalerror = 0
        shuffle(samp_order, N)
        for i in xrange(N):
            shuffle(grp_order, N_GROUP)
            shuffle(disc_order, N_DISC_PER_GROUP)
            ii = samp_order[i]
            x = X+ii*M

            AND_c_muls = 1
            outs = outputs
            for g in xrange(N_GROUP_2):
                gg = grp_order[g]
                prod = 1
                for d in xrange(N_DISC_PER_GROUP_2):
                    sum = 0
                    discs = discriminants+gg+disc_order[d]
                    for j in xrange(M): sum += discs[j]*x[j]
                    out = 1/(1+exp(-sum))
                    outs[0] = out; outs += 1
                    prod *= out
                outputsAND[g] = prod
                AND_c_muls *= 1-prod
            erro = 0.1 + 0.8*Y[ii] - (1 - AND_c_muls)
            totalerror += erro*erro

            outs = outputs
            for g in xrange(N_GROUP_2):
                gg = grp_order[g]
                mp = AND_c_muls/(1-outputsAND[g])*outputsAND[g]*erro
                for d in xrange(N_DISC_PER_GROUP_2):
                    dd = gg+disc_order[d]
                    discs = discriminants+dd
                    prevs = prevupdates+dd
                    term2 = mp * (1 - outs[0]); outs += 1
                    for j in xrange(M):
                        update = x[j]*term2 + MOMENTUM*prevs[j]
                        discs[j] += EPSILON*update
                        prevs[j] = update

        totalerror = sqrt(totalerror/N)
        with gil: __print('       Epoch #%d error=%f' % (e+1,totalerror))

    free(samp_order)
    free(prevupdates)
    return 0
