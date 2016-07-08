#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
Frangi filter using the eigenvectors of the Hessian to compute the likeliness of an image region
to contain vessels or other image ridges, according to the method described by Frangi (1998).
Adapted from the MATLAB version by M. Schrijver (2001) and D. Kroon (2009).

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters.pxi"

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport floor, fabs, sqrt, exp, M_PI
from cython.parallel cimport parallel
from openmp cimport omp_get_num_threads, omp_get_thread_num

ctypedef npy_double dbl
ctypedef dbl* dbl_p

DEF BETA=0.5

def frangi(ndarray im, dbl sigma, ndarray out=None, int nthreads=1):
    """
    Computes the 2D Frangi filter using the eigenvectors of the Hessian to compute the likeliness
    of an image region to contain vessels or other image ridges, according to the method described
    by Frangi (1998).
    
    Inputs:
        im          the input image, must be a 2D grayscale image from 0.0 to 1.0
                    the edges of the image (int(3*sigma+0.5) pixels) are dropped so the image
                    should be pre-padded
        sigma       the sigma used
        
    `beta` is fixed to 0.5 and `c` is dynamically calculated as half of the maximum Frobenius norm
    of all Hessian matrices. Assumes ridges are black. To do the opposite, invert the image.
    
    Outputs:
        out         the filtered image
                    if provided must be a double, 2D, C-contiguous, behaved, proper-sized array
    
    Tempoarary memory usage is approximately 3 times the size of the image.

    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by M. Schrijver (2001) and D. Kroon (2009)
    
    Differences for the MATLAB version:
      * the image must be pre-padded
      * only a single sigma is calculated
      * FrangiBetaOne (renamed to `beta`) is fixed to 0.5
      * FrangiBetaTwo (renamed to `c`) is always dynamically calculated now
      * BlackWhite and verbose arguments have been dropped
      * does not output sigmas and directions
    
    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by M. Schrijver (2001) and D. Kroon (2009)
    """
    # Check arguments
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2 or PyArray_STRIDE(im, 1) != sizeof(dbl): raise ValueError("Invalid im")
    if sigma <= 0.0: raise ValueError('Invalid sigma')
    cdef intp padding = <intp>(3.0*sigma+0.5)
    cdef intp H = PyArray_DIM(im, 0) - 2*padding, W = PyArray_DIM(im, 1) - 2*padding, N = H*W
    cdef intp[2] dims
    if out is None:
        dims[0] = H; dims[1] = W
        out = PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
    elif not PyArray_ISCARRAY(out) or PyArray_TYPE(out) != NPY_DOUBLE or \
         PyArray_NDIM(out) != 2 or PyArray_DIM(out, 0) != H or PyArray_DIM(out, 1) != W:
        raise ValueError('Invalid output array')
    nthreads = get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows

    # Calculate the scaled 2D hessian
    cdef ndarray u, uu, v
    u, uu, v = get_hessian_filters(sigma)
    from ._correlate import correlate_xy
    cdef ndarray Dxx = correlate_xy(im, v,  u,  None, nthreads)
    cdef ndarray Dxy = correlate_xy(im, uu, uu, None, nthreads)
    cdef ndarray Dyy = correlate_xy(im, u,  v,  None, nthreads)
    
    # Get ready to calculate the vesselness
    cdef intp a, b, i, _nthreads
    cdef double inc
    cdef dbl_p pDxx = <dbl_p>PyArray_DATA(Dxx), pDxy = <dbl_p>PyArray_DATA(Dxy), pDyy = <dbl_p>PyArray_DATA(Dyy)
    cdef dbl_p lam1 = pDxx, lam2 = pDxy # re-use pDxx and pDxy for lambda1 and lambda2
    cdef dbl_p pOut = <dbl_p>PyArray_DATA(out), C
    cdef dbl c = 0.0
    
    # Main computations
    with nogil:
        if nthreads == 1:
            c = -2.0/eigvals(N, pDxx, pDxy, pDyy)
            vesselness(N, lam1, lam2, pOut, c)
        else:
            # Calculate (abs sorted) eigenvalues and the dynamic c value
            C = <dbl_p>malloc(nthreads*sizeof(dbl))
            if C is NULL:
                with gil: raise MemoryError()
            memset(C, 0, nthreads*sizeof(dbl))
            with parallel(num_threads=nthreads):
                i = omp_get_thread_num()
                _nthreads = omp_get_num_threads() # in case there is a difference...
                inc = N / <double>_nthreads # a floating point number, use the floor of adding it together
                a = <intp>floor(inc*i)
                b = N if i == _nthreads - 1 else (<intp>floor(inc*(i+1)))
                C[i] = eigvals(b-a, <dbl_p>(<char*>pDxx+a), <dbl_p>(<char*>pDxy+a), <dbl_p>(<char*>pDyy+a))
            for i in xrange(nthreads):
                if C[i] > c: c = C[i]
            free(C)
            c = -2.0/c
        
            # Calculate the vesselness
            with parallel(num_threads=nthreads):
                i = omp_get_thread_num()
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = N / <double>nthreads # a floating point number, use the floor of adding it together
                a = <intp>floor(inc*i)
                b = N if i == nthreads - 1 else (<intp>floor(inc*(i+1)))
                vesselness(b-a, <dbl_p>(<char*>lam1+a), <dbl_p>(<char*>lam2+a), <dbl_p>(<char*>pOut+a), c)

    # Return output
    return out

cdef dbl SQRT_2_PI = sqrt(2.0*M_PI)
cdef dict filters = {} # sigma -> tuple of 3 vectors
cdef get_hessian_filters(dbl sigma):
    """Gets the filters for the Hessian 2D for the given sigma and caches them."""
    cdef intp r, w, i
    cdef dbl sigma2, factor, x
    fs = filters.get(sigma)
    if fs is None:
        r = <intp>(3.0*sigma+0.5); w = 2*r+1
        sigma2 = sigma*sigma
        factor = 1.0/(SQRT_2_PI*sigma2)
        v  = PyArray_EMPTY(1, &w, NPY_DOUBLE, False)
        uu = PyArray_EMPTY(1, &w, NPY_DOUBLE, False)
        u  = PyArray_EMPTY(1, &w, NPY_DOUBLE, False)
        for i in xrange(w):
            x = r - i # x goes from r to -r
            v[i]  = factor*exp(-x*x/(2.0*sigma2))
            uu[i] = v[i]*x
            u[i]  = v[i]*(x*x-sigma2)
        filters[sigma] = fs = u, uu, v
    return fs # u, uu, v

cdef dbl eigvals(intp N, dbl_p Dxx, dbl_p Dxy, dbl_p Dyy) nogil:
    """
    Calculate the eigenvalues from the Hessian matrix of an image, sorted by absolute value.

    `eigvals(Dxx,Dxy,Dyy,H,W,stride)`
    
    Inputs:
        N               the number of values in the arrays
        Dxx,Dxy,Dyy     the outputs from hessian2
        
    Outputs:
        Dxx,Dxy         the eigenvalues, sorted by absolute value
        
    Returns:
        Frobenius maximum norm of all Hessian matrices
    """
    cdef intp i
    cdef dbl summ, diff, temp, mu1, mu2, lam1, lam2, S2, c = 0.0
    for i in xrange(N):
        summ = Dxx[i] + Dyy[i]
        diff = Dxx[i] - Dyy[i]
        temp = sqrt(diff*diff + 4*Dxy[i]*Dxy[i])

        # Compute the eigenvalues
        mu1 = 0.5*(summ + temp) # mu1 = (Dxx + Dyy + sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
        mu2 = 0.5*(summ - temp) # mu2 = (Dxx + Dyy - sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2

        # Sort eigenvalues by absolute value abs(eig1) < abs(eig2)
        if fabs(mu1) > fabs(mu2): lam1 = mu2*mu2; lam2 = mu1 # lambda1 is always used squared
        else:                     lam1 = mu1*mu1; lam2 = mu2
        Dxx[i] = lam1; Dxy[i] = lam2

        # Calculate the dynamic c value
        if lam2 > 0.0:
            S2 = lam1 + lam2*lam2
            if S2 > c: c = S2

    # Return the Frobenius maximum norm of all Hessian matrices
    return c

cdef void vesselness(intp N, dbl_p lambda1, dbl_p lambda2, dbl_p out, dbl c) nogil:
    """
    Calculates the vesselness based on the absolute-value sorted eigenvalues.

    `vesselness(N,lambda1,lambda2,out,c)`
    
    Inputs:
        N               the number of values in the arrays
        lambda1,lambda2 the outputs from eigval2image
        c               the constant value `c`
        
    Outputs:
        out             the vesselness values
    """
    cdef intp i
    cdef dbl lam1, lam2, Rb2, S2
    for i in xrange(N):
        lam2 = lambda2[i]
        if lam2 > 0.0:
            # Compute similarity measures
            lam1 = lambda1[i]; lam2 = lam2*lam2
            Rb2 =     exp(lam1/lam2*(-1/(2*BETA*BETA))) # exp(-Rb^2/(2*beta^2)); Rb = lambda1 / lambda2
            S2  = 1.0-exp((lam1+lam2)*c) # 1-exp(-S^2/(2*c^2));   S = sqrt(sum(lambda_i^2))
            # Compute vessel-ness
            out[i] = Rb2 * S2
        else: out[i] = 0

