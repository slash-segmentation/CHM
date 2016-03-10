"""
Filters used by the CHM segmentation algorithm.

These are based on the MATLAB functions originally used. The CHANGED comments state what differences
there are from the MATLAB versions - mostly not returning unused return values or hard-coding some
values in. General changes not listed for each filter since they apply to all filters:
  * generate (features,H,W) arrays instead of (features,H*W)
  * take an optional "out" argument to store the results in instead of allocating it themselves,
    all uses of them use this out argument; they do support keeping it None for allocation though
  * take an optional "region" argument that describes the region of the image to process (as a
    sequence of top, left, bottom, right) that allows a filter to use the neighboring pixel data
    instead of padding an image; if None then padding is added if necessary
  * take an option nthreads argument to set the number of threads (defaulting to single threaded),
    although some functions aren't multi-threaded so won't benefit
  * some filters accepted color or non-float64 images, now all inputs must be float64 2D arrays

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from collections import namedtuple

from pysegtools.general.delayed import delayed
from pysegtools.general import cython

cython.install()
import __chm_filters

from chm_utils import get_image_region

########## Utilities ##########
def __separate_filter(f):
    """
    This takes a 2D convolution filter and separates it into 2 1D filters (vertical and horizontal).
    If it can't be separated then the original filter will be returned.
    """
    from numpy import finfo, sqrt
    from numpy.linalg import svd
    u, s, v = svd(f)
    tol = s[0] * max(f.shape) * finfo(s.dtype).eps
    if (s[1:] > tol).any(): return f
    scale = sqrt(s[0])
    return u[:,0]*scale, v[0,:]*scale

def __run_threads(func, total, min_size=1, nthreads=1):
    """
    Runs a bunch of threads, over "total" items, chunking the items. The function is given a
    "threadid" (a value 0 to nthreads-1, inclusive) along with a start and stop index to run over
    (where stop is not included).
    """
    from multiprocessing import cpu_count
    from threading import Thread
    from math import floor
    nthreads = min(nthreads, (total + (min_size // 2)) // min_size, cpu_count())
    if nthreads <= 1:
        func(0, 0, total)
    else:
        inc = total / nthreads
        inds = [0] + [int(floor(inc*i)) for i in xrange(1, nthreads)] + [total]
        threads = [Thread(target=func, args=(i, inds[i], inds[i+1])) for i in xrange(nthreads)]
        for t in threads: t.start()
        for t in threads: t.join()


##### OPTIMIZATIONS #####
# Reduce padding:
#   HoG  (+1 due to the algorithm)
#   Sift (+3? has lots of correlate1d's)
#
# Could further optimize reflective padding by not actually adding padding but instead use the
# intrinsic padding abilities of the various scipy ndimage functions (no way to turn off their
# padding anyways...). This has been done for ConstructNeighborhoods. However the other filters
# won't really benefit since currently Gabor cannot use it and it has by-far the largest padding
# requirement.
#
# These are in general optimized for feature-at-once (all pixels are done for a single feature)
# Some filters however operate pixel-at-once (all features are done for a single pixel)
# The following are pixel-at-once:
#   Haar (although only 2 features so not really a problem)
#   HoG  (36 features, a bit more of a problem)
#   SIFT (at least during the final normalization step and has 128 features!)

########## Cumulative Filter ##########
#@profile
def Filterbank(im, out=None, region=None, nthreads=1):
    # Like other filters this supports out, region, and nthreads
    # It pre-computes the regioned-image and output
    from chm_utils import im2double

    # Pad/region the image
    P = MaxFilterPadding()
    im = im2double(get_image_region(im, P, region)[0])
    region = (P, P, im.shape[0]-P, im.shape[1]-P)

    # Get the output for the filters
    if out is None:
        from numpy import empty
        out = empty((TotalFilterFeatures(), im.shape[0] - 2*P, im.shape[1] - 2*P))

    # Run the filters
    nf_start = 0
    for f,_,nf in __filters:
        f(im, out=out[nf_start:nf_start+nf], region=region, nthreads=nthreads)
        nf_start += nf
    
    return out

def MaxFilterPadding():    return max(f[1] for f in __filters)
def TotalFilterFeatures(): return sum(f[2] for f in __filters)

def __get_filters():
    GP = max(len(GF[0]) // 2 for GF in __gabor_filters) # max Gabor padding
    SP = (__sift.patch_sz // 2) - 1                     # SIFT padding (at least in part)
    return [   # filter, padding needed, features produced
        (ConstructHaar,                8,   2),
        (ConstructHOG,                 7,  36),
        (ConstructEdgeFilt,            6,  49),
        (ConstructNeighborhoodsGabor, GP, 180), # GP = 18
        (ConstructSift,               SP, 128), # SP = 7
        (lambda im,out=None,region=None,nthreads=1:
         ConstructNeighborhoods(im, StencilNeighborhood(10), out=out, region=region, nthreads=nthreads), 10, 81),
    ]
__filters = delayed(__get_filters, list)

########## Harr Filter ##########
# Note: implemented in Cython
def ConstructHaar(im, out=None, region=None, nthreads=1):
    im, region = get_image_region(im, 8, region)
    ii = __chm_filters.cc_cmp_II(im) # INTERMEDIATE: im.shape + (16,16)
    return __chm_filters.cc_Haar_features(ii, 16, out=out, nthreads=nthreads)


########## HoG Filter ##########
# Note: implemented in Cython and C
#@profile
def ConstructHOG(im, out=None, region=None, nthreads=1):
    # OPT: the HoG calculator could be sped up slightly by not zero-padding each region
    im, region = get_image_region(im, 7, region)
    return __chm_filters.ConstructHOG(im, 15, out=out, nthreads=nthreads)


########## Edge Detection Filter ##########
def ConstructEdgeFilt(im, out=None, region=None, nthreads=1):
    # CHANGED: no longer returns optional raw data
    from numpy import sqrt
    from __chm_filters import correlate_xy
    im, region = get_image_region(im, 6, region)
    imx = correlate_xy(im, __edge_filter[0], __edge_filter[1], nthreads=nthreads) # INTERMEDIATE: im.shape + (6,3)
    imx *= imx
    imy = correlate_xy(im, __edge_filter[1], __edge_filter[0], nthreads=nthreads) # INTERMEDIATE: im.shape + (6,3)
    imy *= imy
    imx += imy
    sqrt(imx, out=imx)
    region = (region[0]-3, region[1]-3, region[2]-3, region[3]-3)
    return ConstructNeighborhoods(imx, SquareNeighborhood(3), out=out, region=region, nthreads=nthreads)

def __d2dgauss(n, sigma):
    # CHANGED: no longer accepts 2 ns or sigmas or a rotation
    from numpy import ogrid, sqrt
    n2 = -(n+1)/2+1
    ox, oy = ogrid[n2:n2+n, n2:n2+n]
    h = __gauss(oy, sigma) * __dgauss(ox, sigma)
    h /= sqrt((h*h).sum())
    h.flags.writeable = False
    return h
def __gauss(x,std): from numpy import exp, sqrt, pi; return exp(-x*x/(2*std*std)) / (std*sqrt(2*pi))
def __dgauss(x,std): return x * __gauss(x,std) / (std*std) # first order derivative of gauss function # CHANGED: removed - sign to create correlation kernel instead of convolution kernel
__edge_filter = __separate_filter(__d2dgauss(7, 1.0))


########## Gabor Filters ##########
#@profile
def ConstructNeighborhoodsGabor(im, out=None, region=None, nthreads=1):
    # This now uses FFTs instead of real-space convolutions as this is >10x faster and just as accurate
    # Have the real-space ones in a separate document for reviewing and comparing
    # NOTE: rfft2/irfft2 are not thread safe before NP 1.9, so this cannot be multi-threaded on older versions, even with different images!
    from numpy import empty, sqrt, copyto
    from numpy.fft import rfft2, irfft2
    from scipy.signal.signaltools import _next_regular # finds the next regular/Hamming number (all prime factors are 2, 3, and 5) - needed to make fft functions fast

    im, region = get_image_region(im, 18, region)

    H,W = im.shape
    H -= 36; W -= 36
    if out is None:
        out = empty((len(__gabor_filters), H, W), dtype=im.dtype)
    
    im_fft_sh = tuple(_next_regular(x) for x in im.shape)
    im_fft = rfft2(im, s=im_fft_sh) # INTERMEDIATE: (im.shape[0],(im.shape[1]+1)//2) + (?,?)

    tmp1 = empty(out.shape[1:], dtype=im.dtype) # INTERMEDIATE: im.shape
    tmp2 = empty(out.shape[1:], dtype=im.dtype) # INTERMEDIATE: im.shape
    
    for i,(GF1,GF2) in enumerate(__gabor_filters):
        start = len(GF1) // 2 + 18
        GF1 = rfft2(GF1, s=im_fft_sh); GF1 *= im_fft # INTERMEDIATE: (im.shape[0],(im.shape[1]+1)//2) + (?,?)
        GF2 = rfft2(GF2, s=im_fft_sh); GF2 *= im_fft # INTERMEDIATE: (im.shape[0],(im.shape[1]+1)//2) + (?,?)
        GF1 = irfft2(GF1, s=im_fft_sh) # INTERMEDIATE: im.shape + (?,?)
        GF2 = irfft2(GF2, s=im_fft_sh) # INTERMEDIATE: im.shape + (?,?)
        copyto(tmp1, GF1[slice(start, H + start), slice(start, W + start)])
        copyto(tmp2, GF2[slice(start, H + start), slice(start, W + start)])
        tmp1 *= tmp1
        tmp2 *= tmp2
        tmp1 += tmp2
        sqrt(tmp1, out=out[i])
        
    return out

def ConstructNeighborhoodsGabor_FFTW(im, out=None, region=None, nthreads=1):
    # OPT: it would possibly be better to not multi-thread the FFTs (except the whole-image one) and
    # instead multi-thread the loop itself
    
    # This version using a much faster FFT library: FFTW. The code is otherwise the same as in
    # ConstructNeighborhoodsGabor. This will use multi-threads as well. This is in general twice as
    # fast in single-threaded mode, and gets better with more threads on larger images.
    from itertools import izip
    from numpy import empty, sqrt, copyto

    # INTERMEDIATES: for fftw: (these persist across calls, ~24 MB for 1000x1000)
    #           im.shape + (?,?)
    #        2*[(im.shape[0],im.shape[1]//2+1) + (?,?/2)] (complex)
    # INTERMEDIATES: im.shape
    
    im, region = get_image_region(im, 18, region)
    
    H, W = sh = im.shape[0] - 36, im.shape[1] - 36 # post-filtering shape

    if out is None:
        out = empty((len(__gabor_filters), H, W), dtype=im.dtype)

    IF2 = empty(sh, dtype=im.dtype)

    fft_im, fft_gf, ifft = __get_fftw_plans(im.shape, nthreads)
    ifft_out = ifft.get_output_array()
    N = 1 / (ifft_out.shape[0] * ifft_out.shape[1]) # scaling factor
    
    im = __fftw(fft_im, im)
    
    for IF1,(GF1,GF2) in izip(out, __gabor_filters):
        start = len(GF1) // 2 + 18

        GF1 = __fftw(fft_gf, GF1)
        GF1 *= im
        ifft.execute()
        copyto(IF1, ifft_out[start:H+start, start:W+start])
        IF1 *= IF1

        GF2 = __fftw(fft_gf, GF2)
        GF2 *= im
        ifft.execute()
        copyto(IF2, ifft_out[start:H+start, start:W+start])
        IF2 *= IF2

        IF1 += IF2
        sqrt(IF1, out=IF1)
        IF1 *= N
        
    return out

def __fftw(plan, src):
    """Copy the src into the plan's input array, padded with zeros, then run the plan."""
    from numpy import copyto

    # Copy data to upper-left corner, and zero-out bottom and right edges
    dst = plan.get_input_array()
    dH, dW = dst.shape
    sH, sW = src.shape
    if dH < sH or dW < sW: raise ValueError()
    copyto(dst[:sH, :sW], src)
    dst[sH:,:] = 0
    dst[:,sW:] = 0

    plan.execute()
    return plan.get_output_array()

def __get_fftw_plans(sh, nthreads):
    """
    Create plans for running FFTW with Gabor filters. This caches the plans and buffers for
    same-sized images and number of threads.
    """
    from scipy.signal.signaltools import _next_regular
    sh = tuple(_next_regular(x) for x in sh)
    ID = sh + (nthreads,)
    plans = __get_fftw_plans.plans.get(ID)
    if plans is None:
        from numpy import float64, complex128
        from pyfftw import simd_alignment, n_byte_align_empty, FFTW
        fsh = sh[:-1] + (sh[-1]//2 + 1,)
        A = n_byte_align_empty(sh,  simd_alignment, float64)
        B = n_byte_align_empty(fsh, simd_alignment, complex128)
        C = n_byte_align_empty(fsh, simd_alignment, complex128)
        fft_im = FFTW(A, B, (-2, -1), 'FFTW_FORWARD',  ('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'), nthreads)
        fft_gf = FFTW(A, C, (-2, -1), 'FFTW_FORWARD',  ('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'), nthreads)
        ifft   = FFTW(C, A, (-2, -1), 'FFTW_BACKWARD', ('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'), nthreads)
        __get_fftw_plans.plans[ID] = plans = (fft_im, fft_gf, ifft)
    return plans
__get_fftw_plans.plans = {}

try:
    # See if we can use FFTW or give a warning that we can't
    import pyfftw #pylint: disable=unused-import
    ConstructNeighborhoodsGabor = ConstructNeighborhoodsGabor_FFTW
except ImportError:
    from warnings import warn
    warn('Unable to import pyfftw, using the NumPy FFT functions to calculate the Gabor filters. They are much slower.')

def __gabor_fn(sigma,theta,lmbda,psi,gamma):
    """
    sigma: Gaussian envelope
    theta: orientation
    lmbda: wavelength
    psi: phase
    gamma: aspect ratio
    """
    from numpy import ceil, ogrid, sin, cos, exp, pi

    # Bounding box
    imax = int(ceil(max(1,3*sigma)))
    y,x = ogrid[-imax:imax+1,-imax:imax+1]

    # Rotation
    x_theta =  x*cos(theta)+y*sin(theta)
    y_theta = -x*sin(theta)+y*cos(theta)

    # Calculate exp(-((x_theta^2+y_theta^2*gamma^2)/(2*sigma^2)))*cos(2*pi/lmbda*x_theta+psi)
    ang = (2*pi/lmbda)*x_theta
    ang += psi
    cos(ang, out=ang)
    x_theta *= x_theta
    y_theta *= gamma
    y_theta *= y_theta
    x_theta += y_theta
    x_theta *= (-1.0 / (2*sigma*sigma))
    exp(x_theta, out=x_theta)
    x_theta *= ang
    x_theta.flags.writeable = False
    return x_theta

def __get_gabor_filters():
    # Not separated even when possible because FFT convolutions don't work that way
    from numpy import pi, arange
    from itertools import product
    sigma = [2, 3, 4, 5, 6]
    lambdaFact = [2, 2.25, 2.5]
    orient = arange(pi/6, 2*pi+.1, pi/6) # [pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6, pi, 7*pi/6, 4*pi/3, 3*pi/2, 5*pi/3, 11*pi/6, 2*pi]
    phase = [0, pi/2]
    gamma = [1]
    return [[__gabor_fn(s,o,l*s,p,g) for p in phase] for s, g, o, l in product(sigma, gamma, orient, lambdaFact)]
__gabor_filters = delayed(__get_gabor_filters, list) # ~2MB, ~30ms


########## SIFT Filter ##########
#@profile
# Note: partially implemented in Cython
def ConstructSift(im, out=None, region=None, nthreads=1):
    from scipy.ndimage.filters import correlate1d
    half_patch_sz = (__sift.patch_sz // 2) - 1
    im, region = get_image_region(im, half_patch_sz, region, 'constant') # OPT: +3? for multiple correlate1d's
    gauss_filter = __sift.gauss_filter
    im = correlate1d(im, gauss_filter, 0, mode='nearest') # INTERMEDIATE: im.shape
    im = correlate1d(im, gauss_filter, 1, mode='nearest')
    return __dense_sift(im, out=out, nthreads=nthreads)

def ConstructSift_OPT(im, out=None, region=None, nthreads=1):
    gauss_filter, padding = __sift.gauss_filter, __sift.padding_sz
    im, region = get_image_region(im, padding, region, 'constant')
    im = __chm_filters.correlate_xy(im, gauss_filter, gauss_filter, nthreads=nthreads) # INTERMEDIATE: im.shape
    return __dense_sift_OPT(im, out=out, nthreads=nthreads)

def __gen_gauss(sigma):
    # CHANGED: no longer deals with anisotropic Gaussians or 2 sigmas
    from numpy import ceil
    f_wid = 4 * ceil(sigma) + 1
    return __fspecial_gauss(f_wid, sigma)

def __gen_dgauss(sigma):
    # CHANGED: same changes as to __gen_gauss
    # CHANGED: only returns on direction of gradient, not both
    # laplacian of size sigma
    from numpy import gradient
    G, _ = gradient(__gen_gauss(sigma))
    return G * 2 / abs(G).sum()

def __fspecial_gauss(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',shape,sigma)
    """
    # CHANGED: unlike MATLAB's fspecial('gaussian', ...) this does not accept a tuple for shape/sigma
    from numpy import ogrid, exp, finfo
    n = (shape-1)/2.
    y,x = ogrid[-n:n+1,-n:n+1]
    h = exp(-(x*x+y*y)/(2.*sigma*sigma))
    h[h < finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0: h /= sumh
    return h

def __get_sift_static():
    # This function generates all of the static variables used by SIFT when first used.
    from numpy import arange

    gauss_filter = __separate_filter(__fspecial_gauss(7, 1.0))[0]
    gauss_filter.flags.writeable = False
    
    dgauss_filter = __separate_filter(__gen_dgauss(1.0))
    dgauss_filter[0].flags.writeable = False
    dgauss_filter[1].flags.writeable = False

    patch_sz = 16 # must be even (actually should be a multiple of num_bins I think)
    padding_sz = (patch_sz // 2) - 1 + 3 + 2
    # if this is changed, it is likely other things have to change as well
    num_angles, num_bins = 8, 4
    #alpha = 9 # parameter for attenuation of angles (must be odd) [hard coded]

    # Convolution formulation
    # Note: the original has an apparent bug where the weight_x vector is not centered (has 2 extra zeros on right side)
    bins_patch = num_bins/patch_sz
    ## Original definition (lots of leading and trailing zeros, 2 extra trailing 0s)
    #weight_x = 1 - abs(arange(1, patch_sz+1) - (half_patch_sz//2) + 0.5)*bins_patch
    #weight_x *= (weight_x > 0)
    ## No leading zeros, 2 trailing zeros, compatible with original definition and slighlty faster and needing less padding
    #weight_x = 1 - abs(arange(1, num_bins-1, bins_patch) - 0.5*(num_bins-bins_patch))
    #weight_x = concatenate((weight_x, [0,0]))
    ## No trailing/leading zeros (if this is used origin=weight_x_origin is needed to make it compatible)
    weight_x = 1 - abs(arange(1, num_bins-1, bins_patch) - 0.5*(num_bins-bins_patch))
    weight_x.flags.writeable = False
    weight_x_origin = -2 # correlate origin, for convolution it needs to be 1

    return __sift_static(gauss_filter, dgauss_filter, patch_sz, padding_sz, num_bins, num_angles, weight_x, weight_x_origin)
__sift_static = namedtuple('sift_static', ('gauss_filter', 'dgauss_filter', 'patch_sz', 'padding_sz', 'num_bins', 'num_angles', 'weight_x', 'weight_x_origin'))
__sift = delayed(__get_sift_static, __sift_static)

def __dense_sift(im, out, nthreads):
    # CHANGED: forced grid_spacing = 1 and patch_size is now a constant defined globally
    # CHANGED: the input im can be modified slightly in place (scaling)
    # CHANGED: no longer returns optional grid
    # CHANGED: a few pieces have been removed and made into Cython functions for speed and multi-threading
    from numpy import empty, empty_like, sqrt, arctan2, cos, sin
    from scipy.ndimage.filters import correlate1d

    im *= 1/im.max() # can modify im here since it is always the padded image

    H, W = im.shape

    dgauss_filter = __sift.dgauss_filter
    tmp = empty_like(im) # INTERMEDIATES: im.shape

    # vertical edges
    correlate1d(im, dgauss_filter[1], 0, mode='constant', output=tmp)
    imx = correlate1d(tmp, dgauss_filter[0], 1, mode='constant') # INTERMEDIATE: im.shape
    
    # horizontal edges
    correlate1d(im, dgauss_filter[0], 0, mode='constant', output=tmp)
    imy = correlate1d(tmp, dgauss_filter[1], 1, mode='constant') # INTERMEDIATE: im.shape

    im_theta = arctan2(imy, imx, out=tmp)
    im_cos, im_sin = cos(im_theta), sin(im_theta, out=tmp) # INTERMEDIATE: im.shape (cos)
    del im_theta, tmp
    
    imx *= imx; imy *= imy; imx += imy
    im_mag = sqrt(imx, out=imx) # gradient magnitude
    del imx, imy # cleanup 1 intermediate (imy)
    
    num_angles, num_bins, patch_sz = __sift.num_angles, __sift.num_bins, __sift.patch_sz
    im_orientation = __chm_filters.sift_orientations(im_mag, im_cos, im_sin, num_angles, nthreads=nthreads)
    del im_mag, im_cos, im_sin # cleanup 3 intermediates

    def thread(_, start, stop):
        weight_x, weight_x_origin = __sift.weight_x, __sift.weight_x_origin
        tmp = empty_like(im) # INTERMEDIATE: im.shape * nthreads
        for a in xrange(start, stop):
            correlate1d(im_orientation[a,:,:], weight_x, 0, mode='constant', origin=weight_x_origin, output=tmp)
            correlate1d(tmp, weight_x, 1, mode='constant', origin=weight_x_origin, output=im_orientation[a,:,:])
    # OPT: these take about 12% of the time of SIFT (when single-threaded)
    __run_threads(thread, num_angles)
    
    H, W = H-patch_sz+2, W-patch_sz+2
    if out is None:
        out = empty((num_bins*num_bins*num_angles, H, W), dtype=im.dtype)

    # OPT: takes about 19% of time in SIFT (when single-threaded)
    sift_arr = out.reshape((num_bins*num_bins, num_angles, H, W))
    __chm_filters.sift_neighborhoods(sift_arr, im_orientation, num_bins, patch_sz//num_bins, nthreads=nthreads)
    #del im_orientation # cleanup large intermediate
    im_orientation = None # cannot delete since it is used in a nested scope, this should work though
    
    # normalize SIFT descriptors
    # OPT: takes about 55% of time in SIFT (when single-threaded)
    __chm_filters.sift_normalize(out.reshape((num_bins*num_bins*num_angles, H*W)), nthreads=nthreads)
    return out

def __dense_sift_OPT(im, out, nthreads):
    # CHANGED: forced grid_spacing = 1 and patch_size is now a constant defined globally
    # CHANGED: the input im can be modified slightly in place (scaling)
    # CHANGED: no longer returns optional grid
    # CHANGED: a few pieces have been removed and made into Cython functions for speed and multi-threading
    from numpy import empty, sqrt, arctan2, cos, sin
    from __chm_filters import correlate_xy

    im *= 1/im.max() # can modify im here since it is always the padded image

    # vertical and horizontal edges
    dgauss_filter = __sift.dgauss_filter
    imx = correlate_xy(im, dgauss_filter[0], dgauss_filter[1]) # INTERMEDIATE: im.shape + (?,?)
    imy = correlate_xy(im, dgauss_filter[1], dgauss_filter[0]) # INTERMEDIATE: im.shape + (?,?)
    del im # cleanup 1 intermediate

    im_theta = arctan2(imy, imx) # INTERMEDIATE: im.shape
    im_cos, im_sin = cos(im_theta), sin(im_theta, out=im_theta) # INTERMEDIATE: im.shape (cos)
    del im_theta

    # gradient magnitude
    imx *= imx; imy *= imy; imx += imy
    del imy # cleanup 1 intermediate
    im_mag = sqrt(imx) # INTERMEDIATE: im.shape  (we don't want to use imx because it isn't C-contiguous)
    del imx # cleanup 1 intermediate
    
    H, W = im_mag.shape
    dt = im_mag.dtype
    num_angles, num_bins, patch_sz = __sift.num_angles, __sift.num_bins, __sift.patch_sz
    im_orientation = __chm_filters.sift_orientations(im_mag, im_cos, im_sin, num_angles, nthreads=nthreads) # INTERMEDIATE: im.shape * 8
    del im_mag, im_cos, im_sin # cleanup 3 intermediates

    #from scipy.ndimage.filters import correlate1d
    #def thread(i, start, stop):
    #    weight_x, weight_x_origin = __sift.weight_x, __sift.weight_x_origin
    #    tmp = empty((H,W), dtype=dt) # INTERMEDIATE: im.shape * nthreads
    #    for a in xrange(start, stop):
    #        correlate1d(im_orientation[a,:,:], weight_x, 0, mode='constant', origin=weight_x_origin, output=tmp)
    #        correlate1d(tmp, weight_x, 1, mode='constant', origin=weight_x_origin, output=im_orientation[a,:,:])
    #        # TODO: origin? __chm_filters.correlate_xy(im_orientation[a,:,:], weight_x, weight_x, output=)
    ## OPT: these take about 12% of the time of SIFT (when single-threaded)
    #__run_threads(thread, num_angles)
    weight_x, weight_x_origin = __sift.weight_x, __sift.weight_x_origin
    for a in xrange(num_angles):
        im_orientation[a,:-7,:-7] = __chm_filters.correlate_xy(im_orientation[a,:,:], weight_x, weight_x)
        im_orientation[a,-7:,:] = 0
        im_orientation[a,:,-7:] = 0
    
    H, W = H-patch_sz+2, W-patch_sz+2
    if out is None:
        out = empty((num_bins*num_bins*num_angles, H, W), dtype=dt)

    # OPT: takes about 19% of time in SIFT (when single-threaded)
    sift_arr = out.reshape((num_bins*num_bins, num_angles, H, W))
    __chm_filters.sift_neighborhoods(sift_arr, im_orientation, num_bins, patch_sz//num_bins, nthreads=nthreads)
    #del im_orientation # cleanup large intermediate
    im_orientation = None # cannot delete since it is used in a nested scope, this should work though
    
    # normalize SIFT descriptors
    # OPT: takes about 55% of time in SIFT (when single-threaded)
    __chm_filters.sift_normalize(out.reshape((num_bins*num_bins*num_angles, H*W)), nthreads=nthreads)
    return out


########## Neighborhood Construction ##########
# Note: implemented in Cython
def ConstructNeighborhoods(im, offsets, out=None, region=None, nthreads=1):
    # CHANGE: permanently now like ConstructNeighborhoods(padReflect(im, *), ____Neighborhood(*), 0)
    # CHANGE: region is handled in such a way that the image is never padded, but instead reflection is handled directly
    # CHANGE: converted to Cython so that it could be multi-threaded and faster
    from numpy import intp
    if offsets.dtype != intp and offsets.dtype.kind == 'i' and offsets.dtype.itemsize == intp(0).itemsize:
        offsets = offsets.view(intp)
    return __chm_filters.ConstructNeighborhoods(im, offsets, out=out, region=region, nthreads=nthreads)

    # Final Python method - just places all the various offsets via copies - 11.6x faster than original method
    #radius = abs(offsets).max()
    #padsize = 2*radius
    #im, region = get_image_region(im, radius, region)
    #H, W, dims = im.shape[0] - padsize, im.shape[1] - padsize, offsets.shape[1]
    #offsets = offsets + radius
    #if out is None:
    #    from numpy import empty
    #    out = empty((dims, H, W), dtype=im.dtype)
    #for i in xrange(dims):
    #    x,y = offsets[:,i]
    #    out[i,:,:] = im[y:y+H,x:x+W]
    #return out
    
    # Original method, was kinda slow (81.2ms for 81x25760 output)
    #from numpy import arange, tile, repeat
    #npix = W * H
    #pos_x = tile(offsets[0,:,None], (1,npix)); pos_x += tile(repeat(arange(W), H), (dims,1)); pos_x += radius
    #pos_y = tile(offsets[1,:,None], (1,npix)); pos_y += tile(       arange(H),     (dims,W)); pos_y += radius
    #return im[pos_y, pos_x]

    # As strided method (although in reverse order), worked fine (25.5ms for 160x161x81 output)
    #from numpy.lib.stride_tricks import as_strided
    #from numpy import arange
    #pos_x = offset[0,None,None,:] + radius + arange(W)[:,None,None]
    #pos_x = as_strided(pos_x, (H, W, dims), (0, pos_x.strides[1], pos_x.itemsize))
    #pos_y = offset[1,None,None,:] + radius + arange(H)[:,None,None]
    #pos_y = as_strided(pos_y, (H, W, dims), (pos_y.strides[0], 0, pos_y.itemsize))
    #return im[pos_y, pos_x]

def __memoize(f):
    """
    This decorator causes neighborhoods to be cached. Since each nieghbor type is really only used
    at 1 or 2 different radii, this means they only need to be calculated once.
    """
    #pylint: disable=protected-access
    f.__memos = memos = {}
    def memoized(radius):
        x = memos.get(radius)
        if x is None:
            x = f(radius)
            x.flags.writeable = False
            memos[radius] = x
        return x
    return memoized

# Scipy ndimage functions take footprints (boolean masks) instead of neighborhoods (list of indices)
# like the MATLAB functions, so we provide both here. Basically the neighborhoods are the footprints
# with a call to indices or where and subtracting radius.

@__memoize
def SquareNeighborhood(radius):
    from numpy import indices, intp
    diam = 2*radius + 1
    return (indices((diam, diam)) - radius).reshape((2, -1)).astype(intp, copy=False).view(intp) # last two are to make sure it is intp dtype - numpy bug needs both to overcome!

@__memoize
def SquareFootprint(radius):
    from numpy import ones
    diam = 2*radius + 1
    return ones((diam, diam), dtype=bool)

@__memoize
def StencilNeighborhood(radius):
    from numpy import ogrid, array, where, intp
    r,c = ogrid[-radius:radius+1,-radius:radius+1]
    return array(where((abs(r)==abs(c)) + (r*c==0))[::-1], dtype=intp) - radius

@__memoize
def StencilFootprint(radius):
    from numpy import ogrid
    r,c = ogrid[-radius:radius+1,-radius:radius+1]
    return ((abs(r)==abs(c)) + (r*c==0))


############ Pad Image ##########
##def padReflect(im, r):
##    # CHANGED: no longer zero-pads images that are smaller than r (or returns optional zero-padded image)
##    # NOTE: most uses of this were passing to ConstructNeighboorhood and it has been integrated into
##    # that function. All other uses have been changed to just call numpy's pad function.
##    from numpy import pad
##    return pad(im, r, mode=str('symmetric'))
##    #"""
##    #function [impad] = padReflect(im,r)
##    #
##    #Pad an image with a border of size r, and reflect the image into the border.
##    #
##    #David R. Martin <dmartin@eecs.berkeley.edu>
##    #March 2003
##    #"""
##    #from numpy import empty, vstack, hstack, flipud, fliplr
##    #if r > im.shape[0]: im = vstack((im, zeros(r - im.shape[0], im.shape[1])))
##    #if r > im.shape[1]: im = hstack((im, zeros(im.shape[0], r - im.shape[1])))
##    #impad = empty((im.shape[0]+2*r, im.shape[1]+2*r))
##    #impad[r:-r,r:-r] = im # middle
##    #impad[:r,r:-r] = flipud(im[:r,:]) # top
##    #impad[-r:,r:-r] = flipud(im[-r:,:]) # bottom
##    #impad[r:-r,:r] = fliplr(im[:,:r]) # left
##    #impad[r:-r,-r:] = fliplr(im[:,-r:]) # right
##    #impad[:r,:r] = flipud(fliplr(im[:r,:r])) # top-left
##    #impad[:r,-r:] = flipud(fliplr(im[:r,-r:])) # top-right
##    #impad[-r:,:r] = flipud(fliplr(im[-r:,:r])) # bottom-left
##    #impad[-r:,-r:] = flipud(fliplr(im[-r:,-r:])) # bottom-right
##    #return impad
