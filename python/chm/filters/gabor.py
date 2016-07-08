"""
Gabor Filter. Implemented with FFTs.

NOTE: I have the real-space ones in a separate document for reviewing and comparing. They do not
have 'compat' added, but that would be easy to add.

NOTE: also tested doing complex Fourier transforms and it was slower and more code (some parts were
less code, but overall more).

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

try:
    # See if we can use FFTW or give a warning that we can't
    import pyfftw #pylint: disable=unused-import
    _have_pyfftw = True
except ImportError:
    _have_pyfftw = False
    from warnings import warn
    warn('Unable to import pyfftw, using the NumPy FFT functions to calculate the Gabor filters. They are much slower.')

class Gabor(Filter):
    """
    Computes several different Gabor filters on the image producing using all combinations of the
    following parameters to create the kernel:
      sigma:      2, 3, 4, 5, 6
      lambdaFact: 2, 2.25, 2.5   (lambaFact = lamba / sigma)
      orient:     pi/6, pi/3, pi/2, 2pi/3, 5pi/6, pi, 7pi/6, 4pi/3, 3pi/2, 5pi/3, 11pi/6, 2pi
      gamma:      1
    Kernels uses a phase of 0 and pi/2 are computed, and combined using the L2 norm into a single
    feature, resulting in a total of 180 features.

    The original MATLAB code used real-space transforms while this one uses FFTs which makes it >10x
    faster (since the large FFT for the image itself can be calculated just once) and is just as
    accurate.

    There are two versions of this function, one using the NumPy FFT functions and the other using
    FFTW if that library is available. The FFTW version is faster and multi-threaded while the
    NumPy one cannot be used multi-threaded due to a bug in all NumPy version before v1.9.

    This uses O(4*im.size) intermediate memory. However, O(3*im.size) of that is permanent across
    calls for each different image size and number of threads given to this function if FFTW is
    used.

    Compatibility notes:
    The original MATLAB code has a serious bug in it that it uses the imfilter function with a uint8
    input which causes the filter output to be clipped to 0-255 and rounded to an integer. This is a
    major problem as the Gabor filters have negative values (all of which are lost) and can produce
    results above the input range, along with the losing lots of resolution in the data (from ~16
    significant digits to ~3). So, when compat is True (the default), the data is put through the
    same simplification process. When it is False, then much better results are produced.
    """
    def __init__(self, compat=False):
        super(Gabor, self).__init__(Gabor.__padding, len(Gabor.__filters))
        self.__compat = compat

    def __call__(self, im, out=None, region=None, nthreads=1):
        # INTERMEDIATE: im.shape (for either numpy or pyfttw)
        from itertools import izip
        from numpy import empty
        from ._base import get_image_region, round_u8_steps, hypot, copy

        padding = Gabor.__padding
        compat = self.__compat
        im, region = get_image_region(im, padding, region)
        H, W = im.shape[0]-2*padding, im.shape[1]-2*padding # post-filtering shape
        if out is None: out = empty((len(Gabor.__filters), H, W), dtype=im.dtype)
        IF2 = empty((H, W), dtype=im.dtype)
        conv1, conv2 = Gabor.__get_convolve(im, nthreads)
        
        for IF1,(GF1,GF2) in izip(out, Gabor.__filters):
            start = len(GF1) // 2 + padding

            # Calculate the real and imaginary parts of the filter
            copy(IF1, conv1(GF1)[start:H+start, start:W+start], nthreads)
            copy(IF2, conv2(GF2)[start:H+start, start:W+start], nthreads)
            
            # Get the magnitude of the complex value
            if compat:
                round_u8_steps(IF1)
                round_u8_steps(IF2)
            else:
                # TODO: this really should not be necessary
                IF1.clip(0.0, 1.0, IF1)
                IF2.clip(0.0, 1.0, IF2)
            hypot(IF1, IF2, IF1, nthreads)
            
        return out


    ##### 'Convolution' Methods #####
    @staticmethod
    def __get_convolve_numpy(im, nthreads):
        # INTERMEDIATE:
        #     2*(im.shape[0],(im.shape[1]+1)//2) + (?,?)
        #     2*(im.shape[0],(im.shape[1]+1)//2) + (?,?)    [during call to conv]
        #     im.shape + (?,?)                              [during call to conv]
        from numpy.fft import rfft2, irfft2
        from ._base import next_regular
        im_fft_sh = tuple(next_regular(x) for x in im.shape)
        im = rfft2(im, s=im_fft_sh)
        def conv(K):
            F = rfft2(K, s=im_fft_sh)  
            F *= im
            return irfft2(F, s=im_fft_sh)
        return conv, conv

    @staticmethod
    def __get_convolve_pyfftw(im, nthreads):
        # INTERMEDIATES: for fftw: (these persist across calls, ~24 MB for 1000x1000)
        #           im.shape + (?,?)
        #        2*[(im.shape[0],im.shape[1]//2+1) + (?,?/2)] (complex)
        fft_im, fft_gf, ifft = Gabor.__get_fftw_plans(im.shape, nthreads)
        im = Gabor.__fftw(fft_im, im)
        ifft_out = ifft.output_array
        # TODO: don't precompute division?
        N1 = 1 / (ifft_out.shape[0] * ifft_out.shape[0]) # scaling factors
        N2 = 1 / (ifft_out.shape[1] * ifft_out.shape[1])
        def conv1(K):
            from numpy import multiply
            F = Gabor.__fftw(fft_gf, K)
            multiply(F, im, out=F)
            ifft.execute()
            multiply(ifft_out, N1, out=ifft_out)
            return ifft_out
        def conv2(K):
            from numpy import multiply
            F = Gabor.__fftw(fft_gf, K)
            multiply(F, im, out=F)
            ifft.execute()
            multiply(ifft_out, N2, out=ifft_out)
            return ifft_out
        return conv1, conv2

    __get_convolve = (__get_convolve_pyfftw if _have_pyfftw else __get_convolve_numpy)


    ##### FFTW Helpers #####
    @staticmethod
    def __fftw(plan, src):
        """Copy the src into the plan's input array, padded with zeros, then run the plan."""
        from numpy import copyto

        # Copy data to upper-left corner, and zero-out bottom and right edges
        dst = plan.input_array
        dH, dW = dst.shape
        sH, sW = src.shape
        if dH < sH or dW < sW: raise ValueError()
        copyto(dst[:sH, :sW], src)
        dst[sH:,:].fill(0)
        dst[:sH,sW:].fill(0)

        plan.execute()
        return plan.output_array

    @staticmethod
    def __get_fftw_plans(sh, nthreads):
        """
        Create plans for running FFTW with Gabor filters. This caches the plans and buffers for
        same-sized images and number of threads.
        """
        from ._base import next_regular
        sh = tuple(next_regular(x) for x in sh)
        ID = sh + (nthreads,)
        plans = Gabor.__fftw_plans.get(ID)
        if plans is None:
            from numpy import float64, complex128
            from pyfftw import empty_aligned, FFTW
            fsh = sh[:-1] + (sh[-1]//2 + 1,)
            A = empty_aligned(sh,  float64)    # real-space image, kernel, or output
            B = empty_aligned(fsh, complex128) # fourier-space image
            C = empty_aligned(fsh, complex128) # fourier-space kernel
            fft_im = FFTW(A, B, (-2, -1), 'FFTW_FORWARD',  ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'), nthreads)
            fft_gf = FFTW(A, C, (-2, -1), 'FFTW_FORWARD',  ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'), nthreads)
            ifft   = FFTW(C, A, (-2, -1), 'FFTW_BACKWARD', ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'), nthreads)
            Gabor.__fftw_plans[ID] = plans = (fft_im, fft_gf, ifft)
        return plans
    
    __fftw_plans = {}


    ##### Static Fields #####
    __filters = None
    __padding = None

    @staticmethod
    def __get_filters():
        # Not separated even when possible because FFT convolutions don't work that way
        # Result takes ~2MB, calculation takes ~30ms
        from numpy import pi, arange
        from itertools import product
        sigma = [2, 3, 4, 5, 6]
        lambdaFact = [2, 2.25, 2.5]
        orient = arange(pi/6, 2*pi+.1, pi/6) # [pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6, pi, 7*pi/6, 4*pi/3, 3*pi/2, 5*pi/3, 11*pi/6, 2*pi]
        phase = [0, pi/2] # doing 0 and pi/2 gets the real and imaginary parts of a filter at phase 0
        gamma = [1]
        return [[Gabor.__get_filter(s,o,l*s,p,g) for p in phase]
                for s, g, o, l in product(sigma, gamma, orient, lambdaFact)]

    @staticmethod
    def __get_filter(sigma,theta,lmbda,psi,gamma):
        """
        sigma: Gaussian envelope
        theta: orientation
        lmbda: wavelength
        psi: phase
        gamma: aspect ratio
        """
        from numpy import ceil, ogrid, sin, cos, exp, pi, negative

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
        x_theta /= -2*sigma*sigma
        exp(x_theta, out=x_theta)
        x_theta *= ang
        if psi != 0: negative(x_theta, x_theta) # TODO: this is a HACK to make the Fourier transform versions work
        x_theta.flags.writeable = False
        return x_theta

    @staticmethod
    def __static_init__():
        Gabor.__filters = Gabor.__get_filters()
        Gabor.__padding = max(len(GF[0]) // 2 for GF in Gabor.__filters)

Gabor.__static_init__()
