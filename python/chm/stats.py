"""
Wrappers for optimized stats functions that don't copying data excessively and are multi-threaded
when possible. Includes:
    
    cy_range(_0/_1) - calculates min and max of a 1D or 2D array along a particular axis
    cy_mean_stdev(_0/_1) - calculates mean and std deviation of a 1D or 2D array along an axis
    cy_percentile(_0/_1) - gets the values of percentiles of a 1D or 2D array along an axis

Typically these have the greatest speed benefit over the similar NumPy functions when the data is
very large and particularily if it is in a memory-mapped file. Even when using a single thread they
prove to by a few times faster than the NumPy versions for large data sets. For example, the
cy_range calculates min and max faster than NumPy calculates just min.

In the stats module there are wrappers for these functions that choose which version to call or
fall back to the NumPy varieties if needed.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

def __get_stats(X, axis, nthreads, opts, cy, cy0, cy1, np):
    """
    Generalized function from calling the __stats.cy_... functions and falling back to some numpy function.
    X is the data, axis is the axis to calculate over (or None for flattened data), nthreads in the number
    of threads to try to use with the cy_ functions, opts is a squence of additional options that need
    to be given to the Cython functions. The cy, cy0, and cy1 functions are the 1D, 2D axis-0, and 2D
    axis-1 Cython functions to use (they take X, *opts, and nthreads as a keyword argument). np is the
    numpy-fallback function (takes X and axis).
    """
    from numpy import asarray, rollaxis
    
    # No axis - flattened data
    if axis is None: return cy(X.ravel('K'), *opts, nthreads=nthreads)
    
    # Adjust axis and shape of data
    # By the end of this X is 2D and axis is 0 or 1
    if axis < 0: axis += X.ndim
    if axis < 0 or axis >= X.ndim: raise ValueError('axis')
    if X.ndim != 2:
        raise ValueError('2D arrays only supported with axis argument')
        # TODO: this is possibly not correct, but very confusing to get right
        # additionally, the results would have to be reshaped to match the numpy functions
        #if axis == 0: X = X.reshape((-1, X.shape[-1]))
        #else:
        #    if axis != X.ndim-1: X = rollaxis(X, axis, X.ndim)
        #    X = X.reshape((X.shape[0], -1))
        #    axis = 1
    
    if X.flags.forc:
        # Get axis stats the fast way
        if X.flags.f_contiguous: X = X.T; axis = 1-axis
        return (cy1 if axis == 1 else cy0)(X, *opts, nthreads=nthreads)
    
    # Get axis stats the slow way
    return asarray(np(X, axis))

def min_max(X, axis=None, nthreads=1):
    """Get the min and max of X, either from the flattened data (default) or along a given axis."""
    from .__stats import cy_min_max, cy_min_max_0, cy_min_max_1
    return __get_stats(X, axis, nthreads, (),
                       cy_min_max, cy_min_max_0, cy_min_max_1,
                       lambda X,axis:(X.min(axis), X.max(axis)))

def mean_stdev(X, axis=None, ddof=0.0, nthreads=1):
    """Calculate the mean and standard deviation of X, either from the flattened data (default) or along a given axis."""
    from .__stats import cy_mean_stdev, cy_mean_stdev_0, cy_mean_stdev_1
    return __get_stats(X, axis, nthreads, (ddof,),
                       cy_mean_stdev, cy_mean_stdev_0, cy_mean_stdev_1,
                       lambda X,axis:(X.mean(axis), X.std(axis, ddof=ddof)))

def median(X, axis=None, overwrite=False, nthreads=1):
    """Calculate the median of X, either from the flattened data (default) or along a given axis. If overwrite is True (no the default) then the data in X may be changed."""
    from numpy import median
    from .__stats import cy_percentile, cy_percentile_0, cy_percentile_1
    return __get_stats(X, axis, nthreads, (0.5, overwrite),
                       cy_percentile, cy_percentile_0, cy_percentile_1,
                       lambda X,axis:median(X, axis, overwrite_input=overwrite))

def percentile(X, qs, axis=None, overwrite=False, nthreads=1):
    """Calculate the values at a single or multiple percentile value in X. Uses either the flattened data (default) or along a given axis. The percentiles should be given as floating point values from 0.0 to 1.0. A scalar or a sequence of them can be given. If overwrite is True (no the default) then the data in X may be changed."""
    from numpy import asarray, percentile
    from .__stats import cy_percentile, cy_percentile_0, cy_percentile_1
    return __get_stats(X, axis, nthreads, (qs,),
                       cy_percentile, cy_percentile_0, cy_percentile_1,
                       lambda X,axis:percentile(X, asarray(qs)*100, axis, overwrite_input=overwrite))

def median_mad(X, axis=None, nthreads=1):
    """
    Calculate the median and MAD values of X either from the flattened data (default) or along a given axis.
    The MAD values are standardized to match standard deviation (thus divided by ~0.6745).
    """
    from numpy import abs
    med = median(X, axis, nthreads=nthreads)
    mad = X - (med if axis is None else (med[None, :] if axis == 0 else med[:, None]))
    mad = median(abs(mad, mad), axis, overwrite=True, nthreads=nthreads)
    mad *= 1.482602218505602 # 1/scipy.stats.norm.ppf(3/4)
    return med, mad