"""
The core imresize functions in pure Python. These are only used when the Cython versions could not
be accessed. They are much slower than the Cython versions.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from numpy import array, ndarray
from pysegtools.general.delayed import delayed

class __min_max_class(dict):
    def __missing__(self, t):
        from numpy import iinfo
        ii = iinfo(t)
        self[t] = ret = (ii.min, ii.max)
        return ret
__get_min_max = __min_max_class().__getitem__

def imresize(im, weights, indices):
    from numpy import empty, float64
    from itertools import izip
    out = empty((weights.shape[0], im.shape[1]), dtype=im.dtype)
    if im.dtype.type == float64:
        for w,i,o in izip(weights, indices, out):
            w.dot(im[i,:], out=o)
    elif im.dtype.kind == 'f':
        tmp = empty(im.shape[1])
        for w,i,o in izip(weights, indices, o):
            o[:] = w.dot(im[i,:], out=tmp)
    else:
        mn, mx = __get_min_max(im.dtype.type)
        tmp = empty(im.shape[1])
        for w,i,o in izip(weights, indices, out):
            w.dot(im[i,:], out=tmp).clip(mn, mx, out=tmp).round(out=o)
    return out

__bicubic_weights = delayed(lambda:array([-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875]), ndarray)
__fast_inds = delayed(lambda:array([[0, 0, 0, 0, 1, 2, 3, 4],
                                    [0, 0, 1, 2, 3, 4, 5, 6],
                                    [-5, -4, -3, -2, -1, -1, -1, -1],
                                    [-7, -6, -5, -4, -3, -2, -1, -1]]), ndarray)

def imresize_fast(im, weights=__bicubic_weights, indices=__fast_inds):
    from numpy import empty, float64
    from itertools import izip
    Rin = im.shape[0]
    Rout = (Rin+1) // 2
    out = empty((Rout, im.shape[1]), dtype=im.dtype)
    indices = indices[:Rout].copy()
    indices[2:,:] += (Rin + (Rin % 2))
    indices.clip(0, Rin - 1, out=indices)
    special_outs = [out[0, :], out[1, :], out[-1, :], out[-2, :]]
    if im.dtype.type == float64:
        for i,o in izip(indices, special_outs):
            weights.dot(im[i,:], out=o)
        for i,o in enumerate(out[2:-2]):
            weights.dot(im[1+2*i:9+2*i,:], out=o)
    elif im.dtype.kind == 'f':
        tmp = empty(im.shape[1:])
        for i,o in izip(indices, special_outs):
            o[:] = weights.dot(im[i,:], out=tmp)
        for i,o in enumerate(out[2:-2]):
            o[:] = weights.dot(im[1+2*i:9+2*i,:], out=tmp)
    else:
        mn, mx = __get_min_max(im.dtype.type)
        tmp = empty(im.shape[1:])
        for i,o in izip(indices, special_outs):
            weights.dot(im[i,:], out=tmp).clip(mn, mx, out=tmp).round(out=o)
        for i,o in enumerate(out[2:-2]):
            weights.dot(im[1+2*i:9+2*i,:], out=tmp).clip(mn, mx, out=tmp).round(out=o)
    return out
