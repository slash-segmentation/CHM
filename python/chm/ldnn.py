"""
Logistic Disjunctive Normal Network (LDNN) Model implementation.

This implements the algorithm as described in Seyedhosseini, Sajjadi, and Tasdizen (2013).

This was originally in the MATLAB files LearnAndOrNetMEX, UpdateDiscriminants(_SB),
EvaluateAndOrNetMEX, genOutput, genOutput(_SB). Most variables and functions have been renamed to
match the paper, e.g. discriminants changed to w or weights. Some parts were removed, such as
validation and regularization from LearnAndOrNetMEX (most of it had already been removed).

As a shortcut the biases b_ij are lumped into the weights w_ijk and an extra "feature" is added to
the feature vector X that is always 1.

Several of these methods were in C code before but have been converted into Python because it is
either faster or no significant difference. Several are in the Cython model _ldnn.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from .classifier import Classifier

class LDNN(Classifier):
    """
    LDNN classifier from Seyedhosseini, Sajjadi, and Tasdizen - 2013. It uses a stochastic gradient
    descent with either mini-batches or the "dropout" method.
    
    There are several parameters that influence the classifier:
        N         number of groups (number of ORs)
        M         number of nodes/discriminants per group (number of ANDs)
        dropout   use "dropout" method of gradient descent described by Hinton et al. 2012
        batchsz   if not using dropout then this is the size of the batches to process
        niters    number of iterations of gradient descent to run
        rate      the step size or learning rate of gradient descent
        momentum  the momentum of the learning process of gradient descent
    """
    __weights = None # combined w_ijk and b_ij values, shaped like NxMx(n+1)
    __params = None
    
    _def_params_L0S1 = {'N':10,'M':20,'dropout':False,'batchsz':100,'niters':15,'rate':0.01,'momentum':0.5}
    _def_params_L0   = {'N':10,'M':20,'dropout':False,'batchsz':100,'niters':6, 'rate':0.01,'momentum':0.5}
    _def_params      = {'N':24,'M':24,'dropout':True,               'niters':15,'rate':0.05,'momentum':0.5}
    
    @classmethod
    def get_default_params(cls, stage, level):
        if level == 0: return cls._def_params_L0S1 if stage == 1 else cls._def_params_L0
        return cls._def_params
    
    def __init__(self, params=None, weights=None):
        """
        Creates or loads an LDNN classifier.
        
        The parameters for the classifer are specified as a dictionary. See the class documentation
        for available parameters.
        
        If weights is provided as a string that file is loaded as an NPY file. If it is an array it
        must be NxMxn (where n can be any value). If not provided then the classifier needs to be
        learned before it can be evaluated.
        """
        self.__params = LDNN._def_params.copy()
        if params is not None: self.__params.update(params)
        if weights is not None:
            if isinstance(weights, basestring):
                from os.path import abspath, exists
                from numpy import load
                path = abspath(weights)
                if exists(path): self.__load(load(path, 'r'))
                else: raise ValueError('weights cannot be loaded')
            else: self.__load(weights)
    def __load(self, weights):
        """Internal load function - checks the weights and sets the submodel as loaded."""
        from numpy import float64, ndarray
        assert(weights.shape[:2] == (self.__params['N'],self.__params['M']))
        self.__weights = weights.view(ndarray).astype(float64, copy=False)
    def save_weights(self, path):
        from numpy import save
        save(path, self.__weights)
    @property
    def learned(self): return self.__weights is not None
    @property
    def features(self): return self.__weights.shape[-1] if self.learned else None
    @property
    def extra_features(self): return 1
    def copy(self): return self.__class__(self.__params)
    @property
    def evaluation_memory(self):
        from numpy import float64
        N,M,n,itmsz = self.__params['N'], self.__params['M'], self.features or 0, float64(0).itemsize
        return itmsz*(n+N*(M+1))
    def evaluate(self, X, nthreads=1):
        assert(self.__weights is not None)
        from .utils import set_lib_threads
        set_lib_threads(nthreads)
        return f_(sigma_ij(self.__weights, X), self.__params.get('dropout', False))
    def learn(self, X, Y, nthreads=1):
        assert(self.__weights is None)
        from .utils import set_lib_threads
        set_lib_threads(nthreads)
        self.__weights = learn(X, Y, **self.__params) # TODO: save?

def f_(s, dropout=False, out=None):
    """
    Calculate f = 1-prod(1-prod(sigma_ij)) from equation 10 in Seyedhosseini et al 2013.
    
    Inputs:
        s       NxMxP matrix of sigma_ij values where N is the number of groups (ORs), M is the
                number of nodes/discriminants per group (ANDs), and P is the number of pixels
        dropout Calculate 1-sqrt(prod(1-sqrt(prod(sigma_ij)))) [default is False]
    
    Returns a contiguous length-P array of the classifier results.
    """
    from numpy import rollaxis, prod, sqrt, subtract
    g = prod(rollaxis(s, 2), axis=2)
    del s
    f = prod(subtract(1.0, sqrt(g, g) if dropout else g, out=g), axis=1, out=out)
    del g
    return subtract(1.0, sqrt(f, f) if dropout else f, out=f)

def f_g(s, out_f=None, out_g=None):
    """
    Calculate g_i=prod(sigma_ij) and f=1-prod(1-g_i) from equation 10 in Seyedhosseini et al 2013.
    
    Inputs:
        s    NxMxP matrix of sigma_ij values where N is the number of groups (ORs), M is the number
             of nodes/discriminants per group (ANDs), and P is the number of pixels
    
    Returns a contiguous length-P array of the classifier results (f) and a C-contiguous NxP matrix
    of the intermediate g_i values.
    """
    from numpy import rollaxis, prod, subtract
    if out_g is not None: out_g = out_g.T
    g = prod(rollaxis(s, 2), axis=2, out=out_g)
    del s
    f = prod(subtract(1.0, g), axis=1, out=out_f)
    return (subtract(1.0, f, out=f), g.T)

def sigma_ij(W, X, out=None):
    """
    Calculates sigma_ij = 1/(1+exp(-W*X+B))) from equation 9 in Seyedhosseini et al 2013.
    
    Inputs:
        W   NxMx(n+1) matrix of weights representing all w_ijk and biases b_ij (in the final row).
            n is the number of features, N is the number of groups (ORs), and M is the  of
            nodes/discriminants per group (ANDs)
        X   nxP or (n+1)xP matrix of pixels where P is the number of pixels. If it is (n+1)xP then
            the last row will be filled with 1s by this method. Using an (n+1)xP input is faster.
    
    Outputs:
        out NxMxP matrix of the sigma_ij values

    It is best if W, X, and out are contiguous, particularily C-contiguous.
    """
    # Some notes on "dot":
    #   regardless of inputs always produces C-contiguous output
    #   fastest when both inputs are C ordered and ~10% slower when both inputs are F ordered
    #   mixed inputs (one input is C and the other is F) lie inbetween
    from numpy import exp, negative, divide, add
    N,M,n = W.shape # the n is actually n+1
    n_,P = X.shape
    if out is not None:
        if out.shape != (N,M,P): raise ValueError()
        out = out.reshape(N*M, P)
    W = W.reshape(N*M, n)
    if n_ == n:
        X[-1].fill(1)
        s = W.dot(X, out=out)
    elif n_ == n - 1:
        s = W[:,:-1].dot(X, out=out)
        s += W[:,-1:]
    else: raise ValueError()
    del W,X
    return divide(1.0, add(1.0, exp(negative(s, out=s), out=s), out=s), out=s).reshape(N, M, P)

def __print(s):
    """Like print(...) but pre-pends the current timestamp and forces a flush."""
    import sys, datetime
    print('%s %s'%(str(datetime.datetime.utcnow())[:19], s))
    sys.stdout.flush()

__cy_ldnn = None
def __cy():
    """
    Import the Cython __ldnn module which is a bit difficult - mainly we just need to make sure the
    mtrand module is loaded first.
    """
    global __cy_ldnn #pylint: disable=global-statement
    if __cy_ldnn is None:
        import numpy, ctypes, os.path
        path = os.path.abspath(os.path.join(numpy.get_include(), '..', '..', 'random', 'mtrand'))
        for ext in ('.so', '.dylib', '.pyd', '.dll'):
            if os.path.isfile(path+ext): path += ext; break
        else: raise ImportError('Cannot find mtrand library')
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        try: import numpy.random.mtrand # this may just always work, but I don't know
        except ImportError: pass
        from . import __ldnn
        __cy_ldnn = __ldnn
    return __cy_ldnn

def learn(X, Y, N=10, M=20, dropout=False, niters=15, batch_size=100, rate=0.01, momentum=0.5, disp=True): #pylint: disable=too-many-arguments
    """
    Calculate the initial weights using multilevel K-Means clustering on a subset of the data. The
    clusters are used according to the end of section 3 of Seyedhosseini et al 2013.
    
    Inputs:
        X   (n+1)xP matrix of feature vectors where n is the number of features and P is the number
            of pixels, the last row will be filled with 1s by this method, must be Fortran-ordered
        Y   P-length array of bool labels
    
    Parameters:
        N           number of groups to create (ORs)
        M           number of nodes/discriminants per group to create (ANDs)
        dropout     use "dropout" gradient descent
        niters      number of times to go through all of the samples during gradient descent
        batch_size  size of the mini-batches in gradient descent if dropout is False
        rate        gradient descent learning rate
        momentum    amount the previous gradient influences the next gradient
        disp        display messages about progress
    
    Output:
        A C-contiguous NxMx(n+1) matrix of initial weights representing all w_ijk and biases b_ij
        (in the final row).
    """
    if disp:
        __print('    Number of training samples = %d'%X.shape[1])
        __print('    Clustering...')
    W = init_weights(X, Y, N, M)
    if disp: __print('    Gradient descent...')
    if dropout:
        gradient_descent_dropout(X, Y, W, niters, rate, momentum, disp)
    else:
        gradient_descent(X, Y, W, niters, batch_size, rate, momentum, disp)
    return W

def init_weights(X, Y, N, M):
    """
    Calculate the initial weights using multilevel K-Means clustering on a subset of the data. The
    clusters are used according to the end of section 3 of Seyedhosseini et al 2013.
    
    Inputs:
        X   (n+1)xP matrix of feature vectors where n is the number of features and P is the number
            of pixels, the last row will be filled with 1s by this method, must be Fortran-ordered
        Y   P-length array of bool labels
        
    Parameters:
        N   number of groups to create (ORs)
        M   number of nodes/discriminants per group to create (ANDs)

    Output:
        A C-contiguous NxMx(n+1) matrix of initial weights representing all w_ijk and biases b_ij
        (in the final row).
    """
    from numpy import moveaxis
    from numpy.linalg import norm
    
    assert(X.ndim == 2 and Y.ndim == 1 and X.shape[1] == Y.shape[0])
    assert(X.flags.f_contiguous and Y.flags.forc)
    
    #from numpy.random import randn
    #return randn(N, M, X.shape[0]) #random initialization
    
    # NOTE: the extra 'feature' is set to all 1s at this point and won't effect kmeans
    X[-1,:] = 1.0
   
    # First bullet point on page 5 of Seyedhosseini et al 2013
    # Calculating C+ and C-
    calc_kmeans = __cy().calc_kmeans #pylint: disable=no-member
    Cp = calc_kmeans(N, X,  Y)
    Cn = calc_kmeans(M, X, ~Y)

    # Second and Third bullet points on page 5 of Seyedhosseini et al 2013
    # Calculating initial weights and biases
    # Weights are unit-length vectors from all pairs from C- to C+
    # Biases are setup so that sigma values are 0.5 at the midpoint between C- and C+
    weights = Cp[:,:,None] - Cn[:,None,:]
    baises  = Cp[:,:,None] + Cn[:,None,:]
    del Cp, Cn
    weights /= norm(weights, axis=0) # normalize the vectors
    baises *= weights
    baises = baises.sum(axis=0, out=weights[-1,:,:])
    baises *= -0.5
    return moveaxis(weights, 0, 2)

def gradient_descent(X, Y, W, niters=15, batch_size=100, rate=0.01, momentum=0.5, disp=True):
    """
    Uses stochastic gradient descent with mini-batches to minimize the weights of the LDNN
    classifier using the training data provided. Parts of this are implemented in Cython.
    
    Inputs:
        X   (n+1)xP matrix of feature vectors where n is the number of features and P is the number
            of pixels, the last row will be filled with 1s by this method, must be Fortran-ordered
        Y   P-length array of bool labels
        
    Inputs/Outputs:
        W   NxMx(n+1) matrix of weights and biases, C-ordered

    Outputs:
        The total error for each epoch is returned as sqrt(sum(error^2)/P)

    Parameters:
        niters      number of times to go through all of the samples
        batch_size  size of the mini-batches
        rate        influence of gradient
        momentum    amount the previous gradient influences the next gradient
        disp        display messages about progress
    """
    #pylint: disable=too-many-locals
    from numpy import arange, zeros, empty
    from numpy.random import shuffle
    from math import sqrt
    
    assert(Y.ndim == 1 and W.ndim == 3 and X.shape == (W.shape[2],Y.shape[0]))
    assert(X.flags.f_contiguous and Y.flags.forc and W.flags.c_contiguous)
    
    # Matrix sizes
    (N,M,n),P = W.shape,len(Y)
    
    # Cython functions
    grad, desc = __cy().gradient, __cy().descent #pylint: disable=no-member
    
    # Allocate memory
    order = arange(P)
    total_error = empty(niters)
    f_full,g_full = empty(batch_size), empty((N,batch_size))
    grads, prevs = empty((N,M,n)), zeros((N,M,n)) # ~2 MB each
    
    for it in xrange(niters):
        shuffle(order)
        totalerror = 0.0
        f,g = f_full,g_full
        for p in xrange(0, P, batch_size):
            x = X[:,order[p:p+batch_size]]
            y = 0.8*Y[order[p:p+batch_size]]; y+=0.1 # target 0.1 and 0.9 - see LeCun, Bottou, Orr, Muller 1998 section 4.5

            # Calculate the sigmas, gs, and classifier (eqs 9 and 10 from Seyedhosseini et al 2013) 
            s = sigma_ij(W, x) # slower with out argument
            if len(y) != batch_size: f,g = f_g(s) # the last mini-batch might not be full
            else: f_g(s, f, g)

            # Calculate error (part of eq 11 from Seyedhosseini et al 2013)
            yf2 = y-f; yf2 *= yf2
            totalerror += yf2.sum()
            
            # Calculate gradient (eqs 12 and 13 from Seyedhosseini et al 2013)
            grad(f, g, s, x, y, grads) # implemented in Cython
            
            # Update weights with a gradient step
            desc(grads, prevs, W, rate, momentum) # implemented in Cython

        # Report total error for the iteration
        total_error[it] = sqrt(totalerror/P) # Similar to eq 11 from Seyedhosseini et al 2013 but slightly different
        if disp: __print('       Iteration #%d error=%f' % (it+1,total_error[it]))
    return total_error

def gradient_descent_dropout(X, Y, W, niters=15, rate=0.05, momentum=0.5, disp=True):
    """
    Uses stochastic gradient descent with "dropout" to minimize the weights of the LDNN classifier
    using the training data provided. The dropout method was proposed by Hinton et al 2012. Nearly
    completely implemented in Cython.
    
    Inputs:
        X   (n+1)xP matrix of feature vectors where n is the number of features and P is the number
            of pixels, the last row will be filled with 1s by this method, must be Fortran-ordered
        Y   P-length array of bool labels
        
    Inputs/Outputs:
        W   NxMx(n+1) matrix of weights and biases, C-ordered

    Outputs:
        The total error for each epoch is returned as sqrt(sum(error^2)/P)

    Parameters:
        niters      number of times to go through all of the samples
        rate        influence of gradient
        momentum    amount the previous gradient influences the next gradient
        disp        display messages about progress
    """
    #pylint: disable=no-member
    from numpy import int8
    assert(X.shape == (W.shape[2],Y.shape[0]))
    return __cy().gradient_descent_dropout(X, Y.view(int8), W, niters, rate, momentum, disp)
