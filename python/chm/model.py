"""
CHM Models. An OO version of several of the MATLAB functions and allows for adapting to future types
of models, including pure-Python models.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty
try: import cPickle as pickle
except ImportError: import pickle
        
__all__ = ['Model']

class Model(object):
    """
    Represents an entire model, across all stages and levels. Has the properties path, nstages, and
    nlevels along with retrieving a sub-model through model[stage,level] and parameters of the model
    may also be available through model['param'].
    """
    __metaclass__ = ABCMeta
    def __init__(self, path, model):
        self._path = path
        self._nstages = len(model)
        self._nlevels = len(model[0]) - 1
        self._model = model
    @property
    def path(self): return self._path
    @property
    def nstages(self): return self._nstages
    @property
    def nlevels(self): return self._nlevels
    def __contains__(self, name): return False
    def _get_param(self, name): raise KeyError() #pylint: disable=unused-argument, no-self-use
    def __getitem__(self, i):
        if isinstance(i, tuple) and len(i) == 2: return self._model[i[0]-1][i[1]]
        return self._get_param(i)
    def __iter__(self):
        for stage in self._model:
            for level in stage:
                yield level
    # When pickled just reloads completely from the path to the model
    def __getstate__(self): return self._path
    def __setstate__(self, state): self.__init__(state)
    
    @staticmethod
    def load(path):
        """
        Loads a model from a path. Examines the path location to determine the model type to use. If
        a model is given, it is returned un-altered.
        """
        if isinstance(path, Model): return path
        from os.path import abspath, join, isdir, isfile
        path = abspath(path)
        if isdir(path) and isfile(join(path, 'param.mat')): return MatlabModel(path)
        return PythonModel(path)

    @staticmethod
    def create(path, nstages, nlevels, fltr, cntxt_fltr=None, restart=False, **extra_info):
        """
        Creates a new, blank, model that will be saved to the given path. It will have the given
        number of stages and levels (with the final stage only having a single level). If filter or
        cntxt_fltr is an instance of Filter then they will be used for all submodels. They can also
        be an iterable of Filter objects and then they are used for each distinct level. Finally,
        they can be an iterable of iterables of Filters which gives every submodel a different
        filter. Additionally, if cntxt_fltr is not given it defaults to `Intensity.Stencil(7)`.

        Initially, all submodels in the model are blank. Blank submodels within a blank model cannot
        be used for evaluation. Once their 'learn' method is called they are no longer blank.

        If `restart` is True, then the model is re-loaded if it already exists, all submodels that
        are blank are deleted along with any stages or levels that are not part of the model anymore
        or need to be re-done due to a change in number of stages or levels. Then blank submodels
        are added as necessary which are given the filters and context filters specified.
        """
        #pylint: disable=protected-access
        return PythonModel._create(path, nstages, nlevels, fltr, cntxt_fltr, restart, extra_info)

    @staticmethod
    def iter_stg_lvl(nstages, nlevels):
        """
        Utility to iterate over nstages and nlevels properly: go over 1 to nstages-1 and nested 0 to
        nlevels then produce (nstages, 0).
        """
        for s in xrange(1,nstages):
            for l in xrange(nlevels+1):
                yield (s,l)
        yield (nstages, 0)

    @staticmethod
    def nested_list(nstages, nlevels, fn):
        """
        Utility to produce a nested list such that the outer list is stages and inner list is
        levels. The last stage has a single level.
        """
        return [[fn(s, l) for l in xrange(0,nlevels+1)] for s in xrange(1,nstages)]+[[fn(nstages, 0)]]
        
class MatlabModel(Model):
    """
    A model created by MATLAB. The filters used have increased compatibility with the original
    MATLAB code, even in situations where the filters were originally implemented incorrectly. The
    model data is stored in single folder in the MAT files param.mat and MODEL_level#_stage#.mat.
    """
    def __init__(self, path):
        from os.path import join
        from pysegtools.general.matlab import openmat
        self.__params = openmat(join(path, 'param.mat'), 'r')
        model = Model.nested_list(int(self.__params['Nstage'].data[0]),
                                  int(self.__params['Nlevel'].data[0]),
                                  lambda s,l:AndOrNetSubModel.loadmat(self, path, s, l))
        super(MatlabModel, self).__init__(path, model)
    def __contains__(self, name): return name in self.__params
    def _get_param(self, name):
        """Gets a value stored in the params.mat file of the model."""
        from pysegtools.general.matlab import mat_nice
        return mat_nice(self.__params[name])

    __filters = None
    __cntxt_fltr = None
    @classmethod
    def get_filters(cls):
        """Get the fixed set of filters used by MATLAB models."""
        if cls.__filters is None:
            from .filters import FilterBank, Haar, HOG, Edge, Gabor, SIFT, Intensity
            cls.__filters = FilterBank((Haar(True), HOG(), Edge(True), Gabor(True), SIFT(True), Intensity.Stencil(10)))
            cls.__cntxt_fltr = Intensity.Stencil(7)
        return cls.__filters, MatlabModel.__cntxt_fltr

class PythonModel(Model):
    """
    A model created by Python. The model data is stored in a set of files, the 'master' file listing
    the relative locations of the other files. This is an additional info file for each sub-model
    which references an npy file which has the actual sub-model data. The default organization is a
    single folder with all files and the sub-model files named like the the main with with the
    addition of -#-#(.npy) where the # are the stage and level of the submodel. The initializer
    accepts either the path to the main model info file or a folder that contains the main model
    file named 'model'. The info files are pickled dictionaries.
    """
    def __init__(self, path, get_sub_model=None, info=None):
        from os.path import isdir, join, dirname, abspath, relpath
        if info is None and isdir(path): path = join(path, 'model')
        folder = dirname(path)
        self._path = path
        if info is None:
            with open(path, 'rb') as f: info = pickle.load(f)
            self._nstages = info['nstages']
            self._nlevels = info['nlevels']
            model = [[AndOrNetSubModel.load(self, abspath(join(folder,sm))) for sm in sms]
                     for sms in info['submodels']]
        else:
            self._nstages = nstages = info['nstages']
            self._nlevels = nlevels = info['nlevels']
            info['submodels'] = [[relpath(sm, folder) for sm in sms] for sms in info['submodels']]
            model = Model.nested_list(nstages, nlevels, lambda s,l:get_sub_model(self,s,l))
        self.__info = info
        super(PythonModel, self).__init__(path, model)
    def __contains__(self, name): return name in self.__info
    def _get_param(self, name): return self.__info[name]
    @staticmethod
    def _create(path, nstgs, nlvls, fltr, cntxt_fltr, restart, extra_info):
        """
        Create a new Python model (or possibly restart a previous model). See `Model.create` for
        information about the parameters except `restart`.
        """
        from os.path import abspath, isdir, dirname, exists, join
        from os import mkdir
        
        # Basic argument validation
        path = abspath(path)
        if restart and not exists(path): restart = False
        if nstgs < 2: raise ValueError('nstages')
        if nlvls < 1: raise ValueError('nlevels')
        fltr = PythonModel.__get_filters(nstgs, nlvls, fltr)
        if cntxt_fltr is None:
            from .filters import Intensity
            cntxt_fltr = Intensity.Stencil(7)
        cntxt_fltr = PythonModel.__get_filters(nstgs, nlvls, cntxt_fltr) #pylint: disable=redefined-variable-type
        
        # Get the submodels
        if restart:
            if isdir(path): path = join(path, 'model')
            folder = dirname(path)
            sms,info = PythonModel.__check_submodels(path, nstgs, nlvls)
            if sms is None: restart = False
        else:
            if not exists(path): mkdir(path)
            elif not isdir(path): raise ValueError('If path exists, it must be a directory')
            folder,path = path, join(path, 'model')
        if restart:
            info.update(extra_info)
            info['submodels'] = Model.nested_list(nstgs, nlvls, lambda s,l:sms[s-1][l] or ('%s-%d-%d'%(path,s,l)))
            def get_sub_model(m,s,l):
                # Load or create
                if sms[s-1][l] is not None:
                    return AndOrNetSubModel.load(m,abspath(join(folder,sms[s-1][l])))
                return AndOrNetSubModel.create(m,s,l,fltr[s-1][l],cntxt_fltr[s-1][l])
        else:
            info = dict(extra_info)
            info['submodels'] = Model.nested_list(nstgs, nlvls, lambda s,l:join(folder,'model-%d-%d'%(s,l)))
            # Always loads
            get_sub_model = lambda m,s,l:AndOrNetSubModel.create(m,s,l,fltr[s-1][l],cntxt_fltr[s-1][l])
            
        # Save the model data
        info['nstages'] = nstgs
        info['nlevels'] = nlvls
        with open(path, 'wb') as f: pickle.dump(info, f, 2)
        
        # Load the model
        return PythonModel(path, get_sub_model, info)

    @staticmethod
    def __check_submodels(path, nstgs, nlvls):
        """
        Loads all of the submodels for model while restarting, deleting submodels that cannot be
        used while restarting. The `nstgs` and `nlvls` are the new number of stages and levels. The
        `path` is the path to the model file itself.
        
        Returns the submodel paths in a list-of-lists (with None for any the need to be recreated)
        and the loaded info from the model.
        """
        from os.path import abspath, dirname, exists, join
        from os import remove
        folder = dirname(path)
        try:
            with open(path, 'rb') as f: info = pickle.load(f)
        except (pickle.UnpicklingError, IOError, EOFError): return None,None
        if not isinstance(info, dict) or 'submodels' not in info: return None,None
        sms = info['submodels']
        max_stg_keep = 1 if info['nlevels'] != nlvls else nstgs # if number of levels changed, any stage above 1 needs to be trashed
        for s,sm in enumerate(sms):
            for l,sm in enumerate(sm):
                sm = abspath(join(folder, sm))
                if not exists(sm): sms[s][l] = None; continue
                with open(sm, 'rb') as f: sm_info = pickle.load(f)
                s,l = sm_info['stage'], sm_info['level']
                disc = sm_info.get('discriminants')
                disc = None if disc is None else abspath(join(dirname(sm), disc))
                remove_it = s > max_stg_keep or l > nlvls or (s == nstgs and l != 0)
                if remove_it or disc is None or not exists(disc):
                    remove(sm)
                    sms[s-1][l] = None
                    if disc is not None and exists(disc): remove(disc)
        remove_rem = False
        for s,sm in enumerate(sms):
            for l,sm in enumerate(sm):
                if remove_rem and sm is not None:
                    sms[s][l] = None
                    if not exists(sm): continue
                    sm = abspath(join(folder, sm))
                    with open(sm, 'rb') as f: sm_info = pickle.load(f)
                    remove(sm)
                    disc = sm_info.get('discriminants')
                    if disc is not None:
                        disc = abspath(join(dirname(sm), disc))
                        if exists(disc): remove(disc)
                elif sm is None: remove_rem = True
        return sms, info

    @staticmethod
    def __get_filters(nstgs, nlvls, f):
        """
        Gets a nested list of filter objects given either a filter, a non-nested list of filters,
        or a nested list of filters. If a single filter is given, it is used for every level of
        every stage. If a non-nested list is used, it is used as the filters for each level and
        copied for each stage. Otherwise it is returned as-is.
        """
        from .filters import Filter
        if isinstance(f, Filter):
            # single filter for all stages and levels
            return [[f]*(nlvls+1)]*(nstgs-1)+[[f]]
        f = list(f)
        if isinstance(f[0], Filter):
            # a filter for each level, but same set for all stages
            if len(f) != nlvls: raise ValueError('wrong number of filters')
            return [f]*(nstgs-1)+[[f[0]]]
        return f # assume a list of lists of filters

class SubModel(object):
    """Represents part of a model for a single stage and level."""
    __metaclass__ = ABCMeta

    def __init__(self, model, stage, level, fltr, cntxt_fltr):
        import weakref
        self._model = None if model is None else weakref.ref(model)
        self._stage = stage
        self._level = level
        self._filter = fltr
        self._context_filter = cntxt_fltr
        self._loaded = False

    @property
    def model(self): return self._model() # _model is a weak-reference
    @property
    def level(self): return self._level
    @property
    def stage(self): return self._stage
    @property
    def loaded(self): return self._loaded
    def _set_loaded(self): self._loaded = True
    
    @abstractproperty
    def features(self):
        """
        Returns the number of features used by this model. This is at least:
            self.image_filter.features + self.ncontexts * self.context_filter.features
        If it is more than that, additional features do not need to be initialized before evaluating
        or learning.
        """
        return self.image_filter.features + self.ncontexts * self.context_filter.features
    
    @property
    def image_filter(self):
        """Returns the filter used for images for this model."""
        return self._filter
    
    @property
    def context_filter(self):
        """Returns the filter used for the contexts for this model."""
        return self._context_filter

    @property
    def ncontexts(self):
        """
        Returns the number of contexts this model uses, if level is not 0 and stage is not 1, this
        is self.level otherwise it is self.model.nlevels+1.
        """
        return self.level if self.stage == 1 or self.level > 0 else (self.model.nlevels+1)

    def filter(self, im, contexts, out=None, region=None, cntxt_rgn=None, nthreads=1):
        """
        Filters an image and the contexts for use with this model.

        im        image to filter, already downsampled as appropiate
        contexts  context images
        out       output data (default is to allocate it)
        region    region of im to use (default uses all)
        cntxt_rgn the region of the contexts to use (default uses region)
        nthreads  the number of threads to use (default is single-threaded)
        """
        import gc
        from numpy import empty, float64
        
        F_image = self.image_filter
        F_cntxt = self.context_filter

        if len(contexts) != self.ncontexts: raise ValueError('Wrong number of context images')
        if region is None: region = (0, 0, im.shape[0], im.shape[1])
        if cntxt_rgn is None: cntxt_rgn = region

        # Create the feature matrix
        sh = (self.features, region[2]-region[0], region[3]-region[1])
        if out is None: out = empty(sh)
        elif out.shape != sh or out.dtype != float64:
            raise ValueError('Output not right shape or data type')
            
        # Calculate filter features
        F_image(im, out=out[:F_image.features], region=region, nthreads=nthreads)
        gc.collect()

        # Calculate context features
        x = F_image.features
        for cntxt in contexts:
            y = x+F_cntxt.features
            F_cntxt(cntxt, out[x:y], cntxt_rgn, nthreads)
            x = y

        # Done!
        return out
        
    @abstractmethod
    def _evaluate(self, X, nthreads):
        """
        Evaluates the feature matrix X with the model. Internal method to be implemented by
        inheritors. The array X has been checked to be 2 dimensional float64 with the first
        dimension equal to self.features at this point.
        """
        pass
    
    def evaluate(self, X, nthreads=1):
        """
        Evaluates the feature matrix X with the model (matrix is features by pixels). The sub-model
        must have been loaded or learned before this is called.
        """
        if not self._loaded: raise ValueError('Model no loaded/learned')
        import gc
        gc.collect() # evaluation is very memory intensive, make sure we are ready
        from numpy import float64
        if X.ndim < 2: raise ValueError('X must be at least 2D')
        if X.shape[0] != self.features: raise ValueError('X has the wrong number of features')
        sh = X.shape[1:]
        X = X.astype(float64, copy=False).reshape((X.shape[0], -1))
        return self._evaluate(X, nthreads).astype(float64, copy=False).reshape(sh)

    @abstractproperty
    def evaluation_memory(self):
        """The amount of memory required to evaluate the model per pixel."""
        return 0

    @abstractmethod
    def _learn(self, X, Y, nthreads):
        """
        Learns the feature matrix X (features by pixels) with Y is the labels with a length of
        pixels.Internal method to be implemented by inheritors. The array X has been checked to be
        2 dimensional float64 with the first dimension equal to self.features at this point along
        with Y being a 1-dimensional bool array of the same length as X.
        """
        pass

    def learn(self, X, Y, nthreads=1):
        """
        Learns the feature matrix X (features by pixels) with Y is the labels with a length of
        pixels.
        """
        if self._loaded: raise ValueError('Model already loaded/learned')
        import gc
        gc.collect() # learning is very memory intensive, make sure we are ready
        from numpy import float64
        if X.ndim < 2: raise ValueError('X must be at least 2D')
        if X.shape[0] != self.features: raise ValueError('X has the wrong number of features')
        X = X.astype(float64, copy=False).reshape((X.shape[0], -1))
        if Y.ndim != 1 or Y.shape[0] != X.shape[1]: raise ValueError('Y must be 1D of the same length as X')
        if Y.dtype != bool: Y = Y > 0
        self._learn(X, Y, nthreads)

class AndOrNetSubModel(SubModel):
    """
    AndOrNet Submodel.

    Originally in LearnAndOrNetMEX, UpdateDiscriminants, EvaluateAndOrNetMEX, genOutput
    """
    _disc = None
    _info = None
    # nGroup == nORs, nDiscriminantPerGroup == nANDs
    _nGrp, _nDPG = 24, 24

    @staticmethod
    def loadmat(model, path, stage, level):
        """
        Loads a AndOrNet sub-model from a path to a folder and the stage/level. This may return a
        AndOrNetSubModel_SB if the model indicates it is the SB variety.
        """
        #pylint: disable=protected-access
        from pysegtools.general.matlab import openmat, mat_nice
        from numpy import float32
        from os.path import join
        path = join(path, 'MODEL_level%d_stage%d.mat'%(level,stage))
        with openmat(path, 'r') as mat: info = mat_nice(mat['model'])
        smtype = AndOrNetSubModel_SB if model['discriminants'].dtype == float32 else AndOrNetSubModel
        assert(smtype._nGrp == info['nGroup'] and smtype._nDPG == info['nDiscriminantPerGroup'])
        f, cf = MatlabModel.get_filters()
        sm = smtype(model, stage, level, f, cf)
        sm._load(info.pop('discriminants')) #pylint: disable=no-member
        sm._info = info
        return sm
    @staticmethod
    def load(model, path):
        """
        Loads a AndOrNet sub-model from a path to a Python sub-model file. This may return a
        AndOrNetSubModel_SB if the model indicates it is the SB variety.
        """
        from os.path import abspath, dirname, join, exists
        path = abspath(path)
        folder = dirname(path)
        from numpy import load
        with open(path, 'rb') as f: info = pickle.load(f)
        smtype = AndOrNetSubModel_SB if info.get('sb', False) else AndOrNetSubModel
        sm = smtype(model, info['stage'], info['level'], info['filter'], info['context_filter'])
        discs = info.get('discriminants')
        #pylint: disable=protected-access
        if discs is not None:
            path = abspath(join(folder, discs))
            if not exists(path): del info['discriminants']
            else: sm._load(load(path, 'r'))
        sm._info = info
        return sm
    def _load(self, disc):
        """
        Internal load function. Checks the discriminants shape and data type and sets the submodel
        as loaded.
        """
        assert(disc.shape[0] == super(AndOrNetSubModel, self).features+1)
        assert(disc.shape[1] == self._nGrp*self._nDPG)
        from numpy import float64, ndarray
        self._disc = disc.view(ndarray).astype(float64, copy=False)
        self._set_loaded()

    @staticmethod
    def create(model, stage, level, fltr, cntxt_fltr):
        """
        Creates a new, blank, Python submodel, writing out the basic information to a file named
        after the model path with the stage and level appended. There are no discriminants since
        this submodel starts out blank - learn must be called.
        """
        smtype = AndOrNetSubModel_SB if level==0 else AndOrNetSubModel # TODO: user option?
        sm = smtype(model, stage, level, fltr, cntxt_fltr)
        sm._save() #pylint: disable=protected-access
        return sm
    def _save(self, disc=None, **extra_info):
        """
        Saves a Python submodel by writing out a pickled dictionary to the model's path with the
        stage and level names added along with an NPY file containing the discriminants. The pickled
        dictionary includes the stage, level, filter, and context filter information.
        The discriminants are only saved if available.
        """
        from os.path import basename
        path = self.model.path + ('-%d-%d'%(self.stage, self.level))

        # Collect model information
        info = dict(extra_info)
        info['stage'] = self.stage
        info['level'] = self.level
        info['filter'] = self.image_filter
        info['context_filter'] = self.context_filter
        if disc is not None:
            # Save (and load) discriminants
            from numpy import save
            info['discriminants'] = basename(path)+'.npy'
            save(path+'.npy', disc)
            self._load(load(path+'.npy', 'r'))

        # Save model information
        self._info = info
        with open(path, 'wb') as f: info = pickle.dump(info, f, 2)

    @property
    def features(self):
        if self._disc is not None: return self._disc.shape[0]
        return super(AndOrNetSubModel, self).features + 1
    @property
    def evaluation_memory(self):
        if self._disc is None:
            from numpy import float64
            return float64(0).itemsize*(self.features+self._nGrp*(self._nDPG+1))
        return self._disc.itemsize*(self._disc.shape[0]+self._disc.shape[1]+self._nDPG)
    def _evaluate(self, X, nthreads):
        # ORIGINALLY: genOutput
        # CHANGED: no longer in "native" code, instead pure Python
        from .utils import set_lib_threads
        set_lib_threads(nthreads)

        # Some notes on "dot":
        #   regardless of inputs always produces C-contiguous output
        #   fastest when both inputs are C ordered and ~10% slower when both inputs are F ordered
        #   mixed inputs (one input is C and the other is F) lie inbetween
        
        from numpy import prod, exp, divide, subtract, negative, sqrt
        #npixels = X.shape[1]
        X[-1].fill(1)
        tt = self._disc.T.dot(X) # different in SB
        del X
        # SB has extras being added
        negative(tt, out=tt)
        exp(tt, out=tt)
        tt += 1.0
        divide(1.0, tt, out=tt)
        tg = prod(tt.reshape((self._nDPG, self._nGrp, -1), order='F'), axis=0)
        del tt
        sqrt(tg, out=tg) # not in SB
        subtract(1.0, tg, out=tg)
        to = prod(tg, axis=0)
        del tg
        sqrt(to, out=to) # not in SB
        return subtract(1.0, to, out=to)
    def _learn(self, X, Y, nthreads): self._run_learn(X, Y, False, 15, nthreads)
    def _run_learn(self, X, Y, sb, maxepoch, nthreads):
        """
        A wrapper for _train.learn that calls self._save with the resulting discriminants. The
        _train.learn module is a bit tricky to load, but basically we need to make sure the mtrand
        dynamic library is loaded first.
        """
        from .utils import set_lib_threads
        set_lib_threads(nthreads)
        
        # Import the __train module
        # A bit difficult, but we just need to make sure the mtrand module is loaded first
        import numpy, ctypes, os.path
        path = os.path.abspath(os.path.join(numpy.get_include(), '..', '..', 'random', 'mtrand'))
        for ext in ('.so', '.dylib', '.pyd', '.dll'):
            if os.path.isfile(path+ext): path += ext; break
        else: raise ImportError('Cannot find mtrand library')
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        try: import numpy.random.mtrand # this may just always work, but I don't know
        except ImportError: pass
        from ._train import learn #pylint: disable=no-name-in-module

        # Run learning and save
        self._save(learn(X, Y, sb, maxepoch, nthreads), sb=sb)

class AndOrNetSubModel_SB(AndOrNetSubModel):
    """
    AndOrNet Submodel - "SB" variety.

    There are only a few differences between SB and the regular variety. The SB variety is used for
    level 0 submodels.

    During evaluation, this one handles the extra discriminants row differently and doesn't take
    square roots at various times. 

    Originally in UpdateDiscriminants_SB and genOutput_SB
    """
    _nGrp, _nDPG = 10, 20
    def _evaluate(self, X, nthreads):
        # ORIGINALLY: genOutput_SB
        # CHANGED: no longer in "native" code, instead pure Python (my Cython attempt was 100x slower than this)
        # CHANGED: returns float64 instead of float32 array
        # CHANGED: x and discriminants no longer required to be float32 matrices (but works just fine when they are)
        from .utils import set_lib_threads
        set_lib_threads(nthreads)

        from numpy import prod, exp, divide, subtract, negative
        #nfeats, npixels = X.shape
        #nfeats_p1, ndisc = discriminants.shape
        tt = self._disc[:-1,:].T.dot(X[:-1,:])
        del X
        tt += self._disc[-1,:,None]
        negative(tt, out=tt)
        exp(tt, out=tt)
        tt += 1.0
        divide(1.0, tt, out=tt)
        tg = prod(tt.reshape((self._nDPG, self._nGrp, -1), order='F'), axis=0)
        del tt
        subtract(1.0, tg, out=tg)
        to = prod(tg, axis=0)
        del tg
        return subtract(1.0, to, out=to)
    def _learn(self, X, Y, nthreads): self._run_learn(X, Y, True, 15 if self.stage == 1 else 6, nthreads)
