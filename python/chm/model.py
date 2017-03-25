"""
CHM Models. An abstraction of the classifier and CHM modelling system supporting several levels,
stages, filters, and classifiers. Can load data from MATLAB or Python based models.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function
        
__all__ = ['Model', 'SubModel']

class Model(object):
    """
    Represents an entire model, across all stages and levels. Has the properties path, nstages, and
    nlevels along with retrieving a sub-model through model[stage,level] and parameters of the
    model may also be available through model['param'].
    
    In the default implementation all of the model metadata is stored in file which is a pickled
    dictionary. Additional files may be saved with it containing various other pieces of data.
    """
    def __init__(self, path, model=None, info=None):
        from os.path import abspath, isfile
        self._path = path = abspath(path)
        if not isfile(path): raise ValueError('path')
        if model is None:
            from pysegtools.general.json import load
            info = load(path)
            self._nstages,self._nlevels,model = info['nstages'],info['nlevels'],info['submodels']
        assert(len(model[-1]) == 1 and len(model[0]) > 0 and all(len(sm) == len(model[0]) for sm in model[1:-1]))
        #pylint: disable=access-member-before-definition
        if hasattr(self, '_nstages'): assert(self._nstages == len(model))
        if hasattr(self, '_nlevels'): assert(self._nlevels == len(model[0])-1)
        self._info,self._model = info,model
        self._nstages,self._nlevels = len(model),len(model[0])-1
        for m in self: m.model = self
    @property
    def path(self): return self._path
    @property
    def nstages(self): return self._nstages
    @property
    def nlevels(self): return self._nlevels
    def __contains__(self, name): return name in self._info
    def _get_param(self, name): return self._info[name]
    def __getitem__(self, i):
        if isinstance(i, tuple) and len(i) == 2: return self._model[i[0]-1][i[1]]
        return self._get_param(i)
    def __iter__(self):
        for stage in self._model:
            for level in stage:
                yield level
    
    def save(self, path=None):
        """
        Save the model. If a path is given, that is used. Otherwise the originally loaded path or
        last path saved to is used.
        """
        self._path = path or self._path
        from pysegtools.general.json import save
        save(self._path, self._info)
   
    @classmethod
    def load(cls, path):
        """
        Loads a model from a path. Examines the path location to determine the model type to use.
        If a model is given, it is returned un-altered.
        """
        if isinstance(path, Model): return path
        from os.path import abspath, join, isdir, isfile
        path = abspath(path)
        if (isfile(path) and path.endswith('param.mat') or
            isdir(path) and isfile(join(path, 'param.mat'))): return MatlabModel(path)
        return cls(path)

    @classmethod
    def create(cls, path, nstages, nlevels, classifier, fltr, cntxt_fltr=None, restart=False, **extra_info):
        """
        Creates a new, blank, model that will be saved to the given path. It will have the given
        number of stages and levels (with the final stage only having a single level).
        
        The arguments fltr, cntxt_fltr, and classifier can be either a single value which will then
        be used for every level and stage. If they are an sequence then each value is used for the
        levels and repeated for each stage. If they are an sequence of sequences it describes a
        different Filter/Classifier for each stage and level. Finally, if not provided at all then
        the context filter defaults to `Intensity.Stencil(7)`.

        Initially, all submodels in the model are blank. Blank submodels within a blank model
        cannot be used for evaluation. Once their `learn` method is called they are no longer
        blank.

        If `restart` is True, then the model is re-loaded if it already exists, all submodels that
        are not learned are removed along with any stages or levels that are not part of the model
        anymore or need to be re-done due to a change in number of stages or levels. Then un-learned
        submodels are created as necessary from the given filters and context filters specified.
        """
        from os.path import abspath, exists
        from .filters import Filter
        from .classifier import Classifier
        from pysegtools.general.json import save
        
        # Basic argument validation
        path = abspath(path)
        if restart and not exists(path): restart = False
        if nstages < 2: raise ValueError('nstages')
        if nlevels < 1: raise ValueError('nlevels')
        classifier = cls.__expand(nstages, nlevels, classifier, Classifier)
        fltr = cls.__expand(nstages, nlevels, fltr, Filter)
        if cntxt_fltr is None:
            from .filters import Intensity
            cntxt_fltr = Intensity.Stencil(7)
        cntxt_fltr = cls.__expand(nstages, nlevels, cntxt_fltr, Filter)
        
        if restart:
            # (Re-)Create submodels and info
            info = cls.__check_submodels(path, nstages, nlevels)
            sms = info['submodels']
            model = Model.nested_list(nstages, nlevels, lambda s,l:
                (SubModel(s,l,fltr[s-1][l],cntxt_fltr[s-1][l],classifier[s-1][l])
                if s > len(sms) or l >= len(sms[s-1]) or sms[s-1][l] is None else sms[s-1][l]))
            info.update({'nstages':nstages,'nlevels':nlevels,'submodels':model})
            
        else:
            # Create submodels and info
            model = Model.nested_list(nstages, nlevels, lambda s,l:
                SubModel(s,l,fltr[s-1][l],cntxt_fltr[s-1][l],classifier[s-1][l]))
            info = {'nstages':nstages,'nlevels':nlevels,'submodels':model}

        # Save the data
        info.update(extra_info)
        save(path, info)

        # Load the model
        return cls(path)
    
    @classmethod
    def __check_submodels(cls, path, nstgs, nlvls):
        """
        Loads all of the submodels for a model while restarting, reseting submodels that cannot be
        used while restarting. The `nstgs` and `nlvls` are the new number of stages and levels. The
        `path` is the path to the model file itself. Returns the loaded info from the model with
        some submodels possibly reset.
        """
        from pysegtools.general.json import load
        try: info = load(path)
        except IOError: return None
        if not isinstance(info, dict) or not all(k in info for k in ('submodels', 'nstages', 'nlevels')): return None
        sms = info['submodels']
        max_stg = 1 if info['nlevels'] != nlvls else nstgs # if number of levels changed, any stage above 1 needs to be trashed
        reset = False
        for s,sm in enumerate(sms):
            for l,sm in enumerate(sm):
                if sm is None or sm.stage-1 != s or sm.level != l:
                    print('Corrupted model - completely replacing')
                    return None
                reset = reset or s+1 > max_stg or l > nlvls or (s+1 == nstgs and l > 0) or not sm.classifier.learned
                if reset: sms[s][l] = None # TODO: sms[s][l].copy() instead?
        return info

    @staticmethod
    def __expand(nstages, nlevels, x, clazz):
        """
        Gets a nested list of filter/classifier objects given either a filter/classifier, a
        non-nested sequence of filters/classifiers, or a nested list of filters/classifiers. If a
        single value is given, it is used for every level of every stage. If a non-nested list is
        used, it is used as the filters/classifiers for each level and copied for each stage.
        Otherwise it is returned as-is (but converted to lists).
        """
        from collections import Iterable
        dup = lambda x:(x.copy() if hasattr(x, 'copy') else x) # checks if the value has the copy method and possibly copies it
        if isinstance(x, clazz):
            # single filter/classifier for all stages and levels
            return [[dup(x) for _ in xrange(nlevels+1)] for _ in xrange(nstages-1)]+[[dup(x)]]
        x = list(x)
        if isinstance(x[0], clazz):
            # a filter/classifier for each level, but same set for all stages
            if len(x) != nlevels+1: raise ValueError('wrong number of filters/classifiers')
            return [[dup(X) for X in x] for _ in xrange(nstages-1)] + [[dup(x[0])]]
        # a list of lists of filters/classifiers
        x = [list(y) if isinstance(y, Iterable) else [y] for y in x]
        if len(x) != nstages or any(len(y) != nlevels+1 for y in x[:-1]) or len(x[-1]) != 1:
            raise ValueError('wrong number of filters/classifiers')
        return x
    
    @staticmethod
    def iter_stg_lvl(nstages, nlevels):
        """
        Utility to iterate over nstages and nlevels properly: go over 1 to nstages-1 and nested 0
        to nlevels then produce (nstages, 0).
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
        from os.path import join, dirname, isdir, abspath
        from pysegtools.general.matlab import openmat, mat_nice
        path = abspath(path)
        if isdir(path): path = join(path, 'param.mat')
        self._path = path
        folder = dirname(path)
        with openmat(path, 'r') as mat: info = {e.name:mat_nice(e.data) for e in mat}
        self._nstages, self._nlevels = int(info['Nstage']), int(info['Nlevel'])
        def load(stg, lvl):
            from numpy import float32
            from .ldnn import LDNN
            path = join(folder, 'MODEL_level%d_stage%d.mat'%(lvl,stg))
            with openmat(path, 'r') as mat: info = mat_nice(mat['model'])
            disc = info['discriminants']
            N,M = int(info['nGroup']), int(info['nDiscriminantPerGroup'])
            classifier = LDNN({'dropout':disc.dtype!=float32,'N':N,'M':M}, disc.T.reshape(N, M, -1))
            f,fc = self.__get_filters()
            return SubModel(stg, lvl, f, fc, classifier)
        model = Model.nested_list(self._nstages, self._nlevels, load)
        super(MatlabModel, self).__init__(path, model, info)

    __filters = None
    __cntxt_fltr = None
    @classmethod
    def __get_filters(cls):
        """Get the fixed set of filters used by MATLAB models."""
        if cls.__filters is None:
            from .filters import FilterBank, Haar, HOG, Edge, Gabor, SIFT, Intensity
            cls.__filters = FilterBank((Haar(True), HOG(True), Edge(True), Gabor(True), SIFT(True), Intensity.Stencil(10)))
            cls.__cntxt_fltr = Intensity.Stencil(7)
        return cls.__filters, cls.__cntxt_fltr

class SubModel(object):
    __model = None
    
    """Represents part of a model for a single stage and level."""
    def __init__(self, stage, level, fltr, cntxt_fltr, classifier):
        self.__stage = stage
        self.__level = level
        self.__filter = fltr
        self.__context_filter = cntxt_fltr
        self.__classifier = classifier

    def __getstate__(self):
        """Gets the state - everything except the model field"""
        return {
            'stage': self.__stage, 'level': self.__level,
            'filter': self.__filter, 'context_filter': self.__context_filter,
            'classifier': self.__classifier,
        }
    def __setstate__(self, state):
        self.__init__(state['stage'], state['level'], state['filter'], state['context_filter'], state['classifier'])
    
    @property
    def model(self): return self.__model() # _model is a weak-reference
    @model.setter
    def model(self, value): # should only use this in Model.__init__
        assert((isinstance(value, Model) or value is None) and self.__model is None)
        import weakref
        self.__model = None if value is None else weakref.ref(value)
    @property
    def stage(self): return self.__stage
    @property
    def level(self): return self.__level
    @property
    def image_filter(self):
        """Returns the filter used for images for this model."""
        return self.__filter
    @property
    def context_filter(self):
        """Returns the filter used for the contexts for this model."""
        return self.__context_filter
    @property
    def classifier(self):
        """Returns the classifier used for this model."""
        return self.__classifier
    def __nonzero__(self): return bool(self.classifier)
    def __bool__(self): return bool(self.classifier)

    @property
    def features(self):
        """
        Returns the number of features used by this model. This is at least:
            self.image_filter.features + self.ncontexts * self.context_filter.features
        If it is more than that, additional features do not need to be initialized before
        evaluating or learning.
        """
        return self.image_filter.features + self.ncontexts * self.context_filter.features + self.classifier.extra_features
    
    @property
    def ncontexts(self):
        """
        Returns the number of contexts this model uses, if level is not 0 and stage is not 1, this
        is self.level otherwise it is self.model.nlevels+1.
        """
        #pylint: disable=no-member
        return self.level if self.stage == 1 or self.level > 0 or self.model is None else (self.model.nlevels+1)
    
    def filter(self, im, contexts, out=None, region=None, cntxt_rgn=None, nthreads=1):
        """
        Filters an image and the contexts for use with this model. Calculates all features that
        need ot be initialized.

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
    
    def evaluate(self, X, nthreads=1):
        """
        Evaluates the feature matrix X with the model (matrix is features by pixels). The
        classifier must have been loaded or learned before this is called.
        """
        if not self.classifier.learned: raise ValueError('Model no loaded/learned')
        import gc
        gc.collect() # evaluation is very memory intensive, make sure we are ready
        from numpy import float64
        if X.ndim < 2: raise ValueError('X must be at least 2D')
        if X.shape[0] != self.features: raise ValueError('X has the wrong number of features')
        sh = X.shape[1:]
        X = X.astype(float64, copy=False).reshape((X.shape[0], -1))
        return self.classifier.evaluate(X, nthreads).astype(float64, copy=False).reshape(sh)

    def learn(self, X, Y, nthreads=1):
        """
        Learns the feature matrix X (features by pixels) with Y as the labels with a length of
        pixels.
        """
        if self.classifier.learned: raise ValueError('Model already loaded/learned')
        import gc
        gc.collect() # learning is very memory intensive, make sure we are ready
        from numpy import float64
        if X.ndim < 2: raise ValueError('X must be at least 2D')
        if X.shape[0] != self.features: raise ValueError('X has the wrong number of features')
        X = X.astype(float64, copy=False).reshape((X.shape[0], -1))
        if Y.ndim != 1 or Y.shape[0] != X.shape[1]: raise ValueError('Y must be 1D of the same length as X')
        if Y.dtype != bool: Y = Y > 0
        self.classifier.learn(X, Y, nthreads)

def __add_class_names():
    from pysegtools.general.json import add_class_name
    from .filters import Filter
    from .classifier import Classifier
    add_class_name('__submodel__', SubModel)
    add_class_name('__filter__', Filter)
    add_class_name('__classifier__', Classifier)
__add_class_names()
del __add_class_names