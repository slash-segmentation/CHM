#!/usr/bin/env python2
"""
CHM Image Training
Runs CHM training phase on a set of images. Can also be run as a command line program with
arguments.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

__all__ = ["CHM_train"]

# TODO: I know learning wants a Fortran-ordered matrix, what about filters?
#    X[:,start:end].reshape((-1, im.shape)).flags.(c_|f_)contiguous
# is False for order='F' or 'C', however:
#    X[:,start:end].reshape((-1, im.shape))[0].flags.c_contiguous
# is True for order='C' but the equivelent is False for order='F'

def CHM_train(ims, lbls, model=None, masks=None, nthreads=None):
    """
    
    """
    from itertools import izip
    from psutil import cpu_count
    from numpy import empty
    from .utils import im2double, copy

    # Basic checks of images, labels, and masks
    if len(ims) < 1 or len(ims) != len(lbls): raise ValueError('You must provide at least 1 image set and equal numbers of training and label images')
    if any(len(im.dtype) > 1 or im.dtype.kind not in ('iufb') for im in ims): raise ValueError('Images must be grayscale')
    if any(len(lbl.dtype) > 1 or lbl.dtype.kind not in ('iufb') for lbl in lbls): raise ValueError('Labels must be grayscale')
    shapes = [im.shape for im in ims]
    if any(sh != lbl.shape for sh,lbl in izip(shapes,lbls)): raise ValueError('Labels must be the same shape as the corresponding images')
    if masks is not None:
        if len(ims) != len(masks): raise ValueError('The number of mask images must be equal to the number of training/label images')
        if any(len(mask.dtype) > 1 or mask.dtype.kind not in ('iufb') for mask in masks): raise ValueError('Masks must be grayscale')
        if any(sh != mask.shape for sh,mask in izip(shapes,masks)): raise ValueError('Masks must be the same shape as the corresponding images')
        
    # Load the images and labels for level 0
    ims0   = [im2double(im.data) for im in ims]
    lbls0  = [lbl.data>0 for lbl in lbls]
    masks0 = None if masks is None else [mask.data>0 for mask in masks]

    # Get paramters
    shapes = __get_all_shapes(shapes, model.nlevels)
    if nthreads is None: nthreads = cpu_count(True)

    ########## CHM Train Core ##########
    contexts, clabels = None, None # contexts and clabels are indexed by image-index then level
    for sm in model:
        __print('Training stage %d level %d...'%(sm.stage,sm.level))

        ##### Update images, labels, contexts #####
        if sm.level == 0:
            ims, lbls = ims0[:], lbls0[:]
            if masks0 is not None: masks = masks0[:]
            contexts, clabels = __reset_contexts(clabels, shapes, nthreads)
        else:
            __downsample_images(ims, lbls, masks, contexts, clabels, nthreads)

        if sm.loaded:
            __print('  Skipping... (already completed)')
            __load_clabels(sm, clabels)
            continue

        ##### Feature Extraction #####
        __print('  Extracting features...')
        X_full, Y, M = __extract_features(sm, ims, lbls, masks, contexts, nthreads)

        ##### Learning the classifier #####
        __print('  Learning...')
        if M is None:
            # Convert to Fortran-ordered
            X = empty(X_full.shape, order='F')
            copy(X, X_full, nthreads)
        else:
            # Compress and convert to Fortran-ordered
            # OPT: parallelize compress
            X = empty((X_full.shape[0], M.sum()), order='F')
            X_full.compress(M, 0, X)
        #__subsample(X, Y, 3000000, nthreads=nthreads) # TODO: use? increase this for real problems?
        del M
        sm.learn(X, Y, nthreads)
        del X, Y

        ##### Generate the outputs #####
        __print('  Generating outputs...')
        __generate_outputs(sm, X_full, shapes[sm.level], nthreads)
        del X_full
        __load_clabels(sm, clabels)

    ########## Return final labels ##########
    __print('Complete!')
    return [clbl[0] for clbl in clabels]
    
def __print(s):
    """Like print(...) but pre-pends the current timestamp and forces a flush."""
    import sys, datetime
    print('%s %s'%(str(datetime.datetime.utcnow())[:19], s))
    sys.stdout.flush()
    
def __get_all_shapes(shapes, nlvls):
    """Get the downsampled image shapes from the image shapes at level 0"""
    all_shapes = [None]*(nlvls+1)
    all_shapes[0] = shapes
    for lvl in xrange(1, nlvls+1):
        shapes = [(((sh[0]+1)//2),((sh[1]+1)//2)) for sh in shapes]
        all_shapes[lvl] = shapes
    return all_shapes

def __reset_contexts(clabels, shapes, nthreads):
    """
    Get the reset context and clabels variabls when reaching level 0 (either the first time or any
    subsequent time). `clabels` is None if it is the first time, otherwise it is the previous
    clabels as a list of lists, indexed by image-index then level). The shapes are the shapes of
    all of the images as a list of lists, indexed by level then image-index. Returns the appropiate
    contexts and clabels values.
    """
    n = len(shapes[0]) # number of images
    cntxts = [[] for _ in xrange(n)] if clabels is None else \
             [[__upsample(c, lvl, shapes[0][i], nthreads) for lvl,c in enumerate(clbls)]
              for i,clbls in enumerate(clabels)]
    return cntxts, [[] for _ in xrange(n)]

def __upsample(im, L, sh, nthreads):
    """Like MyUpSample but constrains the final shape to sh"""
    from .utils import MyUpSample
    return MyUpSample(im,L,nthreads=nthreads)[:sh[0],:sh[1]]

def __downsample_images(ims, lbls, masks, contexts, clabels, nthreads):
    """
    Downsample the images, labels, and contexts when going from one level to the next. This
    operates on the lists in-place.
    """
    from .utils import MyDownSample, MyMaxPooling
    ims[:]  = [MyDownSample(im,  1, nthreads=nthreads) for im  in ims ]
    lbls[:] = [MyMaxPooling(lbl, 1, nthreads=nthreads) for lbl in lbls]
    if masks is not None:
        masks[:] = [MyMaxPooling(mask, 1, nthreads=nthreads) for mask in masks]
    if len(clabels[0]) == 0: contexts[:] = [[] for _ in ims]
    for i,clbl in enumerate(clabels): contexts[i].append(clbl[-1])
    contexts[:] = [[MyDownSample(c, 1, nthreads=nthreads) for c in cntxts] for cntxts in contexts]

def __extract_features(submodel, ims, lbls, masks, contexts, nthreads):
    """
    Extract all features from the images into a feature vector. Returns X (feature vector), Y
    (labels), and M (mask or None if no masks).
    """
    from itertools import izip
    from numpy import empty
    from .utils import copy_flat
    ends = list(__cumsum(im.shape[0]*im.shape[1] for im in ims))
    npixs, nfeats = ends[-1], submodel.features
    start = 0
    X = empty((nfeats, npixs))
    Y = empty(npixs, bool)
    for im,lbl,cntxts,end in izip(ims, lbls, contexts, ends):
        submodel.filter(im, cntxts, X[:,start:end].reshape((nfeats,)+im.shape), nthreads=nthreads)
        copy_flat(Y[start:end], lbl, nthreads)
        start = end
    if masks is None: return X, Y, None
    start = 0
    M = empty(npixs, bool)
    for mask,end in izip(masks, ends):
        copy_flat(M[start:end], mask, nthreads)
        start = end
    return X, Y, M

def __cumsum(itr):
    """Like `numpy.cumsum` but takes any iterator and results in an iterator."""
    total = 0
    for x in itr: total += x; yield total
    
def __subsample(X, Y, n=3000000, nthreads=1): # TODO: increase this for real problems
    """
    Sub-sample the data. If the number of pixels is greater than 2*n then at most n rows are kept
    where Y is True and n rows from where Y is False. The rows kept are selected randomly.

    X is the feature vectors, a matrix that is features by pixels.
    Y is the label data, and has a True or False for each pixel.

    Returns the possibly subsampled X and Y.
    """
    # OPT: use nthreads, check and improve speed 
    npixs = len(Y)
    if npixs <= 2*n: return X, Y

    from numpy import zeros, flatnonzero
    from numpy.random import shuffle
    n_trues = Y.sum()
    n_falses = npixs-n_trues
    keep = zeros(npixs, bool)
    if n_trues > n:
        ind = flatnonzero(Y)
        shuffle(ind)
        keep[ind[:n]] = True
        del ind
    else: keep |= Y
    if n_falses > n:
        ind = flatnonzero(~Y)
        shuffle(ind)
        keep[ind[:n]] = True
        del ind
    else: keep |= ~Y
    return X[:,keep], Y[keep]

def __generate_outputs(submodel, X, shapes, nthreads):
    """Generates the outputs of the feature vector X from images that have the given shapes."""
    from os.path import exists, join
    from os import mkdir
    from numpy import save
    start = 0
    folder = __get_output_folder(submodel)
    if not exists(folder): mkdir(folder)
    for i,sh in enumerate(shapes):
        path = join(folder, '%03d.npy'%i)
        end = start + sh[0]*sh[1]
        save(path, submodel.evaluate(X[:,start:end], nthreads).reshape(sh))
        start = end

def __load_clabels(submodel, clabels):
    """
    Loads the clabels for the current submodel into the clabels list. These are loaded as
    memory-mapped arrays from NPY files. The clabels list is a list-of-list with the first index
    being the image index and the second being the level.
    """
    from numpy import load, ndarray
    from os.path import join
    folder = __get_output_folder(submodel)
    for i in xrange(len(clabels)):
        path = join(folder, '%03d.npy'%i)
        # Load as a memory-mapped array viewed as an ndarray
        # (should be using 'r' mode but then Cython typed memoryviews reject it)
        clabels[i].append(load(path, 'r+').view(ndarray)) #pylint: disable=no-member

def __get_output_folder(submodel):
    """Get the folder where outputs are written to for a specific submodel."""
    from os.path import join, dirname
    return join(dirname(submodel.model.path), 'output-%d-%d'%(submodel.stage, submodel.level))

def __chm_train_main():
    """The CHM train command line program"""
    # Parse Arguments
    ims, lbls, model, masks, output, dt, nthreads = __chm_train_main_parse_args()

    # Process input
    out = CHM_train(ims, lbls, model, masks, nthreads)

    # Save output
    if output is not None:
        from pysegtools.images.io import FileImageStack
        FileImageStack.create_cmd(output, [__adjust_output(o,dt) for o in out], True).close()

def __adjust_output(out, dt):
    """Adjusts the output array to match the given data type."""
    from numpy import iinfo, dtype
    if dtype(dt).kind == 'u':
        out *= iinfo(dt).max
        out.round(out=out)
    return out.astype(dt, copy=False)

def __chm_train_main_parse_args():
    """Parse the command line arguments for the CHM train command line program."""
    #pylint: disable=too-many-locals, too-many-branches, too-many-statements
    import os.path
    from sys import argv
    from collections import OrderedDict
    from getopt import getopt, GetoptError
    from pysegtools.images.io import FileImageStack
    from .filters import FilterBank, Haar, HOG, Edge, Gabor, SIFT, Intensity
    from .model import Model

    from numpy import uint8, uint16, uint32, float32, float64
    dt_trans = {'u8':uint8, 'u16':uint16, 'u32':uint32, 'f32':float32, 'f64':float64}

    # Parse and minimally check arguments
    if len(argv) < 3: __chm_train_usage()
    if len(argv) > 3 and argv[3][0] != "-":
        __chm_train_usage("You provided more than 2 required arguments")

    # Open the input and label images
    ims = FileImageStack.open_cmd(argv[1])
    lbls = FileImageStack.open_cmd(argv[2])

    # Get defaults for optional arguments
    path = './temp/'
    nstages, nlevels = 2, 4
    # TODO: Gabor and SIFT should not need compat mode here!
    fltrs = OrderedDict((('haar',Haar()), ('hog',HOG()), ('edge',Edge()), ('gabor',Gabor(True)),
                         ('sift',SIFT(True)), ('intensity-stencil-10',Intensity.Stencil(10))))
    cntxt_fltr = Intensity.Stencil(7)
    masks = None
    output, dt = None, uint8
    restart = False
    nthreads = None

    # Parse the optional arguments
    try: opts, _ = getopt(argv[3:], "rm:S:L:f:c:M:o:d:n:N:")
    except GetoptError as err: __chm_train_usage(err)
    for o, a in opts:
        if o == "-m":
            path = a
            if os.path.exists(path) and not os.path.isdir(path):
                __chm_train_usage("Model folder exists and is not a directory")
        elif o == "-S":
            try: nstages = int(a, 10)
            except ValueError: __chm_train_usage("Number of stages must be an integer >= 2")
            if nstages < 2: __chm_train_usage("Number of stages must be an integer >= 2")
        elif o == "-L":
            try: nlevels = int(a, 10)
            except ValueError: __chm_train_usage("Number of levels must be a positive integer")
            if nlevels <= 0: __chm_train_usage("Number of levels must be a positive integer")
        elif o == "-f":
            if len(a) == 0: __chm_train_usage("Must list at least one filter")
            if a[0] == '+':
                fltrs.update((f,__get_filter(f)) for f in a[1:].lower().split(','))
            elif a[0] == '-':
                for f in a[1:].lower().split(','): fltrs.pop(f)
            else:
                fltrs = OrderedDict((f,__get_filter(f)) for f in a.lower().split(','))
        elif o == "-c":
            cntxt_fltr = __get_filter(a.lower())
        elif o == "-M":
            masks = FileImageStack.open_cmd(a)
        elif o == "-o":
            output = a
            FileImageStack.create_cmd(output, None, True)
        elif o == "-d":
            a = a.lower()
            if a not in dt_trans: __chm_train_usage("Data type must be one of u8, u16, u32, f32, or f64")
            dt = dt_trans[a]
        elif o == "-r": restart = True
        elif o == "-n":
            __chm_train_usage("Tasks not supported during training")
        elif o == "-N":
            try: nthreads = int(a, 10)
            except ValueError: __chm_train_usage("Number of threads must be a positive integer")
            if nthreads <= 0: __chm_train_usage("Number of threads must be a positive integer")
        else: __chm_train_usage("Invalid argument %s" % o)
    if len(fltrs) == 0: __chm_train_usage("Must list at least one filter")
    fltrs = FilterBank(fltrs.values()) #pylint: disable=redefined-variable-type
    model = Model.create(path, nstages, nlevels, fltrs, cntxt_fltr, restart)
    return (ims, lbls, model, masks, output, dt, nthreads)

def __get_filter(f):
    """
    Gets a Filter from a string argument. The string must be one of haar, hog, edge, frangi, gabor,
    sift, or intensity-[square|stencil]-#.
    """
    from .filters import Haar, HOG, Edge, Frangi, Gabor, SIFT, Intensity
    if f.startswith('intensity-'):
        f = f[10:]
        if f.startswith('square-'):    f = f[7:]; F = Intensity.Square
        elif f.startswith('stencil-'): f = f[8:]; F = Intensity.Stencil
        try: f = int(f, 10)
        except ValueError: __chm_train_usage("Size of intensity filter must be a positive integer")
        if f <= 0: __chm_train_usage("Size of intensity filter must be a positive integer")
        return F(f)
    filters = {'haar':Haar, 'hog':HOG, 'edge':Edge, 'frangi':Frangi, 'gabor':Gabor, 'sift': SIFT}
    if f not in filters: __chm_train_usage("Unknown filter '%s'"%f)
    return filters[f]()

def __chm_train_usage(err=None):
    import sys
    if err is not None:
        print(err, file=sys.stderr)
        print()
    from . import __version__
    print("""CHM Image Training Phase.  %s

%s <input> <label> <optional arguments>
  input         The input image(s) to read
                Accepts anything that can be given to `imstack -L` except that
                the value must be quoted so it is a single argument.
  label         The label/ground truth image(s) (0=background)
                Accepts anything that can be given to `imstack -L` except that
                the value must be quoted so it is a single argument.
  The inputs and labels are matched up in the order they are given and paired
  images must be the same size.

Optional Arguments:
  -m model_dir  The folder where the model data is save to. Default is ./temp/.
  -S nstages    The number of stages of training to perform. Must be >=2.
                Default is 2.
  -L nlevels    The number of levels of training to perform. Must be >=1.
                Default is 4.
  -f filters... Comma separated list of filters to use on each image to generate
                the features. Options are haar, hog, edge, gabor, sift, frangi,
                and intensity-<type>-#. For the intensity filter, the <type> can
                be either stencil or square and the # is the size in pixels. If
                the whole argument starts with + then it will add them to the
                defaults. If the argument starts with - then they will be
                removed from the defaults.
                Defaults are: haar,hog,edge,gabor,sift,intensity-stencil-10
  -c filter     The filter used to generate features for context images. This
                takes a single filter listed above.
                Default is intensity-stencil-7
  -M mask       Specify a mask of the input images and labels for which pixels
                should be used during training. The default is to use all
                pixels. The mask can be anything that can be given to
                `imstack -L` except that the value must be quoted so it is a
                single argument. It must contain the same number of images as
                input and label along with each image being the same size. Any
                non-zero pixel in the mask represent a point used for training,
                either a positive or negative label.
  -o ouput      Output the results of testing the model on the input data to the
                given image. Accepts anything that is accepted by `imstack -S`
                in quotes. The data is always calculated anyways so this just
                causes it to be saved in a usable format.
  -d type       Set the output type of the data, one of u8 (default), u16, u32,
                f32, or f64; the output image type must support the data type.
  -r            Restart an aborted training attempt. This will restart just
                after the last completed stage/level. You must give the same
                inputs and labels as before for the model to make sense. However
                you do not need to give the same filters, stages, or levels. The
                model will be adjusted as necessary.
  -N nthreads   How many threads to use. Default is to run as many threads as
                there are CPUs. Note: only the extraction of features and
                generation of outputs can use multiple threads and multiple
                tasks are not supported."""
          % (__version__, __loader__.fullname), file=sys.stderr) #pylint: disable=undefined-variable
    sys.exit(0 if err is None else 1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    __chm_train_main()
