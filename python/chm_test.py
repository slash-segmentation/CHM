"""
CHM Image Testing
Runs CHM testing phase on an image. Can also be run as a command line program with arguments.
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

__version__ = 0.1
__all__ = ["CHM_test", # operates over an entire slice by breaking it into tiles
           "testCHM",  # the base function that operates on a single tile
           "testCHM_max_mem"] # gives the expected amount of memory testCHM will use

def __chm_test_main():
    import sys, os.path
    from getopt import getopt, GetoptError
    from scipy.misc import imread, imsave # TODO: use pysegtools instead (already required to use it for MATLAB file reading, so why not?)

    from numpy import uint8, uint16, uint32, float32, float64, iinfo
    dt_trans = {'u8':uint8, 'u16':uint16, 'u32':uint32, 'f32':float32, 'f64':float64}

    # Parse and minimally check arguments
    if len(sys.argv) < 3: __chm_test_usage()
    if len(sys.argv) > 3 and sys.argv[3][0] != "-":
        __chm_test_usage("You provided more than 2 required arguments. Did you accidently use a glob expression without escaping the asterisk?")
    im_path = sys.argv[1]
    output  = sys.argv[2]
    if os.path.isdir(output): __chm_test_usage("Output file already exists as a directory.")

    # Get defaults for optional arguments
    model_dir = './temp/'
    tile_size = None
    tiles = []
    ntasks = None
    nthreads = None
    hist_eq = False
    dt = uint8

    # Parse the optional arguments
    try: opts, _ = getopt(sys.argv[3:], "hm:t:T:n:N:d:")
    except GetoptError as err: __chm_test_usage(err)
    for o, a in opts:
        if o == "-m":
            model_dir = a
            if not os.path.isdir(model_dir): __chm_test_usage("Model folder is not a directory.")
        elif o == "-t":
            try:
                if 'x' in a: W,H = [int(x,10) for x in a.split('x', 1)]
                else:        W = H = int(a,10)
            except ValueError: __chm_test_usage("tile_size must be a positive integer or two positive integers seperated by an x.")
            if W <= 0 or H <= 0: __chm_test_usage("tile_size must be a positive integer or two positive integers seperated by an x.")
            tile_size = W,H
        elif o == "-T":
            try: C,R = [int(x,10) for x in a.split(',', 1)]
            except ValueError: __chm_test_usage("tile position must be two positive integers seperated by a comma.")
            if C < 0 or R < 0: __chm_test_usage("tile position must be two positive integers seperated by a comma.")
            tiles.append((C,R))
        elif o == "-n":
            try: ntasks = int(a, 10)
            except ValueError: __chm_test_usage("ntasks must be a positive integer.")
            if ntasks <= 0: __chm_test_usage("ntasks must be apositive integer.")
        elif o == "-N":
            try: nthreads = int(a, 10)
            except ValueError: __chm_test_usage("nthreads must be a positive integer.")
            if nthreads <= 0: __chm_test_usage("nthreads must be apositive integer.")
        elif o == "-H": hist_eq  = True
        elif o == "-d":
            a = a.lower()
            if a not in dt_trans: __chm_test_usage("data type must be one of u8, u16, u32, f32, or f64.")
            dt = dt_trans[a]
        else: __chm_test_usage("Invalid argument %s." % o)
    if len(tiles) == 0: tiles = None

    # Process input and save to an output
    im = imread(im_path)
    out = CHM_test(im, model_dir, tile_size, tiles, ntasks, nthreads, hist_eq, True)
    if dt in (uint8, uint16, uint32): out = (out*iinfo(dt).max).round()
    out = out.astype(dt)
    imsave(output, out)

def __chm_test_usage(err=None):
    import sys
    if err is not None:
        print(err, file=sys.stderr)
        print()
    print("""CHM Image Testing Phase.  %s
    
%s <input_files> <output_folder> <optional arguments>
  input_file    The input file to read.
  output_file   The output file to save.
	
Optional Arguments:
  -m model_dir  The folder that contains the model data. Default is ./temp/.
                (contains param.mat and MODEL_level#_stage#.mat)
  -t tile_size  Set the tile size to use as WxH. By default the tile size is the
                the same size as the training images (which is believed to be
                optimal). Old models do not include the size of the training
                images and then 1024x1024 is used if not given.
  -T tile_pos   Specifies that only the given tiles be processed by CHM while
                all others simply output black. Each tile is given as C,R (e.g.
                2,1 would be the tile in the second column and first row). Can
                process multiple tiles by using multiple -T arguments. The tiles
                are defined by multiples of tile_size. A tile position out of
                range will be ignored. If not included then all tiles will be
                processed.
  -H            Histogram-equalize the testing images to the training image
                histogram (if provided in the model). If not used, the testing
                data should already be properly equalized.
  -n ntasks     how many separate tasks to run in parallel
                each task processes a single tile of the image
  -N nthreads   how many threads to use per task
                in general, each additional parallel task takes up a lot of
                memory (up to 7.5 GB for 1000x1000 at Nlevel=4) while each
                additional thread per task does not really increase memory
                usage, however running two tasks in parallel each with 1 thread
                each is faster than giving a single task two threads.
                default is to run twice as many tasks as can fit in memory
                (since the max memory is only used for a short period of time)
                and divide the rest of the CPUs among the tasks
                if only one value is given the other is calculated using it
  -d type       set the output type of the data, one of u8 (default), u16, u32,
                f32, or f64"""
          % (__version__, sys.argv[0]), file=sys.stderr)
    sys.exit(1)

def CHM_test(im, modelpath="./temp/", tilesize=None, tiles=None, ntasks=None, nthreads=None, hist_eq=False, ignore_bad_tiles=False):
    """
    CHM_test - CHM Image Testing
    Breaks an image into multiple tiles, runs testCHM on each tile, and combines the results all
    back into a single image which it returns. This is optimized for parallelization of a single
    slice by loading the image and various other items into shared memory and spawning subprocesses
    that each use the shared resources but work on their own tiles.
    
    im is a single image slice - the entire slice as a numpy array
    modelpath is the path to the model folder which contains param.mat and MODEL_level%d_stage%d.mat
        default is "temp"
    tilesize is the size of the tiles to process, either a:
            * a single integer for both width and height
            * a tuple of integers for width and height
        default is either the training image size or 1024x1024 if that is not available
        if this is a multiple of 2^Nlevel then there will be a slight gain in speed and reduction in
        memory (TODO: verify this assertion, it may be completely FALSE)
    tiles is a list of tile coordinates to process
        coordinates are in x,y format
        they go from 0,0 to ((im.shape[1]-1) // tilesize[0], (im.shape[0]-1) // tilesize[1])
        default is all tiles
        if not processing all tiles, the resulting image will be black wherever a tile was skipped
    ntasks is how many separate tasks to run in parallel, each processing a single tile of the image
    nthreads is how many threads to use per task
        in general, each additional parallel task takes up a lot of memory (up to 7.5 GB for
        1000x1000 at Nlevel=4) while each additional CPU per task does not really increase memory
        usage, however running two tasks in parallel each with 1 CPU is faster than giving a single
        task two CPUs.
        default is to run twice as many tasks as can fit in memory (since the max memory is only
        used for a short period of time) and divide the rest of the CPUs among the tasks
        if only one is given, the other is calculated using its value
    hist_eq is whether the slice should be exact histogram equalized to the training data set
        default is False since this is a time intesive process and is not parallelized (make sure to
        pre-process the data!)
        can also set to 'approx' which is decent and much faster than the exact histogram
        equalization used, but it is still not parallelized (even though it could be...)
    ignore_bad_tiles is whether invalid tiles in the tiles argument cause an exception or are
        silently ignored, default is to throw an exception
    """
    import os.path, time
    from collections import Sequence
    from psutil import cpu_count, virtual_memory
    from ctypes import c_double, memmove
    from multiprocessing import Process, Value, Queue
    from multiprocessing.sharedctypes import RawArray
    from numpy import float64, intp, iinfo, frombuffer, indices, array
    from pysegtools.general.matlab import openmat, mat_nice
    from chm_utils import MyDownSample
    from warnings import warn
    
    # Get params
    modelpath = os.path.abspath(modelpath)
    params = openmat(os.path.join(modelpath, 'param.mat'), 'r')
    Nlevel, Nstage = int(params['Nlevel'].data[0]), int(params['Nstage'].data[0])
    
    # Parse tile size
    if tilesize is None:
        if 'TrainingSize' not in params:
            warn('tilesize was not given and model does not contain the training image size, using 1024x1024')
            tilesize = (1024, 1024)
        else:
            tilesize = tuple(int(x) for x in params['TrainingSize'].data.ravel()) # TODO: is this reversed?
    if isinstance(tilesize, Sequence) and len(tilesize) == 2:
        tilesize = (int(tilesize[1]), int(tilesize[0]))
    else:
        tilesize = (int(tilesize), int(tilesize))
    tiles_aligned = all(sz & ((1<<Nlevel)-1) == 0 for sz in tilesize)
        
    # Deal with hist eq
    if hist_eq:
        if 'hgram' in params:
            from pysegtools.images.filters.hist import histeq, histeq_exact
            im = (histeq if hist_eq == 'approx' else histeq_exact)(im, params['hgram'].data)
        else:
            warn('training data histogram not included in model, make sure you manually perform histogram equalization on the testing data.')

    # Figure out which tiles to process
    max_tile_x, max_tile_y = (im.shape[1]+tilesize[1]-1)//tilesize[1], (im.shape[0]+tilesize[0]-1)//tilesize[0]
    if tiles is None:
        tiles = indices((max_tile_x, max_tile_y)).T.reshape((-1,2)) # TODO: are x and y swapped?
    else:
        tiles = array(tiles, dtype=intp)
        if tiles.ndim != 2 or tiles.shape[1] != 2: raise ValueError('Invalid tile coordinates shape')
        if ignore_bad_tiles:
            tiles = tiles[(tiles >= 0).all(axis=1) & ((tiles[:,0] <= max_tile_x) & (tiles[:,1] <= max_tile_y))]
        elif (tiles < 0).any() or (tiles[:,0] > max_tile_x).any() or (tiles[:,1] > max_tile_y).any(): # TODO: are x and y swapped?
            raise ValueError('Invalid tile coordinates')
     
    # Get ntasks and nthreads
    ncpus, ntiles = cpu_count(True), len(tiles) # we use the logical number of CPUs, not physical
    if ntasks is None and nthreads is None:
        chm_max_mem = testCHM_max_mem(tilesize, Nlevel) // 2
        ntasks = max(min(virtual_memory().available // chm_max_mem, ntiles, ncpus), 1)
        nthreads = ncpus // ntasks
    elif ntasks is None:
        chm_max_mem = testCHM_max_mem(tilesize, Nlevel) // 2
        nthreads = int(nthreads)
        ntasks = max(min(virtual_memory().available // chm_max_mem, ntiles, ncpus // nthreads), 1)
    elif nthreads is None:
        ntasks = max(min(int(ntasks), ntiles), 1)
        nthreads = ncpus // ntasks
    else:
        ntasks = int(ntasks)
        nthreads = int(nthreads)
        
    # Create a shared memory image (as floating-point)
    # OPT: if only a select few tiles are chosen, only share those in memory (or at least that bounding box)
    sh, sz, dt = im.shape, im.size, im.dtype
    im_shared_mem = RawArray(c_double, sz)
    im_shared = frombuffer(im_shared_mem, dtype=float64).reshape(sh)
    im_shared[:,:] = im[:,:]
    if dt.kind == 'u':
        im_shared *= 1.0 / iinfo(dt.type).max
    elif dt.kind == 'i':
        ii = iinfo(dt.type)
        im_shared -= ii.min
        im_shared *= 1.0 / (ii.max - ii.min)
    elif dt.kind not in 'fb': raise ValueError('Unknown image format')
    out_shared_mem = RawArray(c_double, sz)
    #memset(out_shared_mem, 0, im.nbytes) # already 0s
    out = frombuffer(out_shared_mem, dtype=float64).reshape(sh)

    # Downscale image
    # OPT: if only a select few tiles are chosen, only downsample the necessary region
    # TODO: it is possible that this actually degrades performance and memory usage - always do the "tiles unaligned" version?
    if tiles_aligned:
        im_shared_mems = [im_shared_mem] + [None] * Nlevel
        for level in xrange(1, Nlevel+1):
            # OPT: have MyDownSample use output memory - is it possible?
            im = MyDownSample(im, 1)
            im_shared_mems[level] = im_shared_mem = RawArray(c_double, im.size)
            memmove(im_shared_mem, (c_double*im.size).from_buffer(im.data), im.nbytes)
    else:
        im_shared_mems = im_shared_mem
    del im_shared, im_shared_mem, im
    
    # Load all the models
    def load_model(stage, level):
        with openmat(os.path.join(modelpath, 'MODEL_level%d_stage%d.mat' % (level, stage)), 'r') as mat:
            model = mat_nice(mat['model'])
        return __conv_to_shared(model['discriminants']), model['discriminants'].shape, \
               int(model['nGroup']), int(model['nDiscriminantPerGroup'])
    # create a list of lists, models[stage][level] gives a tuple of discriminants (array), shape of discs, nGroup, nDiscriminantPerGroup
    models = [(([load_model(stage, level) for level in xrange(Nlevel+1)]) if stage != Nstage else [load_model(stage, 0)])
              for stage in xrange(1, Nstage+1)]

    # Process tiles
    print("Running CHM test with %d task%s and %d thread%s per task" %
          (ntasks, 's' if ntasks > 1 else '', nthreads, 's' if nthreads > 1 else ''))
    init_args = im_shared_mems, out_shared_mem, sh, models, tilesize, nthreads
    if ntasks == 1:
        CHM_test_tile = __chm_test_proc_init(*init_args)
        for tile in tiles: CHM_test_tile(tile)
    else:
        # Make sure the Cython modules are compiled (if we wait until they automatically do it the
        # files can be corrupted by multiple processes writing them at once)
        import __chm_filters; dir(__chm_filters)
        import __imresize;    dir(__imresize)

        # Make a queue of all of the tiles
        tiles_queue = Queue()
        for tile in tiles: tiles_queue.put_nowait(tile)

        # Setup the arguments
        args = (init_args, tiles_queue, Value('L', 0), len(tiles), time.clock())

        # Start ntasks processes and wait for them all to stop
        processes = [Process(target=__run_chm_test_proc, name="CHM-test-%d"%i, args=args) for i in xrange(ntasks)]
        for p in processes: p.daemon = True; p.start()
        for p in processes: p.join()

        # Check that if we actually finished
        if not tiles_queue.empty():
            raise Exception('There was an issue in the underlying processes and the tiles were not completed')
        
    return out

def __run_chm_test_proc(init_args, tiles_queue, completed_tiles, total_tiles, start_time):
    """
    Runs a single CHM test sub-process. init_args are the arguments to __chm_test_proc_init,
    tiles_queue is a multiprocessing Queue of all the tiles to compute (as tuples of x,y),
    completed_tiles is a multiprocess Value for the number of tiles completed (initially 0),
    total_tiles is an int with the total number of tiles, and start_time is a float of seconds
    according to clock().
    """
    from Queue import Empty
    CHM_test_tile = __chm_test_proc_init(*init_args)
    try:
        while not tiles_queue.empty():
            tile = tiles_queue.get_nowait()
            CHM_test_tile(tile)
            __chm_test_progress(tile, completed_tiles, total_tiles, start_time)
    except Empty: pass # all items processed - done!
    except: # other exceptions put the tile back on the queue and re-raise the error
        tiles_queue.put_nowait(tile)
        raise

def __chm_test_progress(tile, completed_tiles, total_tiles, start_time):
    """
    Display a progress update message for CHM test running. See __run_chm_test_proc for definitions
    of completed_tiles, total_tiles, start_time.
    """
    import time
    with completed_tiles.get_lock(): completed_tiles.value += 1
    ntiles = completed_tiles.value
    perc = ntiles/total_tiles
    elapsed = time.clock() - start_time
    secs = elapsed/perc-elapsed
    str_time = "%d:%02d:%02d" % (secs // 3600, (secs % 3600) // 60, round(secs % 60))
    print("Completed %d,%d    Overall %3d%%    Est Time Rem %s" % (tile[0], tile[1], int(round(perc*100)), str_time))

def __chm_test_proc_init(im_shared_mems, out_shared_mem, shape, models, tilesize, nthreads):
    """
    Initialize a CHM test sub-process. im_shared_mems is a shared RawArray of the image,
    out_shared_mem is a shared RawArray of the output image, shape is the shape of the image and
    the output, models is a list of list of tuples of models using shared memory, tilesize is the
    size of each time (as height by width), and nthreads is the number of threads to use in this
    process.

    This returns a method for testing a single tile of the image.
    """
    from itertools import izip
    from numpy import copyto

    Nstage, Nlevel = len(models), len(models[0]) - 1
    
    # Get the images, output, and models in shared memory as numpy arrays
    tiles_aligned = isinstance(im_shared_mems, list)
    if tiles_aligned: # tiles aligned, images are already downsampled
        shapes = [shape]
        for _ in xrange(1, Nlevel+1):
            shapes.append(((shapes[-1][0] + 1) // 2, (shapes[-1][1] + 1) // 2))
        ims = [__conv_from_shared(im, sh) for im,sh in izip(im_shared_mems, shapes)]
    else:
        ims = __conv_from_shared(im_shared_mems, shape)
    out = __conv_from_shared(out_shared_mem, shape, False)
    # a list of lists, models[stage][level] gives a tuple of discriminants (array), nGroup, nDiscriminantPerGroup
    models = [[(__conv_from_shared(d, sh), ng, ndpg) for d, sh, ng, ndpg in stg] for stg in models]

    # Create and return the function for testing individual tiles using all the paramters we have
    TH,TW = tilesize
    if tiles_aligned: # tiles aligned, images are already downsampled
        IH,IW = ims[0].shape
        def __chm_test_tile(tile):
            x,y = tile
            L,T = x*TW, y*TH
            R,B = min(L+TW, IW), min(T+TH, IH)
            regions = [None]*(Nlevel+1)
            last = (t,l,b,r) = (T,L,B,R)
            regions[0] = last = (T,L,B,R)
            for level in xrange(Nlevel+1):
                # TODO: calculate these right
                t = t//2
                l = l//2
                regions[level] = last = (t,l,b,r)
            copyto(out[T:B,L:R], testCHM(ims, models, Nlevel, Nstage, regions, nthreads))
    else:
        IH,IW = ims.shape
        def __chm_test_tile(tile):
            x,y = tile
            L,T = x*TW, y*TH
            R,B = min(L+TW, IW), min(T+TH, IH)
            copyto(out[T:B,L:R], testCHM(ims, models, Nlevel, Nstage, (T,L,B,R), nthreads))
    return __chm_test_tile

def __conv_to_shared(arr):
    """
    Convert a NumPy array of float32 or float64 to a shared memory array of c_float or c_double.
    Copies the data.
    """
    from ctypes import c_float, c_double, memmove
    from multiprocessing.sharedctypes import RawArray
    c_type = c_double if arr.itemsize == 8 else c_float
    shrd = RawArray(c_type, arr.size)
    memmove(shrd, (c_type*arr.size).from_buffer(arr.data), arr.nbytes)
    return shrd

def __conv_from_shared(shrd, shape, readonly=True):
    """
    The reverse of __conv_to_shared, converting a shared memory block to a NumPy array. The type
    must be c_float or c_double. The shape must be provided as that information is lost. By default
    it makes the new array read-only. The data is NOT copied.
    """
    #pylint: disable=protected-access
    from ctypes import c_double
    from numpy import frombuffer, float32, float64
    arr = frombuffer(shrd, dtype=(float64 if shrd._type_ is c_double else float32)).reshape(shape)
    if readonly: arr.flags.writeable = False
    return arr

def testCHM_max_mem(tilesize, Nlevel):
    """
    Gets the theoretical maximum memory usage for CHM-test for a given tile size and number of
    levels, plus a small fudge factor that is larger than any additional random overhead that might
    be needed.

        tilesize - can be the number of pixels in a single tile, or a tuple of width and height
        Nlevel - the number of levels

    This is caclulated using the following formula:
        (476+57*(Nlevel+1))*8*tilesize + 200*8*tilesize + 20*8*tilesize
    Where 476 is the number of filter features, 57 is the number of extra filter features generated
    at each level, 8 is the size of a double-precision floating point, 200 is ???, and 20 is ???.
    (I forgot how I got those last two values, but they are important).

    Theoretical maximum memory usage is 7.31 GB (for 1000x1000 tiles and 4 levels). In practice I am
    seeing this +0.03 GB which is not too much overhead, probably estimating 100 MB overhead would
    be good enough in all cases.
    """
    from chm_filters import TotalFilterFeatures
    try: tilesize = tilesize[0]*tilesize[1]
    except TypeError: pass
    return tilesize*(8*(TotalFilterFeatures()+220+57*(Nlevel+1)) + 100) # the +100 is a fudge-factor

def testCHM(im, models, Nlevel, Nstage, region=None, nthreads=1):
    # CHANGED: dropped NFeatureContexts argument - this is now calculated
    # CHANGED: besides taking just a single im and processing it, this now accepts several different
    # inputs for optimization:
    #    single im and no region - original behavior - processes entire image, padding as necessary
    #    single im and a single region (T,L,B,R) - uses the specified region of the image, padding can take pixels from the image
    #    list of images and regions - lists have one image and region for each level (+1)
    #          each image-region should be a downscale by half of the previous image-region
    # CHANGED: the 'savingpath' can now either be the path of the folder containing the models or it
    # can be the models themselves already loaded, such that models[stage][level] gives a tuple of:
    # (discriminants (array), nGroup, nDiscriminantPerGroup)
    from itertools import izip
    from numpy import empty, float32
    import gc
    from chm_utils import MyDownSample, MyUpSample, get_image_region
    from chm_filters import Filterbank, TotalFilterFeatures, ConstructNeighborhoods, StencilNeighborhood

    # Process the image and region arguments
    # After this we will have a list of images and regions (one for each level)
    if isinstance(im, list):
        # We have a list of images and regions - easy to do
        ims = im
        regions = region
        if len(ims) != Nlevel + 1 or len(regions) != Nlevel + 1: raise ValueError()
        # We will assume that the regions specify decreasing by half regions of the image and the
        # images are downscales
    
    elif region is not None:
        from chm_filters import MaxFilterPadding
        max_pad = MaxFilterPadding()
        T, L, B, R = region #pylint: disable=unpacking-non-sequence
        H, W = B - T, R - L
        ims = [None]*(Nlevel+1)
        regions = [None]*(Nlevel+1)

        # First level is easiest
        ims[0], regions[0] = get_image_region(im, max_pad, region)

        # At the highest level, max_pad*(2^Nlevel) pixels away in the original image become part of
        # the padding. For max_pad=18 and Nlevel=4 that is 288 pixels away!
        pad = (1<<Nlevel)*max_pad
        im, _ = get_image_region(im, pad, region)
        for level in xrange(1, Nlevel+1):
            im = MyDownSample(im, 1)
            pad //= 2; H, W = (H+1)//2, (W+1)//2
            ims[level], regions[level] = get_image_region(im, max_pad, (pad, pad, pad + H, pad + W))
        
        #del im_full, max_pad_full, region_full
        # TODO: All this extra padding and copies of images leads to about +6.3MB additional memory
        # usage over just the original padded image (for 1000x1000 tiles and Nlevel=4). If the
        # levels were copied, some additional memory could be saved (3.2MB, and may result in
        # faster-to-access images at the expense of an additional one-time copy operation).

    else:
        # All regions are None (the padded region will be calculated in Filterbank)
        # All images are simple down samples of the previous level
        ims = [None]*(Nlevel+1)
        regions = [None]*(Nlevel+1)
        ims[0] = im
        for level in xrange(1, Nlevel+1):
            ims[level] = MyDownSample(ims[level-1], 1)

    # Cleanup these references since we now have ims and regions
    del im, region
    
    clabels = [None]*(Nlevel+1)

    feature_contexts = StencilNeighborhood(7)
    n_feature_contexts = feature_contexts.shape[1]
    n_filter_features = TotalFilterFeatures()
    models_already_loaded = isinstance(models, list)
    if not models_already_loaded:
        from os.path import join
        from pysegtools.general.matlab import openmat, mat_nice

    for stage in xrange(1, Nstage+1):
        for level,(im,region) in enumerate(izip(ims, regions)):
            #pylint: disable=cell-var-from-loop
            
            # Get the output image shape for this level
            sh = (region[2]-region[0], region[3]-region[1])

            # Get how many additional context features we will be adding and how to calculate them
            if stage == 1 or level != 0:
                n_contexts = level
                context_calc = lambda j: MyDownSample(clabels[j], level-j)
            else: # stage != 1 and level == 0
                n_contexts = Nlevel + 1
                context_calc = lambda j: MyUpSample(clabels[j], j)[:sh[0], :sh[1]]
            
            # Load the model
            if models_already_loaded:
                discriminants, nGroup, nDiscriminantPerGroup = models[stage-1][level]
            else:
                with openmat(join(models, 'MODEL_level%d_stage%d.mat' % (level, stage)), 'r') as mat:
                    model = mat_nice(mat['model'])
                discriminants = model['discriminants']
                nGroup = int(model['nGroup'])
                nDiscriminantPerGroup = int(model['nDiscriminantPerGroup'])
                del mat, model
            disc_not_sb = discriminants.dtype != float32
            
            # Create the feature vector
            X = empty((n_filter_features + n_feature_contexts*n_contexts + (1 if disc_not_sb else 0),) + sh)
            if disc_not_sb: X[-1, ...] = 1

            # Calculate filter features
            Filterbank(im, out=X[:n_filter_features], region=region, nthreads=nthreads)
            gc.collect()

            start = n_filter_features
            for j in xrange(n_contexts):
                # OPT: remove padding here
                ConstructNeighborhoods(context_calc(j), feature_contexts, out=X[start:start+n_feature_contexts, :, :])
                start += n_feature_contexts
            if stage != 1 and level == 0:
                # we can free a lot of things at this point
                clabels = [None]*(Nlevel+1)
                if stage == Nstage: ims = None
                
            gc.collect() # most memory intensive part is coming up, make sure we are ready

            X = X.reshape((X.shape[0], -1))
            clabels[level] = EvaluateAndOrNetMX(X, discriminants, nGroup, nDiscriminantPerGroup).reshape(sh) #, order='F')
            del X, discriminants
            
            #from numpy import savez_compressed
            #savez_compressed('out_%d_stage%d.npz' % (level, stage), X=X.reshape(), Y=clabels[level])
            
            if stage == Nstage: return clabels[0]

def EvaluateAndOrNetMX(X, discriminants, nGroup, nDiscriminantPerGroup):
    from numpy import float32, float64
    if discriminants.dtype == float32:
        return genOutput_SB(X, discriminants, nGroup, nDiscriminantPerGroup).astype(float64, copy=False)
    else:
        return genOutput(X, discriminants, nGroup, nDiscriminantPerGroup)

#@profile
def genOutput(x, discriminants, nGroup, nDiscriminantPerGroup):
    # CHANGED: no longer in "native" code, instead pure Python
    # The only difference between this one and genOutput_SB is that this one handles the extra
    # discriminants row differently and takes square roots at various times. 
    from numpy import prod, exp, divide, subtract, negative, sqrt
    # nGroup == nORs, nDiscriminantPerGroup == nANDs
    #npixels = x.shape[1]

    tt = discriminants.T.dot(x)
    del x, discriminants
    # no extras being added?
    negative(tt, out=tt)
    exp(tt, out=tt)
    tt += 1.0
    divide(1.0, tt, out=tt)
    tg = prod(tt.reshape((nDiscriminantPerGroup, nGroup, -1), order='F'), axis=0)
    del tt
    sqrt(tg, out=tg) # not in genOutput_SB
    subtract(1.0, tg, out=tg)
    to = prod(tg, axis=0)
    del tg
    sqrt(to, out=to) # not in genOutput_SB
    return subtract(1.0, to, out=to)

#@profile
def genOutput_SB(x, discriminants, nGroup, nDiscriminantPerGroup):
    # CHANGED: no longer in "native" code, instead pure Python (my Cython attempt was 100x slower than this)
    # CHANGED: returns float64 instead of float32 array
    # CHANGED: x and discriminants no longer required to be float32 matrices (but works just fine when they are)
    from numpy import prod, exp, divide, subtract, negative
    #nfeats, npixels = x.shape
    #nfeats_p1, ndisc = discriminants.shape
    nfeats = x.shape[0]
    
    tt = discriminants[:nfeats,:].T.dot(x)
    tt += discriminants[-1,:,None]
    del x, discriminants
    negative(tt, out=tt)
    exp(tt, out=tt)
    tt += 1.0
    divide(1.0, tt, out=tt)
    tg = prod(tt.reshape((nDiscriminantPerGroup, nGroup, -1), order='F'), axis=0)
    del tt
    subtract(1.0, tg, out=tg)
    to = prod(tg, axis=0)
    del tg
    return subtract(1.0, to, out=to)

if __name__ == "__main__":
    __chm_test_main()
