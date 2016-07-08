#!/usr/bin/env python2
"""
CHM Image Testing
Runs CHM testing phase on an image. Can also be run as a command line program with arguments.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

__all__ = ["CHM_test", # operates over an entire image by breaking it into tiles
           "testCHM",  # operates on an entire image
           "CHM_test_max_mem"] # gives the expected amount of memory testCHM will use

def CHM_test(im, model="./temp/", tilesize=None, tiles=None, ntasks=None, nthreads=None, hist_eq=False, ignore_bad_tiles=False):
    """
    CHM_test - CHM Image Testing
    Breaks an image into multiple tiles, runs testCHM on each tile, and combines the results all
    back into a single image which it returns. This is optimized for parallelization of a single
    image by loading the image and various other items into shared memory and spawning subprocesses
    that each use the shared resources but work on their own tiles.
    
    im is a single image slice - the entire image as a numpy array, it may also be given as a
        pysegtools.image.ImageSource
    model is the path to the model folder which contains the model, default is "temp"
    tilesize is the size of the tiles to process, either a:
            * a single integer for both width and height
            * a tuple of integers for width and height
        default is either the training image size or 1024x1024 if that is not available
        if this is a multiple of 2^Nlevel then there may be some gains in speed/memory in certain
        circumstances
    tiles is a list of tile coordinates to process
        coordinates are in x,y format
        they go from 0,0 to ((im.shape[1]-1) // tilesize[0], (im.shape[0]-1) // tilesize[1])
        default is all tiles
        if not processing all tiles, the resulting image will be black wherever a tile was skipped
    ntasks is how many separate tasks to run in parallel, each processing a single tile of the image
    nthreads is how many threads to use per task
        in general, each additional parallel task takes up a lot of memory (up to 7.5 GB for
        1000x1000 with Nlevel=4) while each additional CPU per task does not really increase memory
        usage, however running two tasks in parallel each with 1 CPU is faster than giving a single
        task two CPUs.
        default is to run twice as many tasks as can fit in memory (since the max memory is only
        used for a short period of time) and divide the rest of the CPUs among the tasks
        if only one is given, the other is calculated using its value
    hist_eq is whether the image should be exact histogram equalized to the training data set
        default is False since this is a time intesive process and is not parallelized (make sure to
        pre-process the data!)
        can also set to 'approx' which is decent and much faster than the exact histogram
        equalization used, but it is still not parallelized (even though it could be...)
    ignore_bad_tiles is whether invalid tiles in the tiles argument cause an exception or are
        silently ignored, default is to throw an exception
    """
    from ctypes import c_uint8, c_double, memmove
    from multiprocessing import RawArray
    from numpy import float64, uint8, frombuffer, minimum, hstack, array, copyto
    from .utils import im2double

    im, model, tilesize, tiles, ntasks, nthreads = \
        __parse_args(im, model, tilesize, tiles, ntasks, nthreads, hist_eq, ignore_bad_tiles)
    nthreads_full = ntasks*nthreads

    # Move the image into shared memory (as floating-point)
    sh, sz = im.shape, im.size

    # TODO: implement cropping
    # Need to calculate the right values for full_sz, full_sh, full_rgn, regions, work_sizes, ...
    full_sz, full_sh, full_rgn = sz, sh, ((0,0) + sh)
    # Produces 'regions'
    sh_arr = array(sh)
    tilesize = array(tilesize)
    regions = [None]*(model.nlevels+1)
    regions[0] = hstack((tiles*tilesize, minimum((tiles+1)*tilesize, sh_arr)))
    for level in xrange(1, model.nlevels+1):
        sh_arr = (sh_arr+1)//2
        if (tilesize%2).any(): tiles = __rm_dup_tiles(tiles // 2)
        else:                  tilesize = tilesize // 2
        regions[level] = hstack((tiles*tilesize, minimum((tiles+1)*tilesize, sh_arr)))
    
    im_sm = RawArray(c_uint8, full_sz) # TODO: or double
    im_shared = frombuffer(im_sm, uint8).reshape(full_sh) # TODO: or float64
    # TODO: im2double(im, im_shared, full_rgn, nthreads_full)
    # TODO: use utils.copy
    copyto(im_shared, im.astype(uint8, copy=False))
    del im, im_shared

    ##if I have full_rgn, then:
    #full_sh = full_rgn[2] - full_rgn[0], full_rgn[3] - full_rgn[1]
    #full_sz = full_sh[0] * full_sh[1]
    
    #for a 25k by 25k image:
    #   im_sm requires 4.657 GiB
    #   ds_sm requires 4.657 GiB
    #   out_sms requires 6.203 GiB (4.657 + 1.164 + 0.291 + 0.073 + 0.018)
    #   TOTAL: 15.516 GiB (4.657 + 4.657 + 6.203)

    # Calculate all of the downsampled shapes and sizes
    shapes = __gen_shapes(sh, model.nlevels)
    sizes = [sh[0]*sh[1] for sh in shapes]
    
    # Allocate shared memory for the output/clabels, downsampled images, and contexts of all sizes
    # The ds_sm is used for several purposes
    out_sms = [RawArray(c_double, sz) for sz in sizes]
    ds_sm   = RawArray(c_double, full_sz)

    mems = (im_sm, out_sms, ds_sm, shapes[0])
    return __run_chm_test_procs(mems, model, regions, ntasks, nthreads)

##def __get_all_regions(model, shape, tiles, tilesize):
##    from numpy import asarray, empty, hstack, minimum, maximum, intp
##
##    tilesize = asarray(tilesize)
##    shape = asarray(shape)
##    nstg, nlvl1 = model.nstages, model.nlevels+1
##
##    # Get the total amount of padding we need at level 0 so that we have enough for all levels
##    padding = [max(model[stg,lvl].filter.padding for stage in xrange(nstg + (lvl != 0)))
##               for lvl in xrange(nlvl1)]
##    maxpad = max(pad<<lvl for lvl,pad in enumerate(padding))
##
##    # Shift all tiles so that 0,0 would be in the exact top-left corner
##    tiles_off = tiles.min(axis=0)
##    tiles -= tiles_off
##    tiles_shape = tiles.max(axis=0)
##
##    # Get the area we are working with and the area of that plus padding
##    work_off   = tiles_off*tilesize
##    work_shape = (tiles_shape+1)*tilesize
##    work_rgn   = hstack((work_off, work_off+work_shape))
##    full_off   = maximum(work_off - maxpad, 0)
##    full_rgn   = hstack((full_off, minimum(work_rgn[2:]+max_pad, shape)))
##    full_shape = full_rgn[2:] - full_rgn[:2]
##
##    regions = [None]*nlvl1
##    work_regions = empty((nlvl1, 4), intp)
##    work_shapes  = empty((nlvl1, 4), intp)
##    full_regions = empty((nlvl1, 4), intp)
##    full_shapes  = empty((nlvl1, 4), intp)
##
##    work_regions[0] = work_rgn
##    work_shapes[0]  = work_shape
##    full_regions[0] = full_rgn
##    full_shapes[0]  = full_shape
##
##    for level in xrange(1, nlvl1):
##        # Reduce work_shape, padding, tilesize
##        shape = (shape + 1) // 2
##        work_shapes[level] = work_shape = (work_shape + 1) // 2
##        maxpad //= 2
##        if (tilesize%2).any(): tiles = __rm_dup_tiles(tiles // 2)
##        else:                  tilesize = tilesize // 2
##
##
##
##    shapes  = empty((nlvl1, 2), intp)
##    full_extents = empty((nlvl1, 4), intp)
##    work_extents = empty((nlvl1, 4), intp)
##    full_shapes  = empty((nlvl1, 2), intp)
##    work_shapes  = empty((nlvl1, 2), intp)
##
##    rTL, rBR = tiles*tilesize, minimum((tiles+1)*tilesize, shape)
##    regions[0] = hstack((rTL, rBR))
##    
##    rTL, rBR = rTL.min(axis=0), rBR.max(axis=0)
##    work_extents[0] = hstack((rTL, rBR))
##    work_shapes[0] = work_shape = rBR - rTL
##
##    rTL, rBR = maximum(rTL - maxpad, 0), minimum(rBR + maxpad, shape)
##    full_extents[0] = hstack((rTL, rBR))
##    full_shapes[0] = full_shape = rBR - rTL
##    
##    for level in xrange(1, nl1):
##        # Reduce work_shape, padding, tilesize
##        work_shapes[level] = work_shape = (work_shape + 1) // 2
##        maxpad //= 2
##        if (tilesize%2).any(): tiles = __rm_dup_tiles(tiles // 2)
##        else:                  tilesize = tilesize // 2
##        
##        rTL, rBR = tiles*tilesize, minimum((tiles+1)*tilesize, ?????)
##        regions[0] = hstack((rTL, rBR))
##        
##        rTL, rBR = rTL.min(axis=0), rBR.max(axis=0)
##        #work_extents[0] = hstack((rTL, rBR))
##        work_shapes[0] = work_shape = rBR - rTL
##
##        rTL, rBR = maximum(rTL - maxpad, 0), minimum(rBR + maxpad, shape)
##        #full_extents[0] = hstack((rTL, rBR))
##        full_shapes[0] = full_shape = rBR - rTL
##
##
##        
##        work_shapes[level] = work_shape = (work_shape+1)//2
##        
##        rTL, rBR = tiles*tilesize, minimum((tiles+1)*tilesize, sh)
##        regions[level] = hstack((rTL, rBR))
##
##    return shapes, regions, extents, padded
##
##
##def __get_xxxxx(model, shape, tiles, tilesize):
##    from numpy import minimum, maximum, array
##
##    shapes, regions, extents, padded = __get_all_regions(model, shape, tiles, tilesize)
##
##    scales = [1<<level for level in xrange(model.nlevels+1)]
##    padded_scaled = (padded.T * scales).T
##    padded_scaled = hstack((padded_scaled[:2].min(axis=0), padded_scaled[2:].max(axis=0)))
##
##
##    regions = [region - padded_scaled[:2] for region in regions]
##    
##    work_shapes = [extent[2:] - extent[:2] for extent in extents]
##
##    
##    full_shapes = [padded[2:] - padded[:2] for padded in padded]
##
##    H,W = work_sh = BR - TL
##    work_sz = H * W
##    work_rgn = tuple(TL) + tuple(BR)
##
##    fH,fW = full_sh = BR_pad - TL_pad
##    full_sz = fH * fW
##    full_rgn = tuple(TL_pad) + tuple(BR_pad)
##
##    return regions


def __parse_args(im, model="./temp/", tilesize=None, tiles=None, ntasks=None, nthreads=None, hist_eq=False, ignore_bad_tiles=False):
    """
    Parse the arguments of the CHM_test function.

    im is returned as-is, unwrapped from an ImageSource, and/or histogram equalized.
    The model is loaded and returned as a Model object.
    tilesize is returned as a tuple that is in H,W order (even though it is given in W,H order).
    tiles is returned as a Nx2 array in y,x order and sorted (even though it is given in x,y order).
    ntasks and nthreads are adjusted as necessary.
    """
    from collections import Sequence
    from pysegtools.images import ImageSource
    from .model import Model

    # Make sure the image is not an ImageSource
    if isinstance(im, ImageSource): im = im.data
    
    # Load the model
    model = Model.load(model)
    
    # Parse tile size
    if tilesize is None:
        tilesize = tuple(int(x) for x in model['TrainingSize'].ravel()) \
                   if 'TrainingSize' in model else (1024, 1024)
    elif isinstance(tilesize, Sequence) and len(tilesize) == 2:
        tilesize = (int(tilesize[1]), int(tilesize[0]))
    else:
        tilesize = (int(tilesize), int(tilesize))
        
    # Deal with hist eq
    if hist_eq:
        if 'hgram' in model:
            from pysegtools.images.filters.hist import histeq, histeq_exact
            im = (histeq if hist_eq == 'approx' else histeq_exact)(im, model['hgram'])
        else:
            from warnings import warn
            warn('training data histogram not included in model, make sure you manually perform histogram equalization on the testing data.')

    # Get the list of tiles to process
    tiles = __get_tiles(im, tilesize, tiles, ignore_bad_tiles)
    
    # Get ntasks and nthreads
    ntasks, nthreads = __get_ntasks_and_nthreads(ntasks, nthreads, model, tilesize, len(tiles))

    return im, model, tilesize, tiles, ntasks, nthreads

def __get_tiles(im, tilesize, tiles=None, ignore_bad_tiles=False):
    """
    Get the tile coordinates from the CHM_test parameters. Always returns a Nx2 array of
    coordinates, all of which are valid for the image and tilesize and no duplicates. While the
    input tiles are specified as x,y, this function returns them in y,x. The tilesize argument is
    given in H,W order. Also, the tiles will be sorted by y then by x.
    """
    from numpy import array, indices, intp
    from itertools import izip
    max_y, max_x = ((x+y-1)//y for x,y in izip(im.shape, tilesize))
    if tiles is None: return __sort_tiles(indices((max_y, max_x)).T.reshape((-1,2)))
    tiles = array(tiles, dtype=intp)
    if tiles.ndim != 2 or tiles.shape[1] != 2: raise ValueError('Invalid tile coordinates shape')
    good_tiles = (tiles >= 0).all(axis=1) & ((tiles[:,0] <= max_x) & (tiles[:,1] <= max_y))
    if not good_tiles.all():
        if ignore_bad_tiles: tiles = tiles[good_tiles]
        else: raise ValueError('Invalid tile coordinates')
    return __rm_dup_tiles(tiles[:,::-1])

def __sort_tiles(tiles):
    """Sort a list of tiles."""
    from numpy import lexsort
    return tiles.take(lexsort(tiles.T[::-1]), 0)

def __rm_dup_tiles(tiles):
    """Removes duplicate tiles from a list of tiles. Returns them sorted."""
    from numpy import concatenate
    tiles = __sort_tiles(tiles)
    return tiles.compress(concatenate(([True], (tiles[1:] != tiles[:-1]).all(axis=1))), 0)

def __get_ntasks_and_nthreads(ntasks, nthrds, model, tilesize, max_ntasks):
    """
    Gets the ntasks and nthreads arguments for CHM_test. Tries to maximize the number of tasks
    running based on the amount of memory, then maximize then number of threads based on the number
    of CPUs.
    """
    from psutil import cpu_count, virtual_memory
    base_mem = 0 # TODO: get expected amount of shared memory
    task_mem = CHM_test_max_mem(tilesize, model)//2
    ncpus = cpu_count(True) # we use the logical number of CPUs, not physical
    if ntasks is None and nthrds is None:
        ntasks = __clip((virtual_memory().available-base_mem)//task_mem, 1, min(max_ntasks, ncpus))
        nthrds = __clip(ncpus//ntasks, 1, ncpus)
    elif ntasks is None:
        nthrds = __clip(nthrds, 1, ncpus)
        ntasks = __clip((virtual_memory().available-base_mem)//task_mem, 1, min(max_ntasks, ncpus))
    elif nthrds is None:
        ntasks = __clip(int(ntasks),   1, max_ntasks)
        nthrds = __clip(ncpus//ntasks, 1, ncpus)
    else:
        ntasks = __clip(int(ntasks), 1, max_ntasks)
        nthrds = __clip(int(nthrds), 1, ncpus)
    return ntasks, nthrds
def __clip(x, a, b): return b if x > b else (a if x < a else x)
    
def __gen_shapes(sh, nlevels):
    """
    Generate a list of shapes that starts with the given shape (height by width) and then each
    subsequent shape is half the height and width (rounded up) from the one before it. The list has
    a total of Nlevel+1 items in it.
    """
    shapes = [None]*(nlevels+1)
    shapes[0] = sh
    for level in xrange(1, nlevels+1):
        shapes[level] = sh = (sh[0]+1)//2, (sh[1]+1)//2
    return shapes

def __get_arrays(im_sm, out_sms, ds_sm, sh):
    """
    Get numpy arrays from shared memory for the image (at various downsamplings), outputs/clabels,
    and contexts (downsampled clables). The shape of the level 0 image is required as well. The
    output shared memory needs to be a list of shared memories. The others are a single shared
    memory. Returns a list of images at various downsamplings (one for each level), outputs for
    each level, and a list of contexts for each level (where "each level" means all Nlevel+1
    levels), and the temporary context when reaching level=0.
    """
    from numpy import frombuffer, float64, uint8
    shapes = __gen_shapes(sh, len(out_sms)-1)

    # im_sm is full-sized - it is only used for the original full-sized image
    # each out_sm is work-sized for its level - only ever used for the output of the various levels
    # ds_sm is work-sized* - it is used for several purposes
    #                       for level > 0:
    #                           the first 'half' is used for the downsampled image (each full-sized)
    #                           the second 'half' is used for several contexts (each work-sized*)
    #                       for stage != 1 and level == 0:
    #                           used as a copy of the output from the last level == 0
    #                       for stage == 1 and level == 0: not used
     
    sizes = [sh[0]*sh[1] for sh in shapes]
    sh_sz = zip(shapes, sizes)
    
    ims = [frombuffer(im_sm, uint8, sizes[0]).reshape(shapes[0])]+ \
          [frombuffer(ds_sm, uint8, sz).reshape(sh) for sh,sz in sh_sz[1:]] # TODO: float64
    outs = [frombuffer(o_sm, float64).reshape(sh) for o_sm,sh in zip(out_sms, shapes)]
    
    f64sz = outs[0].itemsize
    off = sizes[0] // 2 * f64sz
    cntxts = [[frombuffer(ds_sm, float64, sz, off+i*sz*f64sz).reshape(sh) for i in xrange(lvl)]
              for lvl,(sh,sz) in enumerate(sh_sz)]

    return ims, outs, cntxts, frombuffer(ds_sm, float64, sizes[0]).reshape(shapes[0])

def __run_chm_test_procs(mems, model, regions, ntasks, nthreads):
    """
    Starts ntasks processes each running the function __run_chm_test_proc. Then handles the queue
    that they are running with and manages downsampling of contexts and other tasks.
    """
    from multiprocessing import JoinableQueue, Process
    from itertools import izip
    from numpy import copyto
    from .utils import MyDownSample1
    print("Running CHM test with %d task%s and %d thread%s per task" %
          (ntasks, 's' if ntasks > 1 else '', nthreads, 's' if nthreads > 1 else ''))
    nthreads_full = ntasks*nthreads

    # View the shared memory as numpy arrays
    ims, outs, contexts, out_tmp = __get_arrays(*mems)

    # Start the child processes
    q = JoinableQueue()
    args = (mems, model, nthreads, q)
    processes = [Process(target=__run_chm_test_proc, name="CHM-test-%d"%p, args=args) for p in xrange(ntasks)]
    for p in processes: p.daemon = True; p.start()
        
    # Go through each stage
    for m in model:
        stage, level = m.stage, m.level
        #print("Preparing for stage %d level %d..." % (stage, level))
        if level == 0:
            # Reset image and copy the level 0 output to temporary
            im = ims[0]
            # TODO: use utils.copy
            copyto(out_tmp, outs[0])
        else:
            # Downsample image and calculate contexts
            im = MyDownSample1(im, ims[level], nthreads_full)
            for c,o in izip(contexts[level-1], contexts[level]): MyDownSample1(c, o, nthreads_full)
            MyDownSample1(outs[level-1], contexts[level][-1], nthreads_full)
            #for i,c in enumerate(contexts[level]): __save('cntxt-%d-%d.png'%(level,i), c)
        #__save('im-%d.png'%(level), im)
        #print(im.min(), im.max())

        # Load the queue and wait
        for region in regions[level]: q.put_nowait((stage, level, tuple(region)))
        __wait_for_queue(q, stage, level, len(regions[level]), processes)
        #__save('out-%d-%d.png'%(stage,level), outs[level])

    # Tell all processes we are done and make sure they all actually terminate
    for _ in xrange(ntasks): q.put_nowait(None)
    for p in processes: p.join()

    # Done! Return the output image
    return outs[0]

def __save(name, im):
    # TODO: remove
    from scipy.misc import imsave
    from numpy import uint8
    if im.dtype == uint8: imsave(name, im)
    else: imsave(name, (im*255).round().astype(uint8))

def __wait_for_queue(q, stage, level, total_tiles, processes):
    """
    Waits for all items currently on the queue to be completed. If able, progress updates are
    outputed to stdout. This is dependent on two undocumented attributes of the JoinableQueue class.
    If they are not available, no updates are produced.

    If a processes crashes, this will raise an error after terminating the rest of the processes.
    """
    import time
    from sys import stdout
    msg = "Computing stage %d level %d..." % (stage, level)
    if hasattr(q, '_cond') and hasattr(q, '_unfinished_tasks'):
        # NOTE: this uses the undocumented attributes _cond and _unfinished_tasks of the
        # multiprocessing.Queue class. If they are not available, we just use q.join() but
        # then we can't check for failures or show progress updates.
        #pylint: disable=protected-access
        start_time = time.time() # NOTE: not using clock() since this thread spends most of its time waiting
        eol,timeout = ('\r',5) if stdout.isatty() else ('\n',60)
        stdout.write(msg+eol)
        stdout.flush()
        last_perc = 0
        with q._cond:
            while True:
                q._cond.wait(timeout=timeout)
                unfinished_tasks = q._unfinished_tasks.get_value()
                if unfinished_tasks == 0: print("%s Completed                          "%msg); break
                __check_processes(processes)
                perc = (total_tiles - unfinished_tasks) / total_tiles
                if perc > last_perc:
                    elapsed = time.time() - start_time
                    secs = elapsed/perc-elapsed
                    stdout.write("%s %4.1f%%   Est %d:%02d:%02d remaining     %s" %
                                 (msg, perc*100, secs//3600, (secs%3600)//60, round(secs%60), eol))
                    stdout.flush()
                    last_perc = perc
    else:
        stdout.write(msg)
        stdout.flush()
        q.join()
        print("Completed")
    __check_processes(processes)

def __check_processes(processes):
    """Checks that all processes are still alive, and if not, kills the rest"""
    if any(not p.is_alive() for p in processes):
        for p in processes:
            if p.is_alive(): p.terminate()
        raise RuntimeError('A process has crashed so we are giving up')

def __run_chm_test_proc(mems, model, nthreads, q):
    """
    Runs a single CHM test sub-process. The first argument is a tuple to expand as the arguments to
    __get_arrays(), the model is a Model object, nthreads is the number of threads this process
    should be using, and q is the JoinableQueue of tiles to work on.

    This will call get() on the Queue until a None is retrieved, then it will stop. If the item is
    not None it must be a stage, level, and region to be processed.
    """
    # Get the images and outputs in shared memory as numpy arrays
    ims, outs, cntxts, out_tmp = __get_arrays(*mems)
    # Protect ourselves against accidental writes to these arrays
    for im in ims: im.flags.writeable = False
    for c in cntxts:
        for c in c: c.flags.writeable = False

    # Process the queue
    prev_level = -1
    while True:
        tile = q.get()
        try:
            if tile is None: break # All done!
            stage, level, region = tile
            if level != prev_level:
                im, out, mod = ims[level], outs[level], model[stage,level]
                if stage != 1 and level == 0:
                    get_contexts = __get_level0_contexts_func(im.shape, out_tmp, outs, mod, nthreads)
                else:
                    _contexts = cntxts[level]
                    get_contexts = lambda region:(_contexts,region)
            contexts, cntxt_rgn = get_contexts(region)
            X = mod.filter(im, contexts, None, region, cntxt_rgn, nthreads)
            del contexts
            out[region[0]:region[2],region[1]:region[3]] = mod.evaluate(X, nthreads)
            del X
        finally: q.task_done()

def __get_level0_contexts_func(shape, out_tmp, outs, model, nthreads):
    """Get a function for calculating the contexts and context region for stage!=1 level 0 tiles."""
    padding = model.context_filter.padding
    scales = [1<<lvl for lvl in xrange(1, len(outs))]
    IH, IW = shape
    def __get_level0_contexts(region):
        from .utils import MyUpSample
        
        # Get the region for the level 0 context and the shape of the output context
        T,L,B,R = region
        T,L,B,R = max(T-padding, 0), max(L-padding,0), min(B+padding,IH), min(R+padding,IW)
        cntxt_rgn = (region[0]-T, region[1]-L, region[2]-T, region[3]-L) # region of the output contexts
        H, W = B-T, R-L # overall shape of the output context
        
        # Get the contexts for each level
        regions = [(T//S, L//S, (B+S-1)//S, (R+S-1)//S) for S in scales]
        offs = [(T,L)] + [(T%S, L%S) for S in scales]
        contexts = [out_tmp] + [MyUpSample(c,lvl,None,rgn,nthreads) for lvl,(c,rgn) in enumerate(zip(outs[1:],regions),1)]
        contexts = [c[T:T+H,L:L+W] for c,(T,L) in zip(contexts,offs)]
        return contexts, cntxt_rgn
    return __get_level0_contexts

def testCHM(im, model, nthreads=1):
    # CHANGED: the 'savingpath' (now called model) can now either be the path of the folder
    # containing the models or it can be an already loaded model
    # CHANGED: dropped Nstage, Nlevel, NFeatureContexts arguments - these are included in the model
    from numpy import empty
    from .utils import MyDownSample1, MyUpSample
    from .model import Model
    model = Model.load(model)
    nstages = model.nstages
    sh = im.shape
    clabels = None
    for sm in model:
        stage, level = sm.stage, sm.level
        if level == 0:
            imx = im
            contexts = [] if stage == 1 else \
                       [MyUpSample(c,i,nthreads=nthreads)[:sh[0],:sh[1]] for i,c in enumerate(clabels)]
            clabels = []
        else:
            imx = MyDownSample1(imx, None, nthreads)
            contexts = [] if level == 1 else [MyDownSample1(c, None, nthreads) for c in contexts]
            contexts.append(MyDownSample1(clabels[-1], None, nthreads))
        X = sm.filter(imx, contexts, nthreads=nthreads)
        if stage == nstages and level == 0: del im, imx, contexts
        clabels.append(sm.evaluate(X, nthreads))
        del X
    return clabels[0]

def CHM_test_max_mem(tilesize, model):
    """
    Gets the theoretical maximum memory usage for CHM-test for a given tile size and a model, plus a
    small fudge factor that is larger than any additional random overhead that might be needed.

        tilesize a tuple of height and width for the size of a tile
        model    the model that will be used during testing

    This is caclulated using the following formula:
        (476+57*(Nlevel+1))*8*tilesize + 200*8*tilesize + 20*8*tilesize
    Where 476 is the number of filter features, 57 is the number of context features generated
    at each level, 8 is the size of a double-precision floating point, 200 is number of
    discriminants used at level = 0 (when there are the most context features), and 20 is the number
    of discriminants per group at that level.

    Theoretical maximum memory usage is 7.31 GB (for 1000x1000 tiles and 4 levels). In practice I am
    seeing this +0.03 GB which is not too much overhead, probably estimating 100 MB overhead would
    be good enough in all cases.

    Note: this now asks the model for it's evaluation memory per pixel, adds 100 bytes per pixel,
    then multiplies by the number of pixels.
    """
    from numpy import asarray
    tilesize = asarray(tilesize)
    sizes = [None]*(model.nlevels+1)
    sizes[0] = tilesize[0]*tilesize[1]
    for lvl in xrange(1, model.nlevels+1):
        if not (tilesize%2).any(): tilesize = tilesize // 2
        sizes[lvl] = tilesize[0]*tilesize[1]
    return max((m.evaluation_memory+100)*sizes[m.level] for m in model) # the +100 bytes/pixel is a fudge-factor


def __chm_test_main():
    """The CHM test command line program"""
    from numpy import iinfo, dtype
    from pysegtools.images.io import FileImageSource

    # Parse Arguments
    im_path, output, model, tilesize, tiles, ntasks, nthreads, hist_eq, dt = \
             __chm_test_main_parse_args()

    # Process input and save to an output
    im = FileImageSource.open(im_path, True)
    out = CHM_test(im, model, tilesize, tiles, ntasks, nthreads, hist_eq, True)
    if dtype(dt).kind == 'u':
        out *= iinfo(dt).max
        out.round(out=out)
    FileImageSource.create(output, out.astype(dt, copy=False), True).close()

def __chm_test_main_parse_args():
    """Parse the command line arguments for the CHM test command line program."""
    #pylint: disable=too-many-locals, too-many-branches
    import os.path
    from sys import argv
    from getopt import getopt, GetoptError
    from pysegtools.general.utils import make_dir
    from pysegtools.images.io import FileImageSource
    
    from numpy import uint8, uint16, uint32, float32, float64
    dt_trans = {'u8':uint8, 'u16':uint16, 'u32':uint32, 'f32':float32, 'f64':float64}

    # Parse and minimally check arguments
    if len(argv) < 3: __chm_test_usage()
    if len(argv) > 3 and argv[3][0] != "-":
        __chm_test_usage("You provided more than 2 required arguments")

    # Check the input image
    im_path = argv[1]
    if not FileImageSource.openable(im_path, True):
        __chm_test_usage("Input image is of unknown type")

    # Check the output image
    output = argv[2]
    if output == '': output = '.'
    output_ended_with_slash = output[-1] == '/'
    output = os.path.abspath(output)
    if output_ended_with_slash:
        if not make_dir(output): __chm_test_usage("Could not create output directory")
    if os.path.isdir(output):
        if os.path.samefile(os.path.dirname(im_path), output):
            __chm_test_usage("If the output is to a directory it cannot be the directory the source image is in")
        output = os.path.join(output, os.path.basename(im_path))
    else:
        if len(os.path.splitext(output)[1]) < 2: output += os.path.splitext(im_path)[1]
        if not make_dir(os.path.dirname(output)): __chm_test_usage("Could not create output directory")
    if not FileImageSource.creatable(im_path, True):
        __chm_test_usage("Output image is of unknown type")

    # Get defaults for optional arguments
    model = './temp/'
    tilesize = None
    tiles = []
    dt = uint8
    hist_eq = False
    ntasks = None
    nthreads = None

    # Parse the optional arguments
    try: opts, _ = getopt(argv[3:], "hm:t:T:n:N:d:")
    except GetoptError as err: __chm_test_usage(err)
    for o, a in opts:
        if o == "-m":
            model = a
            if not os.path.isdir(model): __chm_test_usage("Model folder is not a directory")
        elif o == "-t":
            try:
                if 'x' in a: W,H = [int(x,10) for x in a.split('x', 1)]
                else:        W = H = int(a,10)
            except ValueError: __chm_test_usage("Tile size must be a positive integer or two positive integers seperated by an x")
            if W <= 0 or H <= 0: __chm_test_usage("Tile size must be a positive integer or two positive integers seperated by an x")
            tilesize = W,H
        elif o == "-T":
            try: C,R = [int(x,10) for x in a.split(',', 1)]
            except ValueError: __chm_test_usage("Tile position must be two positive integers seperated by a comma")
            if C < 0 or R < 0: __chm_test_usage("Tile position must be two positive integers seperated by a comma")
            tiles.append((C,R))
        elif o == "-H": hist_eq = True
        elif o == "-d":
            a = a.lower()
            if a not in dt_trans: __chm_test_usage("Data type must be one of u8, u16, u32, f32, or f64")
            dt = dt_trans[a]
        elif o == "-n":
            try: ntasks = int(a, 10)
            except ValueError: __chm_test_usage("Number of tasks must be a positive integer")
            if ntasks <= 0: __chm_test_usage("Number of tasks must be a positive integer")
        elif o == "-N":
            try: nthreads = int(a, 10)
            except ValueError: __chm_test_usage("Number of threads must be a positive integer")
            if nthreads <= 0: __chm_test_usage("Number of threads must be a positive integer")
        else: __chm_test_usage("Invalid argument %s" % o)
    if len(tiles) == 0: tiles = None

    return im_path, output, model, tilesize, tiles, ntasks, nthreads, hist_eq, dt

def __chm_test_usage(err=None):
    import sys
    if err is not None:
        print(err, file=sys.stderr)
        print()
    from . import __version__
    print("""CHM Image Testing Phase.  %s
    
%s <input> <output> <optional arguments>
  input_file    The input image to read.
                Accepts any 2D image accepted by `imstack` for reading.
  output_file   The output file or directory to save to.
                Accepts any 2D image accepted by `imstack` for writing.
	
Optional Arguments:
  -m model_dir  The folder that contains the model data. Default is ./temp/.
                For MATLAB models this folder contains param.mat and
                MODEL_level#_stage#.mat. For Python models this contains model,
                model-#-#, and model-#-#.npy.
  -t tile_size  Set the tile size to use as WxH. By default the tile size is the
                the same size as the training images (which is believed to be
                optimal for accuracy). Old models do not include the size of the
                training images and then 1024x1024 is used if not given. Note
                that for speed and memory it is optimal if this is a multiple of
                2^Nlevel of the model.
  -T C,R        Specifies that only the given tiles be processed by CHM while
                all others simply output black. Each tile is given as C,R (e.g.
                2,1 would be the tile in the third column and second row). Can
                process multiple tiles by using multiple -T arguments. The tiles
                are defined by multiples of tile_size. A tile position out of
                range will be ignored. If not included then all tiles will be
                processed.
  -H            Histogram-equalize the testing images to the training image
                histogram (if provided in the model). If not used, the testing
                data should already be properly equalized.
  -d type       Set the output type of the data, one of u8 (default), u16, u32,
                f32, or f64; the output image type must support the data type.
  -n ntasks     How many separate tasks to run in parallel. Each task processes
                a single tile of the image at a time.
  -N nthreads   How many threads to use per task. In general, each additional
                parallel task takes up a lot of memory (up to 7.5 GB for
                1000x1000 with Nlevel=4) while each additional thread per task
                does not really increase memory usage, however running two tasks
                in parallel each with 1 thread is faster than giving a single
                task two threads. Default is to run twice as many tasks as can
                fit in memory (since the max memory is only used for a short
                period of time) and divide the rest of the CPUs among the tasks.
                If only one value is given the other is calculated using it."""
          % (__version__, sys.argv[0]), file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    __chm_test_main()
