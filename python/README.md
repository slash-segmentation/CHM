Python CHM Train and Test Algorithms
====================================

This package includes the CHM train and test algorithms written in Python with several
enhancements, including major speed and memory improvements, improved and added filters, and
improvements to the processing algorithm.

Models are not always compatible. MATLAB models created with `CHM_train` can be used with the
Python version with only minor differences in output. However, Python models cannot be used with
the MATLAB `CHM_train`.


Installation
------------
The following libraries must be installed:
 * gcc and gfortran (or another C and Fortran compiler)
 * Python 2.7
 * Python headers

The following libraries are strongly recommended:
 * virtualenv
 * linear-algebra package including devel (in order of preference: MKL, ATLAS+LAPACK, OpenBLAS+LAPACK, any BLAS library)
 * devel packages for image formats you wish to read:
    * PNG: zlib (note: uncompressed always available)
    * TIFF: libtiff (note: uncompressed always available)
    * JPEG: libjpeg or libjpeg-turbo
    * etc...
 * hdf5 devel package for reading and writing modern MATLAB files
 * fftw devel package for faster FFT calculations

These can all be installed with various Python managers such as Anaconda, Enthought, Python(x,y),
or WinPython. On Linux machines they can be installed globally with `yum`, `apt-get`, or similar.
For example on CentOS-7 all of these can be installed with the following:

    yum install gcc gcc-gfortran python python-devel python-virtualenv \
                atlas atlas-devel lapack lapack-devel lapack64 lapack64-devel \
                zlib zlib-devel libtiff libtiff-devel libjpeg-turbo libjpeg-turbo-devel \
                hdf5 hdf5-devel fftw fftw-devel

The recommended way to install if from here is to create a Python virtual environment with all of
the dependent Python packages. On Linux machines, setting this up would look like:
    
    # Create the folder for the virtual environment
    # Adjust this as you see fit
    mkdir ~/virtenv
    cd ~/virtenv
    
    # Create and activate the virtual environment
    virtualenv .
    source bin/activate

    # Install some of the dependencies
    # Note: these can be skipped but greatly speeds up the other commands
    pip install numpy
    pip install cython
    pip install scipy
    
    # Install the devel pysegtools (and all dependencies)
    git clone git@github.com:slash-segmentation/segtools.git
    pip install -e segtools[PIL,MATLAB,OPT,tasks]
    python -m pysegtools.imstack --check

    # Install the devel PyCHM
    git clone git@github.com:slash-segmentation/CHM.git
    pip install -e CHM/python[OPT]

Since the pysegtools and CHM packages are installed in editable mode (`-e`), if there are updates
to the code, you can do the following to update them:

    cd ~/virtenv/segtools
    git pull

    cd ~/virtenv/CHM/python
    git pull
    ./setup.py build_ext --inplace # builds any changed Cython modules


CHM Test
--------
Basic usage is:

    python -m chm.test input-image output-image <options>
    
General help will be given if no arguments (or invalid arguments) are provided

    python -m chm.test
    
The CHM test program takes a single 2D input image, calculates the labels according to a model,
and saves the labels to the 2D output image. The images can be anything supported by the `imstack`
program for 2D images. The output-image can given as a directory in which case the input image
filename and type are used. *The MATLAB command line allowed multiple files.*

By default this assumes the model data is stored in ./temp/. Typically this will be stored
somewhere else and using the `-m` option this can be set to any folder. The folder can contain
either a MATLAB model or a Python model. For MATLAB models the folder contains param.mat and
MODEL_level#_stage#.mat. For Python models the folder contains model, model-#-#, and model-#-#.npy.

The CHM test program splits the input image into a bunch of tiles and operates on each tile
separately. The size of the tiles to process can be given with `-t #x#`, e.g. `-t 512x512`. For
MATLAB models that include the training image size, that is used as the default. For Python models
the default used is 1024x1024. For speed and memory it is likely optimal if the tile size is a
multiple of 2^Nlevel of the model (typically Nlevel<=4, so should be a multiple of 16). *The MATLAB
command line called this option `-b`. Additionally, the MATLAB command line program overlapped
tiles (specified with `-o`) which is no longer ever done.*

Instead of computing labels for every tile, individual tiles can be specified using `-T #,#`, e.g.
`-T 0,2` computes the tile in the first column, third row. This option can be specified any number
of times to cause multiple tiles to be calculted. All tiles not calculated will output as black
(0). Any tile indices out of range are simply ignored. *The MATLAB command line called this option
`-t` and indexing started at 1 instead of 0.*

For MATLAB models that include the source histogram, the input images can automatically be
histogram-equalized to it by specifying `-H`. Python models do not include this information and in
general this should be done in a separate process anyways. *The MATLAB command line did histogram
equalization by default and was turned off with `-h`.*

By default the output image data is saved as single-byte grayscale data (0-255). The output data
type can be changed with `-d type` where `type` is one of `u8` (8-bit integer from 0-255), `u16`
(16-bit integer from 0-65535), `u32` (32-bit integer from 0-2147483647), `f32` (32-bit floating
point number from 0.0-1.0), or `f64` (64-bit floating-point number from 0.0-1.0). All of the other
data types increase the resolution of the output data. However, the output image format must
support the data type (for example, PNG supports u8 and u16 while TIFF supports all the types).

Finally, by default the CHM test program will attempt to use as much memory is available and all
logical processors available. If this is not desired, it can be tweaked using the `-n` and `-N`
options. The `-n` option specifies the number of tasks while `-N` specifies the number of threads
per task. Each additional task will take up significant memory (up to 7.5 GiB for 1000x1000 tiles
and Nlevel=4) while each additional thread per task doesn't effect memory significantly. However,
splitting the work into two tasks with one thread each will be much faster than having one task
with two threads. The default uses twice as many tasks as would fit in memory (since the maximum
memory usage is only used for a short period of time) and then divides the threads between all
tasks. If only one of `-n` or `-N` is given, the other is derived based on it. *The MATLAB command
line only had the option to be multithreaded or not with `-s`.*

*The MATLAB command line option `-M` is completely gone as there is no need for MATLAB or MCR to be
installed*


CHM Train
---------
Basic usage is:

    python -m chm.train input-images label-images <optionals>

General help will be given if no arguments (or invalid arguments) are provided

    python -m chm.train

The CHM train program takes a set of input images and labels and creates a model for use with CHM
test. The model created from Python cannot be used with the MATLAB CHM test program. The input
images are specified as a single argument for anything that can be given to `imstack -L` (the value
may need to be enclosed in quotes to make it a single argument). They must be grayscale images. The
labels work similarily. Anything that is 0 is considered background while anything else is
considered a positive label. The inputs and labels are match up in the order they are given and
paired images must be the same size.

By default this stores the model data in ./temp/. This can be set to a different directory using
the `-m`. If the option `-r` is also specified and this folder already contains (part of) a Python
model, then the model is run in 'restart' mode. In restart mode, the previous model is examained
and as much of it is reused as possible. This is useful for when a previous attempt failed partway
through or when desiring to add additional stages or levels to a model. If the filters are changed
from the original model, any completed stages/levels will not use the new filters but new
stages/levels will. The input images and labels must be the same when restarting.

The default number of stages and levels are 2 and 4 respectively. They can be set using `-S #` and
`-L #` respectively. The number of stages must be at least 2 while the number of levels must be at
least 1. Each additional stage will require very large amounts of time to compute, both while
training and testing. Additional levels don't add too much additional time to training or testing,
but do increase both. Typically, higher number of levels are required with larger structures and do
not contribute much for smaller structures. Some testing has shown that more than 2 levels is
pointless - at least for non-massive structures.

The filters used for generating features are, by default, the same used by the MATLAB CHM train but
without extra compatibility. The filters can be adjusted using the `-f` option in various ways. The
available filters are haar, hog, edge, gabor, sift, frangi, and intensity-<type>-# (where <type> is
either stencil or square and # is the size in pixels). To add filters to the list of current
filters do something like `-f +frangi`. To remove filters from the list of current filters do
something like `-f -hog`. Additionally, the list of filters can be specified directly, for example
`-f haar,hog,edge,gabor,sift,intensity-stencil-10` would specify the default filters. More then one
`-f` option can be given and they will build off of each other.

Besides filters being used to generate the features for images, a filter is used on the 'contexts'
from the previous stages and levels to generate additional features. This filter can be specified
with `-c`, e.g. `-c intensity-stencil-7` specifies the default filter used. This only supports a
single filter, so `+`, `-`, or a list of filters cannot be given.

The training algorithm requires also running the testing algorithm internally. Normally this is
saved as a series of NPY files for quick access to the data, but these files are not very useful
as an image format. To save them in a more useable format, the option `-o` can be given. The option
takes anything that can be given to `imstack -S` although quotes may be required to make it a
single argument. This means that if you want to see the results of running CHM test on the training
data you can get it for free. Like CHM test, this support the `-d` argument to specify the data
type used to save the data.

*The MATLAB command line option `-M` is completely gone as there is no need for MATLAB or MCR to be
installed*


Filters
-------

### Haar

Computes the Haar-like features of the image. This uses Viola and Jones' (2001) 2-rectangle
features (x and y directional features) of size 16. It uses their method of computing the integral
image first and then using fast lookups to get the features.

When used for MATLAB models the computations are bit slower but reduces the drift errors compared
to the MATLAB output. Technically it is slightly less accurate, but the numbers are in the range of
1e-8 off for a 1000x1000 image.

### HOG

Computes the HOG (histogram of oriented gradients) features of the image.

The original MATLAB function used float32 values for many intermediate values so the outputs from
this filter are understandably off by up to 1e-7. The improved accuracy is used for MATLAB or
Python models since it just adds more accuracy.

*TODO: more details - reference and parameters used*

### Edge

Computes the edge features of the image. This calculates the edge magnitudes by using convolution
with the second derivative of a Guassian with a sigma of 1.0 then returns all neighboring offsets
in a 7x7 block.

When used for MATLAB models, the tiles/images are padded with 0s instead of reflection. This is not
a good approach since a transition from 0s to the image data will result in a massive edge.

*TODO: more details - reference*

### Frangi

Computes the Frangi features of the image using the eigenvectors of the Hessian to compute the
likeliness of an image region to contain vessels or other image ridges, according to the method
described by Frangi (1998). This uses seven different Gaussian sigmas of 2, 3, 4, 5, 7, 9, and 11
each done with the image and the inverted image (looking for white and black ridges). The beta
value is fixed to 0.5 and c is dynamically calculated as half of the maximum Frobenius norm of all
Hessian matrices.

This is not used by default in Python models or at all for MATLAB models.

### Gabor

Computes several different Gabor filters on the image using all combinations of the following
parameters to create the kernels:
 * sigma:      2, 3, 4, 5, 6
 * lambdaFact: 2, 2.25, 2.5   (lambaFact = lamba / sigma)
 * orient:     pi/6, pi/3, pi/2, 2pi/3, 5pi/6, pi, 7pi/6, 4pi/3, 3pi/2, 5pi/3, 11pi/6, 2pi
 * gamma:      1
 * psi:        0 (phase)
The magnitude of the complex filtered image is used as the feature.

The original MATLAB code has a serious bug in it that it uses the imfilter function with a uint8
input which causes the filter output to be clipped to 0-255 and rounded to an integer before taking
the complex magnitude. This is a major problem as the Gabor filters have negative values (all of
which are set to 0) and can produce results above the input range, along with losing lots of
resolution in the data (from ~16 significant digits to ~3). So for MATLAB models the data is
simplified in this way, otherwise a much higher accuracy version is used.

*TODO: more details - reference*

### SIFT

*TODO*

### Intensity
Computes the neighborhood/intensity features of the image. Basically, moves the image around and
uses the shifted images values. Supports square or stencil neighborhoods of any radius >=1. In
compatibility mode these are always fixed to a particular type (stencil) and radius (10).
