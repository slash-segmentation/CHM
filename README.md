Cascaded Hierarchical Model Automatic Segmentation Algorithm
============================================================

This is an algorithm designed for automatic segmention of cellular structures
in electron microscopy data.

The core algorithm is in the "algorithm" directory while wrappers for it to do
things such as run on a cluster are in the "wrappers" directory.

More details about running the algorithm or any of the wrappers is contained in
readmes in the respective directories.


Basic Usage
===========
If you have a MATLAB license with the Image Processing toolbox you need the
entire algorithm folder. If you don't then you need the wrappers/compiled
folder. You will also need the MCR for the MATLAB listed in the file
matlab-version.txt (downloadable from the Mathworks website).

The two main entry points are CHM_test.sh and CHM_train.sh. The raw MATLAB and
compiled versions work very similarly.

The first thing you need to run in CHM_train. In the most basic usage it takes
a set of training data images (grayscale, 8-bit) and a set of training label
images (0=no label, 1=labeled). These file sets are specified as a comma
seperated list of the following:
 * path to a folder            - all PNGs in that folder
 * path to a file              - only that file
 * path with numerical pattern - get all files matching the pattern
     pattern must have #s in it and end with a semicolon and number range
     the #s are replaced by the values at the end with leading zeros
     example: in/####.png;5-15 would do in/0005.png through in/0015.png
     note: the semicolon needs to be escaped or in quotes in some shells
 * path with wildcard pattern  - get all files matching the pattern
     pattern has * in it which means any number of any characters
     example: in/*.tif does all TIFF images in that directory
     note: the asterisk needs to be escaped or in quotes in some shells
Training will take on the order of a day and require lots of memory (50-150GB)
depending on the size of the dataset. Recommended that you use between 500x500
and 1000x1000 size training images with a total of 20-40 slices.

The output model is by default stored in ./temp. The only files required to
save are the .mat files in the root directory (the subdirectories contain
purely temporary files).

CHM_test then takes the model generated with CHM_train and creates probability
maps for how likely a pixel is the same thing as what was labelled during
training. The basic usage is to give a set of data images to process and the
output directory. This will take about 5-15 min per training-image-sized region
and 5-10 GB of RAM depending on data image and training data size.

Both CHM_test and CHM_train accept optional arguments. They share the arguments
-m to specify the model folder and -s to force single-threaded runs. The number
of training levels and stages while training can be adjusted with -S and -L and
the impact of changing them is being investigated at the moment. For testing
you should consider adding the -o #x# argument to cause blocks to be overlapped
which will remove edge effects both on the border of the images and the
interior of the images but can increase processing time. Values from 25-50 seem
to be good. You can tell CHM_test to only work on select tiles of the image
using -t.

The compiled version takes slightly different flags. First, it ignores the -s
(single-threaded) flag. Second it has an additional argument -M to specify the
location of MATLAB/MCR if it cannot be found automatically.


Testing
=======

In the directory test/ are unit tests written in BATS: Bash Automated Testing
System to test CHM_test.sh and in the future hopefully the rest of CHM.  

BATS can be obtained from here:  https://github.com/sstephenson/bats

To run the main unit tests simply run:

 bats test/

Under test/chm_system_tests are system tests that run CHM against real data.
These tests require Image Magick to be installed as well as matlab with the
image processing toolbox and appropriate licenses.

