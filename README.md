Cascaded Hierarchical Model Automatic Segmentation Algorithm
============================================================

This is an algorithm designed for automatic segmention of cellular structures in electron microscopy data.

The core algorithm is in the "algorithm" directory while wrappers for it to do things such as run on a cluster are in the "wrappers" directory.

More details about running the algorithm or any of the wrappers is contained in readmes in the respective directories.

Testing
=======

In the directory test/ are unit tests written in BATS: Bash Automated Testing System
to test CHM_test.sh and in the future hopefully the rest of CHM.  

BATS can be obtained from here:  https://github.com/sstephenson/bats

To run the main unit tests simply run:

 bats test/

Under test/chm_system_tests are system tests that run CHM against
real data.  These tests require Image Magick to be installed as well
as matlab with the image processing toolbox and appropriate licenses.

