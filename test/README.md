Unit Testing
============

This directory contains unit tests written in BATS: Bash Automated Testing
System to test CHM_test.sh and in the future hopefully the rest of CHM.  

BATS can be obtained from here:  https://github.com/sstephenson/bats

To run the main unit tests simply run from the root directory:

 bats test/

Under chm_system_tests are system tests that run CHM against real data. These
tests require Image Magick to be installed as well as matlab with the image
processing toolbox and appropriate licenses.

