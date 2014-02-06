These are simply compiled versions of the CHM_test, CHM_test_blocks, and
CHM_train MATLAB entry points. These work just as the MATLAB functions do.
The shell scripts provided are modified from the ones in the algorithm folder,
modified to support some special features of the compiled versions (like always
being single-threaded and needing to find the MATLAB directory).

To run these you either need MATLAB installed with the Image Processing Toolbox
or the MATLAB Compiler Runtime (MCR, which is free). They need to be the exact
version which the programs were compiled for, which can be found in the file
"matlab-version.txt" (including the platform and MATLAB version, for example
glxna64, R2011b).

To tell the shell script where to find MATLAB or MCR, you can either make sure
'matlab' is in the PATH, define MCR_DIR to the root directory (which contains
bin, runtime, sys, and other folders), or use the -M command line argument to
specify the root directory.

The shell scripts setup a cache directory in the /tmp/ folder so that the
programs will start faster on subsequent runs. This is especially import when
the program is located on a networked drive being accessed by multiple machines
simultaneously. If the /tmp/ folder is not the desired location for the cache
simply define MCR_CACHE_ROOT before running the script to a directory and make
sure that directory exists. In that root directory will be placed a folder like
.mcrCache7.16 (for MCR v7.16) which will contain the actual cache. If you want
to disable caching, set MCR_CACHE_ROOT to a non-existent directory.
