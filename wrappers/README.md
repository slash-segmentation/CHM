CHM Wrappers
============

This directory contains various CHM utility programs that are described
below.


create_release.sh (Creates CHM releases)
========================================

This script is used by CHM developers to create CHM releases.  To run, 
the caller must be in the CHM/wrappers directory and must have Matlab
installed with Image Processing Toolbox and Matlab compiler.  

Usage: 

Invocation with no arguments generates tarball files and calls git commit & push:

  ./create_release.sh 

Invocation with -s skips git calls,but adds .test suffix to VERSION:

  ./create_release.sh -s

This script creates two release versions of CHM.  One version is just a packaging of
the matlab files and is known as the source version (chm-source).  The other
version is a Matlab compiled version and is known as (chm-compiled).  These versions
are put in directories in CHM/wrappers/ that are tarred and gzipped.  

The naming convention for the chm-source is:  chm-source-VERSION.tar.gz
where VERSION is 2.1.(# of commits to git ie 456)

The naming convention for the chm-compiled is: chm-compiled-VERSION.tar.gz 
where VERSION is (matlab version r2013a)-2.1.(# of commits to git ie 456)
 
