panfishCHM
==========

Runs CHM via Panfish Multi-cluster Submission System.


Overview
========

This source tree contains a a set of wrapper scripts that facilitate 
running CHM via Panfish.  This code is in ALPHA state and more then likely
contains errors and omissions.
  
NO WARRANTY IS PROVIDED.  USE AT YOUR OWN RISK.

Prerequisites
=============

To invoke the **printonly** mode only bash and linux with gzip and
tar commands are required.  To run jobs a properly configured
copy of *Panfish* must be installed as well as *matlab* 2011b or newer
along with necessary licenses as well as the image processing toolbox.

How to setup
============

Assuming you are in CHM/wrappers/panfish directory simply invoke the following
command:

*./deploy.sh local*

The above command will create a directory in the current directory named
*panfishCHM* which will contain *runCHMviaPanfish.sh* along with a copy
of CHM pulled from the CHM/algorithm folder.

After *panfishCHM* is created the file *panfishCHM/panfishCHM.config* must
be set with the correct paths to panfish and matlab.


The *panfishCHM* directory can be moved to another location such as ~/bin/.



How to run
==========

Run the following command to see options

*<PATH TO panfishCHM>/runCHMviaPanfish.sh*


What it does
============

The script *runCHMviaPanfish.sh* takes several command line inputs
and generates a directory with scripts that can be easily run via
Panfish on HPC compute resources.  

The generated directory has this structure:

    dir/
        runCHM.sh
        runCHM.sh.config
        helperfuncs.sh
        CHM.tar.gz
        out/
           log/

Breakdown of above files:

* *runCHM.sh*
 
    Runs CHM on a image slice writing output to *out/* directory.
    The slice is defined by the *SGE_TASK_ID* environment
    variable.  The output written is *out/XXX.tif* where
    *XXX* is the input slice file name.  In addition,
    an *out/log/#.XXX.log* file is written where *#*
    is *JOB_ID.SGE_TASK_ID* environment variables and
    this file contains output from CHM code.

* *runCHM.sh.config* 

    Configuration file used by *runCHM.sh* each line is prefixed 
    with #::: and *runCHM.sh* reads *SGE_TASK_ID* variable and searches
    the configuration file for matching lines to determine what slice to process.

* *helperfuncs.sh* 

    Holds helper functions for runCHM.sh and other scripts.

* *CHM.tar.gz* 

    Contains CHM code, trained model data, and helper scripts.  
    When a job is run this tar ball is copied to a local 
    scratch space on the machine and decompressed.  

* *out/*

    Contains output *.tif slices and *.log files (under log/ subdirectory)
         

*For more information run runCHMviaPanfish.sh with no arguments* 
