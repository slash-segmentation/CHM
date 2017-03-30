#!/bin/bash

usage()
{
  echo "CHM Image Training Phase Script.  @@VERSION@@

$0 <inputs> <labels> <optional arguments>
  inputs          The raw image files to use.
  labels          The label / ground truth images (1=feature, 0=background)
The file names need not line up, but there must be equal amounts of inptus and
labels. They are matched up in the order they are given. If using a folder or
wildcard input the order may not be completely predictable, but most systems
will put these in lexographic order. The files can be specified in multiple
ways. It needs to be one of the these or a comma-separated list of these:
 * path to a folder            - all PNGs and TIFFs in that folder
 * path to a file              - only that file 
 * path with numerical pattern - get all files matching the pattern
     pattern must have #s in it and end with a semicolon and number range
     the #s are replaced by the values at the end with leading zeros
     example: in/####.png;5-15 would do in/0005.png through in/0015.png
     note: the semicolon needs to be escaped or in double quotes in some shells
 * path with wildcard pattern  - get all files matching the pattern
     pattern has * in it which means any number of any characters
     example: in/lbl_*.tif does all TIFF images starting with lbl_ in 'in'
     note: the asterisk needs to be escaped or in double quotes in some shells

Optional Arguments:
  -m model_folder The folder that contains the model data. Default is ./temp/.
                  (contains param.mat and MODEL_level#_stage#.mat)
  -S Nstage       The number of stages of training to perform. Must be >=2.
                  Default is 2.
  -L Nlevel       The number of levels of training to perform. Must be >=1.
                  Default is 4.
  -r              Restart a failed training attempt. This will restart just
                  after the last completed stage/level. You must give the same
                  parameters (data, labels, ...) as before for the model to
                  make sense.
  -n              Number of threads." 1>&2;
  exit 1;
}

# Parse and minimally check arguments
if [[ $# -lt 2 ]]; then usage; fi
if [[ $# -gt 2 ]] && [ "${3:0:1}" != "-" ]; then
  echo "You provided more than 2 required arguments. Did you accidently use a glob expression without escaping the asterisk?" 1>&2; echo; usage; 
fi
INPUTS=$1;
LABELS=$2;
MODEL_FOLDER=./temp/;
declare -i RESTART=0; # 0=FALSE, 1=TRUE
SINGLE_THREAD=; # normally blank, "-nojvm" when single-threaded which disables parellism (along with other unnecessary things)
NTHREADS=0;
declare -i NSTAGE=2;
declare -i NLEVEL=4;
shift 2
while getopts ":n:rm:S:L:" o; do
  case "${o}" in
    s)
      SINGLE_THREAD="-nojvm -singleCompThread";
      ;;
    n)
      NTHREADS="$((OPTARG * 1))";
      ;;
    r)
      RESTART=1;
      ;;
    m)
      MODEL_FOLDER=${OPTARG};
      if [ -f "$MODEL_FOLDER" ]; then echo "Model folder already exists as a regular file." 1>&2; echo; usage; fi;
      ;;
    S)
      NSTAGE=${OPTARG};
      if [[ $NSTAGE -lt 2 ]]; then echo "Invalid number of training stages." 1>&2; echo; usage; fi;
      ;;
    L)
      NLEVEL=${OPTARG};
      if [[ $NLEVEL -lt 1 ]]; then echo "Invalid number of training levels." 1>&2; echo; usage; fi;
      ;;
    *)
      echo "Invalid argument." 1>&2; echo; 
      usage;
      ;;
    esac
done
if [ ! -d "$MODEL_FOLDER" ]; then
  mkdir $MODEL_FOLDER
  if [[ $? -ne 0 ]]; then echo "Model folder could not be created." 1>&2; echo; usage; fi;
fi

# We need to add the path with the script in it to the MATLAB path
# This is a bit complicated since this script is actually a symlink
# See stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
if [ -n "$MATLABPATH" ]; then
    MATLABPATH_ORIGINAL=$MATLABPATH
    export MATLABPATH="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )":$MATLABPATH
else
    export MATLABPATH="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"
fi

# Set 'singleCompThread' for MATLAB if there are a lot of processors
if (( NTHREADS == 1 )); then SINGLE_THREAD="-nojvm -singleCompThread";
elif [[ -n `which nproc 2>/dev/null` ]] && (( `nproc` > 24 )); then SINGLE_THREAD=-singleCompThread; fi;

# Run the main matlab script (need JVM for parallel)
matlab -nodisplay ${SINGLE_THREAD} -r "run_from_shell('CHM_train(''${INPUTS}'',''${LABELS}'',''${MODEL_FOLDER}'',${NSTAGE},${NLEVEL},${RESTART},${NTHREADS});');";
matlab_err=$?;

# Cleanup
stty sane >/dev/null 2>&1 # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
