#!/bin/bash

usage()
{
  echo "CHM Image Training Phase Script.

$0 <inputs> <labels> <optional arguments>
  inputs          The raw image files to use.
  labels          The label / ground truth images (1=feature, 0=background)
The file names need not line up, but there must be equal amounts of inptus and
labels. They are matched up in the order they are given. If using a folder or
wildcard input the order may not be completely predictable, but most systems
will put these in lexographic order. The files can be specified in multiple
ways. It needs to be one of the these or a comma-separated list of these:
 * path to a folder            - all PNGs in that folder
 * path to a file              - only that file 
 * path with numerical pattern - get all files matching the pattern
     pattern must have #s in it and end with a semicolon and number range
     the #s are replaced by the values at the end with leading zeros
     example: in/####.png;5-15 would do in/0005.png through in/0015.png
     note: the semicolon needs to be escaped or in double quotes in some shells
 * path with wildcard pattern  - get all files matching the pattern
     pattern has * in it which means any number of any characters
     example: in/*.tif does all TIFF images in that directory
     note: the asterisk needs to be escaped or in double quotes in some shells

Optional Arguments:
  -m model_folder The folder that contains the model data. Default is ./temp/.
                  (contains param.mat and MODEL_level#_stage#.mat)
  -S Nstage       The number of stages of training to perform. Must be >=2.
                  Default is 2.
  -L Nlevel       The number of levels of training to perform. Must be >=1.
                  Default is 4.
  -s              Single-thread / non-parallel. Normally one small step of this
                  is done in parallel using all available physical cores." 1>&2;
  exit 1;
}

# Parse and minimally check arguments
if [[ $# -lt 2 ]]; then usage; fi
INPUTS=$1;
LABELS=$2;
MODEL_FOLDER=./temp/;
SINGLE_THREAD=; # normally blank, "-nojvm" when single-threaded which disables parellism (along with other unnecessary things)
declare -i NSTAGE=2;
declare -i NLEVEL=4;
shift 2
while getopts ":sm:S:L:" o; do
  case "${o}" in
    s)
      SINGLE_THREAD=-nojvm;
      ;;
    m)
      MODEL_FOLDER=${OPTARG};
      if [ ! -d "$MODEL_FOLDER" ]; then echo "Model folder is not a directory." 1>&2; echo; usage; fi;
      ;;
    S)
      NSTAGE=${OPTARG};
      if [[ $NSTAGE -lt 2 ]]; then echo "Invalid number of training stages." 1>&2; echo; usage; fi;
      ;;
    o)
      NLEVEL=${OPTARG};
      if [[ $NLEVEL -lt 1 ]]; then echo "Invalid number of training levels." 1>&2; echo; usage; fi;
      ;;
    *)
      echo "Invalid argument." 1>&2; echo; 
      usage;
      ;;
    esac
done


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
if [[ -n "${SINGLE_THREAD}" ]]; then SINGLE_THREAD="${SINGLE_THREAD} -singleCompThread";
elif (( `nproc` > 24 )); then SINGLE_THREAD=-singleCompThread; fi;

# Run the main matlab script (need JVM for parallel)
matlab -nodisplay ${SINGLE_THREAD} -r "run_from_shell('CHM_train(''${INPUTS}'',''${LABELS}'',''${MODEL_FOLDER}'',${NSTAGE},${NLEVEL});');";
matlab_err=$?;

# Cleanup
stty sane >/dev/null 2>&1 # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
