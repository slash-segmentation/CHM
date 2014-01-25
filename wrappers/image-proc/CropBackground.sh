#!/bin/bash

usage()
{
    echo "Usage:
$0 <image_in> <t> <l> <b> <r> <image_out>
  image_in        The path to the grayscale image to read.
  t,l,b,r         The top, left, bottom, and right corners of the foreground.
  image_out       The path to the grayscale image to write." 1>&2;
  exit 1;
}

# Parse and minimally check arguments
if [[ $# != 6 ]]; then usage; fi;
IMAGE_IN=$1;
declare -i T=$2;
declare -i L=$3;
declare -i B=$4;
declare -i R=$5;
IMAGE_OUT=$6;

if [[ $T -le 0 || $L -le 0 || $B -le $T || $R -le $L ]]; then echo "Foreground rectangle not valid." 1>&2; echo; usage; fi; 

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


# Run the main matlab script
matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('imwrite(CropBackground(''${IMAGE_IN}'',${T},${L},${B},${R}),''${IMAGE_OUT}'');');";
matlab_err=$?;


# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
