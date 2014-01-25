#!/bin/bash

usage()
{
    echo "Usage:
$0 <image_in> <t> <l> [<h> <w> | <template_image>] <image_out>
  image_in        The path to the grayscale image to read.
  t,l             The top and left corners of the foreground.
  h,w             The height and width of the final image.
  template_image  Get the final image height and width from the template image.
  image_out       The path to the grayscale image to write." 1>&2;
  exit 1;
}

# Parse and minimally check arguments
if [[ $# < 5 || $# > 6 ]]; then usage; fi;
IMAGE_IN=$1;
declare -i T=$2;
declare -i L=$3;
if [[ $T -le 0 || $L -le 0 ]]; then echo "Top-left corner not valid." 1>&2; echo; usage; fi; 
if [[ $# > 5 ]]; then
  declare -i H=$4;
  declare -i W=$5;
  IMAGE_OUT=$6;
  if [[ $H -le $T || $W -le $L ]]; then echo "Height/width not valid." 1>&2; echo; usage; fi; 
else
  TEMPLATE_IMAGE=$4;
  IMAGE_OUT=$5;
  if [[ ! -f $TEMPLATE_IMAGE ]]; then echo "Template file does not exist." 1>&2; echo; usage; fi;
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


# Run the main matlab script
if [[ $# > 5 ]]; then
  matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('imwrite(AddBackground(''${IMAGE_IN}'',${T},${L},${H},${W}),''${IMAGE_OUT}'');');";
else
  matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('i=imfinfo(''${TEMPLATE_IMAGE}'');imwrite(AddBackground(''${IMAGE_IN}'',${T},${L},i(1).Height,i(1).Width),''${IMAGE_OUT}'');');";
fi
matlab_err=$?;


# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
