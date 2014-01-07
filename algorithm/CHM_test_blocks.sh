#!/bin/bash

if [[ $# < 3 || $# > 5 ]]; then
  echo "usage: $0 input_files output_folder blocksize [bordersize=0 [model_folder=./temp/]]";
  exit
fi

inputfiles=$1;
outputfolder=$2;
blocksize=$3;
opt=;
if [[ $# > 3 ]]; then opt="${opt},$4"; fi
if [[ $# > 4 ]]; then opt="${opt},''$5''"; fi

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

# Run the main matlab script (no JVM and multi comp)
matlab -nodisplay -nojvm -r "run_from_shell('CHM_test_blocks(''${inputfiles}'',''${outputfolder}'',${blocksize}${opt});');";
matlab_err=$?;

# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
