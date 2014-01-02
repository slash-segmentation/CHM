#!/bin/bash

if [[ $# < 4 || $# > 5 ]]; then
  echo "usage: $0 train_folder label_folder test_folder output_folder [model_folder=./temp/]";
  exit
fi

date;

trainfolder=$1;
labelfolder=$2;
testfolder=$3;
testoutput=$4;
workfolder=./temp/;
if [[ $# > 4 ]]; then workfolder=$5; fi

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
matlab -nodisplay -singleCompThread -r "run_from_shell('TrainScript(''${trainfolder}'',''${labelfolder}'',''${testfolder}'',''${testoutput}'',''${workfolder}'');');";
matlab_err=$?;

# Cleanup
stty sane # restor terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

date

exit $matlab_err
