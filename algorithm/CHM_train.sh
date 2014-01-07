#!/bin/bash

if [[ $# < 2 || $# > 5 ]]; then
  echo "usage: $0 train_folder label_folder [model_folder=./temp/ [Nstage=2 [Nlevel=4]]]";
  exit
fi

trainfolder=$1;
labelfolder=$2;
opt=;
if [[ $# > 2 ]]; then opt="${opt},''$3''"; fi
if [[ $# > 3 ]]; then opt="${opt},$4"; fi
if [[ $# > 4 ]]; then opt="${opt},$5"; fi

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
ARGS=
if (( `nproc` > 24 )); then ARGS=-singleCompThread; fi;

# Run the main matlab script (need JVM for parallel)
matlab -nodisplay ${ARGS} -r "run_from_shell('CHM_train(''${trainfolder}'',''${labelfolder}''${opt});');";
matlab_err=$?;

# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
