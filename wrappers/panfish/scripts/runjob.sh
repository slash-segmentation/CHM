#!/bin/sh

if [ $# -ne 5 ] ; then
   echo "$0 <matlab dir> <out file> <Nstage> <Nlevel> <Block Size>"
   exit 1
fi

CURDIR=`dirname $0`
MATLAB_DIR=$1
# CHM parameters
OUT_FILE=$2
NSTAGE=$3
NLEVEL=$4
BLOCKSIZE=$5

OUTPUT="`pwd`/out"

# set library path for matlab
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PANFISH_BASEDIR/$MATLAB_DIR/bin/glnxa64

MATLAB_BIN="$PANFISH_BASEDIR/$MATLAB_DIR/bin/matlab"

$MATLAB_BIN -nodesktop -nodisplay -nojvm -r 'TrainScript_testBlocks('\'input\',\'${OUTPUT}\',\'${OUT_FILE}\',${NSTAGE},${NLEVEL},${BLOCKSIZE}');'

EXIT_CODE=$?

exit $EXIT_CODE
