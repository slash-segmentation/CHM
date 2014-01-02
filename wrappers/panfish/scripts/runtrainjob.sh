#!/bin/sh

if [ $# -ne 6 ] ; then
   echo "$0 <matlab dir> <train path> <label path> <out file> <Nstage> <Nlevel>"
   exit 1
fi

CURDIR=`dirname $0`
MATLAB_DIR=$1
# CHM parameters
TRAIN_PATH=$2
LABEL_PATH=$3
OUT_FILE=$4
NSTAGE=$5
NLEVEL=$6

# set library path for matlab
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PANFISH_BASEDIR/$MATLAB_DIR/bin/glnxa64

MATLAB_BIN="$PANFISH_BASEDIR/$MATLAB_DIR/bin/matlab"

$MATLAB_BIN -nodesktop -nodisplay -nojvm -r 'TrainScript_train('\'${TRAIN_PATH}\',\'${LABEL_PATH}\',\'${OUT_FILE}\',${NSTAGE},${NLEVEL}');'

EXIT_CODE=$?

exit $EXIT_CODE
