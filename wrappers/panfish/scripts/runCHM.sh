#!/bin/sh

if [ $# -ne 1 ] ; then
   echo "$0 <Matlab directory>"
   echo "This script runs CHM on a slice/image of data"
   echo "The parameters are set by examining the"
   echo "environment variable SGE_TASK_ID and parsing"
   echo "$RUN_CHM_CONFIG file for inputs corresponding"
   echo "to SGE_TASK_ID."
   echo ""
   echo " <Matlab directory> is the directory where"
   echo "Matlab is installed.  This script assumes"
   echo "<Matlab directory>/bin/matlab is the path"
   echo "to the matlab binary"
   exit 1

fi

MATLAB_DIR=$1

###########################################################
#
# Functions
#
###########################################################

#
# Copies gzip tarball of inputs to $SCRATCH/inputs 
# decompressing the tarball
#
function copyInputsToScratch {
  logStartTime "CopyInputsToScratch"

  getSizeOfPath $BASEDIR/CHM.tar.gz

  logMessage "Copying $NUM_BYTES bytes to scratch"

  /bin/cp $BASEDIR/CHM.tar.gz $1

  if [ $? != 0 ] ; then
     jobFailed "Unable to run: /bin/cp $BASEDIR/CHM.tar.gz $1"
  fi
  
  cd $1

  if [ $? != 0 ] ; then
    jobFailed "Unable to run: cd $1"
  fi

  logStartTime "Uncompressing CHM.tar.gz"

  tar -zxf CHM.tar.gz

  if [ $? != 0 ] ; then
     jobFailed "Unable to run: tar -zxf CHM.tar.gz"
  fi

  /bin/rm -f CHM.tar.gz

  if [ $? != 0 ] ; then
     logWarning "Unable to run /bin/rm -f CHM.tar.gz"
  fi
  
  logEndTime "Uncompressing CHM.tar.gz" $START_TIME 0

  /bin/cp $BASEDIR/${RUN_CHM_CONFIG} $SCRATCH/.

  if [ $? != 0 ] ; then
     jobFailed "Unable to run: /bin/cp $BASEDIR/${RUN_CHM_CONFIG} $SCRATCH/."
  fi

  makeDirectory $SCRATCH/CHM/input

  logStartTime "Copying $INPUT_IMAGE to $SCRATCH/CHM/input/"

  /bin/cp $INPUT_IMAGE $SCRATCH/CHM/input/.

  if [ $? != 0 ] ; then
     jobFailed "Unable to run: /bin/cp $INPUT_IMAGE $SCRATCH/CHM/input/."
  fi

  logEndTime "Copying $INPUT_IMAGE to $SCRATCH/CHM/input/" $START_TIME 0

  logEndTime "CopyInputsToScratch" $START_TIME 0
}






###########################################################
#
# Start of script
#
###########################################################

SCRIPT_DIR=`dirname $0`

. $SCRIPT_DIR/helperfuncs.sh

# Dont allow job to run if SGE_TASK_ID is NOT set
if [ -z "$SGE_TASK_ID" ] ; then
  jobFailed "Variable SGE_TASK_ID must be set to a numeric value"
fi

logStartTime "runCHM.sh" 
CHM_START_TIME=$START_TIME

BASEDIR=`pwd`

# set SCRATCH variable to /tmp/blast.# or to whatever $PANFISH_SCRATCH/blast.#
# if PANFISH_SCRATCH variable is not empty
UUID=`uuidgen`
SCRATCH="/tmp/chm.${UUID}"
if [ -n "$PANFISH_SCRATCH" ] ; then
  SCRATCH="$PANFISH_SCRATCH/chm.${UUID}"
fi

makeDirectory $SCRATCH


INPUT_IMAGE=$PANFISH_BASEDIR/`egrep "^${SGE_TASK_ID}:::" $RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 1`
CHMOPTS=`egrep "^${SGE_TASK_ID}:::" $RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 2 | tail -n 1`

OUTPUT_IMAGE_NAME=`echo $INPUT_IMAGE | sed "s/^.*\///"`
OUTPUT_IMAGE="$SCRATCH/CHM/out/${OUTPUT_IMAGE_NAME}"

LOG_FILE="$SCRATCH/CHM/out/${OUTPUT_IMAGE_NAME}.log"

# copy inputs
copyInputsToScratch $SCRATCH


# run runjob.sh
logStartTime "CHM_test.sh"

logMessage "Writing runjob.sh output to $LOG_FILE"

cd $SCRATCH/CHM

if [ $? != 0 ] ; then
  jobFailed "Unable to run cd $SCRATCH/CHM"
fi

echo "Job.Task:  ${JOB_ID}.${SGE_TASK_ID}" > $LOG_FILE

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PANFISH_BASEDIR/$MATLAB_DIR/bin/glnxa64
export MATLAB_BIN_DIR="$PANFISH_BASEDIR/$MATLAB_DIR/bin"
export PATH=$PATH:$MATLAB_BIN_DIR

$SCRATCH/CHM/CHM_test.sh $INPUT_IMAGE $SCRATCH/CHM/out -m $SCRATCH/CHM/out $CHMOPTS -s >> $LOG_FILE 2>&1

EXIT_CODE=$?

cd $BASEDIR

if [ $? != 0 ] ; then
  jobFailed "Unable to run cd $BASEDIR"
fi

logEndTime "CHM_test.sh" $START_TIME $EXIT_CODE


logStartTime "Copying back $OUTPUT_IMAGE_NAME"

CHM_EXIT_CODE=0

# copy back results
if [ ! -e "$OUTPUT_IMAGE" ] ; then
  logWarning "No $OUTPUT_IMAGE_NAME Found to copy"
  CHM_EXIT_CODE=1
else
  getSizeOfPath $OUTPUT_IMAGE
  logMessage "$OUTPUT_IMAGE is $NUM_BYTES bytes"

  /bin/cp $OUTPUT_IMAGE $BASEDIR/out/.
  if [ $? != 0 ] ; then
     logWarning "Error running /bin/cp $Y $BASEDIR/out/."
     CHM_EXIT_CODE=1
  fi
fi

logEndTime "Copying back $OUTPUT_IMAGE_NAME" $START_TIME $CHM_EXIT_CODE

logStartTime "Copying back ${SGE_TASK_ID}.log file"
LOG_COPY_EXIT=1
if [ -e "$LOG_FILE" ] ; then
   getSizeOfPath $LOG_FILE
   logMessage "$LOG_FILE is $NUM_BYTES bytes"
  /bin/cp $LOG_FILE $BASEDIR/out/log/.
  LOG_COPY_EXIT=$?
  if [ $LOG_COPY_EXIT != 0 ] ; then
     logWarning "Error running /bin/cp $LOG_FILE $BASEDIR/out/log/."
  fi
fi
logEndTime "Copying back ${SGE_TASK_ID}.log file" $START_TIME $LOG_COPY_EXIT

# delete tmp directory
logStartTime "rm $SCRATCH"
/bin/rm -rf $SCRATCH

EXIT_CODE=$?

logEndTime "rm $SCRATCH" $START_TIME $EXIT_CODE

logEndTime "runCHM.sh" $CHM_START_TIME $CHM_EXIT_CODE

exit 0
