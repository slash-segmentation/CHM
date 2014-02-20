#!/bin/sh

if [ $# -ne 1 ] ; then
  echo "$0 <Directory containing matlab binary>
This script runs CHM on a slice/image of data.
The parameters are set by examining the
environment variable SGE_TASK_ID and parsing
$RUN_CHM_CONFIG file for inputs corresponding
to SGE_TASK_ID.

Matlab directory> is the directory where
Matlab is installed.  This script assumes
<Matlab directory>/bin/matlab is the path
to the matlab binary"
   
  exit 1
fi

MATLAB_DIR=$1

###########################################################
#
# Functions
#
###########################################################


#
# Removes scratch directory if it exists
#
function removeScratch {
  logStartTime "rm $SCRATCH"
  if [ ! -d $SCRATCH ] ; then
    logMessage "$SCRATCH is not a directory"
    return 0
  fi

  /bin/rm -rf $SCRATCH
  EXIT_CODE=$?
  logEndTime "rm $SCRATCH" $START_TIME $EXIT_CODE
  return $EXIT_CODE
}

#
# function called when USR2 signal is caught
#
on_usr2() {
  removeScratch
  jobFailed "USR2 signal caught exiting.."
}

# trap usr2 signal cause its what gets sent by SGE when qdel is called
trap 'on_usr2' USR2

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

declare finalImage=`egrep "^${SGE_TASK_ID}:::" $RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 3 | tail -n 1`
declare finalImageName=`echo $finalImage | sed "s/^.*\///"`

declare tempResultImageName=`echo $INPUT_IMAGE | sed "s/^.*\///"`
declare tempResultImage="$SCRATCH/${tempResultImageName}"

LOG_FILE="$SCRATCH/${finalImageName}.log"

# run CHM_test.sh
logStartTime "CHM_test.sh"

logMessage "Writing CHM_test.sh output to $LOG_FILE"

echo "Job.Task:  ${JOB_ID}.${SGE_TASK_ID}" > $LOG_FILE

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PANFISH_BASEDIR/$MATLAB_DIR/bin/glnxa64
export MATLAB_BIN_DIR="$PANFISH_BASEDIR/$MATLAB_DIR/bin"
export PATH=$PATH:$MATLAB_BIN_DIR

$BASEDIR/CHM/CHM_test.sh $INPUT_IMAGE $SCRATCH -m $BASEDIR/CHM/out $CHMOPTS -s >> $LOG_FILE 2>&1

EXIT_CODE=$?

logEndTime "CHM_test.sh" $START_TIME $EXIT_CODE

logStartTime "Copying back $finalImageName"

CHM_EXIT_CODE=0

# copy back results
if [ ! -e "$tempResultImage" ] ; then
  logWarning "No $tempResultImage Found to copy"
  CHM_EXIT_CODE=1
else
  getSizeOfPath $tempResultImage
  logMessage "$tempResultImage is $NUM_BYTES bytes"

  /bin/cp $tempResultImage $BASEDIR/${finalImage}
  if [ $? != 0 ] ; then
     logWarning "Error running /bin/cp $Y $BASEDIR/${finalImage}"
     CHM_EXIT_CODE=1
  fi
fi

logEndTime "Copying back $finalImageName" $START_TIME $CHM_EXIT_CODE

logStartTime "Copying back ${SGE_TASK_ID}.log file"
LOG_COPY_EXIT=1
if [ -e "$LOG_FILE" ] ; then
   getSizeOfPath $LOG_FILE
   logMessage "$LOG_FILE is $NUM_BYTES bytes"
  /bin/cp $LOG_FILE $BASEDIR/out/log/chm/.
  LOG_COPY_EXIT=$?
  if [ $LOG_COPY_EXIT != 0 ] ; then
     logWarning "Error running /bin/cp $LOG_FILE $BASEDIR/out/log/chm/."
  fi
fi
logEndTime "Copying back ${SGE_TASK_ID}.log file" $START_TIME $LOG_COPY_EXIT


# delete tmp directory
removeScratch

logEndTime "runCHM.sh" $CHM_START_TIME $CHM_EXIT_CODE

exit 0
