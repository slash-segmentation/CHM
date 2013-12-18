#!/bin/sh

if [ $# -ne 1 ] ; then
   echo "$0 <Matlab directory>"
   echo "This script runs CHM train on a training set"
   echo "It is assumed the training data resides
   echo "in train and label folders of CHM source
   echo "tree."
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

  logEndTime "CopyInputsToScratch" $START_TIME 0
}






###########################################################
#
# Start of script
#
###########################################################

SCRIPT_DIR=`dirname $0`

. $SCRIPT_DIR/helperfuncs.sh

logStartTime "runCHMtrain.sh" 
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


TRAIN_DATA=$SCRATCH/CHM/train
LABEL_DATA=$SCRATCH/CHM/label
OUTPUT_DIR=$SCRATCH/CHM/out
NSTAGE=`egrep "^1:::" $RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 3 | tail -n 1`
NLEVEL=`egrep "^1:::" $RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 4 | tail -n 1`


LOG_FILE="$SCRATCH/CHM/out/train.log"

# copy inputs
copyInputsToScratch $SCRATCH


# run runjob.sh
logStartTime "runtrainjob.sh"

logMessage "Writing runtrainjob.sh output to $LOG_FILE"

cd $SCRATCH/CHM

if [ $? != 0 ] ; then
  jobFailed "Unable to run cd $SCRATCH/CHM"
fi

echo "Job.Task:  ${JOB_ID}.${SGE_TASK_ID}" > $LOG_FILE

$SCRATCH/CHM/runtrainjob.sh $MATLAB_DIR $TRAIN_DATA $LABEL_DATA $OUTPUT_DIR $NSTAGE $NLEVEL >> $LOG_FILE 2>&1

CHM_EXIT_CODE=$?

cd $BASEDIR

if [ $? != 0 ] ; then
  jobFailed "Unable to run cd $BASEDIR"
fi

logEndTime "runtrainjob.sh" $START_TIME $CHM_EXIT_CODE


logStartTime "Copying back CHM.tar.gz"


# copy back results
if [ ! -d "$OUTPUT_DIR" ] ; then
  logWarning "No $OUTPUT_DIR Found to copy"
  CHM_EXIT_CODE=1
else
  getSizeOfPath $OUTPUT_DIR
  logMessage "$OUTPUT_DIR is $NUM_BYTES bytes"

  cd $SCRATCH

  if [ $? != 0 ] ; then
     jobFailed "Unable to cd to $SCRATCH"
  fi  

  tar -cz CHM > CHM.tar.gz

  /bin/cp  CHM.tar.gz $BASEDIR/.
  if [ $? != 0 ] ; then
     logWarning "Error running /bin/cp CHM.tar.gz $BASEDIR/."
     CHM_EXIT_CODE=1
  fi
fi

logEndTime "Copying back CHM.tar.gz" $START_TIME $CHM_EXIT_CODE

# delete tmp directory
logStartTime "rm $SCRATCH"
/bin/rm -rf $SCRATCH

EXIT_CODE=$?

logEndTime "rm $SCRATCH" $START_TIME $EXIT_CODE

logEndTime "runCHMtrain.sh" $CHM_START_TIME $CHM_EXIT_CODE

exit $CHM_EXIT_CODE
