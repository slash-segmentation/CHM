#!/bin/bash



###########################################################
#
# Functions
#
###########################################################

#
# Given a task id this function gets job parameters set as
# the following variables
# INPUT_IMAGES
# INPUT_LABELS
# TRAIN_OPS
# OUTPUT_DIR
# If the config file does not exist or there was a problem parsing
# function returns with non zero exit code
#
function getCHMTrainJobParametersForTaskFromConfig {
  local jobDir=$1
  local taskId=$2

  getParameterForTaskFromConfig "$taskId" "1" "$jobDir/$RUN_CHM_TRAIN_CONFIG"
  if [ $? != 0 ] ; then
    return 1
  fi
  INPUT_IMAGES=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "2" "$jobDir/$RUN_CHM_TRAIN_CONFIG"
  if [ $? != 0 ] ; then
    return 2
  fi
  INPUT_LABELS=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "3" "$jobDir/$RUN_CHM_TRAIN_CONFIG"
  if [ $? != 0 ] ; then
    return 3
  fi
  TRAIN_OPTS=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "4" "$jobDir/$RUN_CHM_TRAIN_CONFIG"
  if [ $? != 0 ] ; then
    return 4
  fi
  OUTPUT_DIR=$TASK_CONFIG_PARAM
  return 0
}

###########################################################
#
# Start of script
#
###########################################################

# Check if caller just wants to source this file for testing purposes
if [ $# -eq 1 ] ; then
  if [ "$1" == "source" ] ; then
    return 0
  fi
fi

# Look for the helper functions file and bail if not found
SCRIPT_DIR=`dirname $0`
if [ ! -s "$SCRIPT_DIR/.helperfuncs.sh" ] ; then
  echo "$SCRIPT_DIR/.helperfuncs.sh not found" 1>&2
  exit 1
fi

# Source the helper functions file
. $SCRIPT_DIR/.helperfuncs.sh

# Parse the properties file
parseProperties "$SCRIPT_DIR" "$SCRIPT_DIR"

function usage {
  echo "This script runs CHM train on a set of images and labels/maps.

What is run is defined by the environment variable SGE_TASK_ID.
The script uses SGE_TASK_ID and extracts the job from $RUN_CHM_TRAIN_CONFIG
file residing in the same directory as this script.

The script also examines $PANFISH_CHM_PROPS for configuration that is
needed such as path to matlab.  This script will return 0 exit
code upon success otherwise failure. 
 
"
  exit 1
}

# Dont allow job to run if SGE_TASK_ID is NOT set
if [ -z "$SGE_TASK_ID" ] ; then
  usage
fi

# Bail if no config is found
if [ ! -s "$SCRIPT_DIR/$RUN_CHM_TRAIN_CONFIG" ] ; then
  jobFailed "No $SCRIPT_DIR/$RUN_CHM_TRAIN_CONFIG found"
fi

logStartTime "$RUN_CHM_TRAIN_SH" 

declare -r runChmTrainStartTime=$START_TIME

getCHMTrainJobParametersForTaskFromConfig "$SCRIPT_DIR" "${SGE_TASK_ID}"
if [ $? != 0 ] ; then
  logEndTime "$RUN_CHM_TRAIN_SH" $runChmStartTime 1
  exit 1
fi


# Parse the config for job parameters

declare -r inputImages="$PANFISH_BASEDIR/$INPUT_IMAGES"
declare -r inputLabels="$PANFISH_BASEDIR/$INPUT_LABELS"
declare -r trainOpts="$TRAIN_OPTS"
declare -r outputDir="$SCRIPT_DIR/$OUTPUT_DIR"

declare compiledMatlabFlag="-s"

# Set environment variables if we are NOT using compiled matlab
# we make this determination by looking for CHM_TEST_BLOCKS_BINARY
# in the chm bin directory.  If that file is there we will use that
if [ ! -s "$PANFISH_BASEDIR/$CHM_BIN_DIR/$CHM_TRAIN_BINARY" ] ; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PANFISH_BASEDIR}/${MATLAB_DIR}/bin/glnxa64
  export MATLAB_BIN_DIR="${PANFISH_BASEDIR}/${MATLAB_DIR}/bin"
  export PATH=$PATH:$MATLAB_BIN_DIR
else
  compiledMatlabFlag="-M ${PANFISH_BASEDIR}/${MATLAB_DIR}"
fi

# Todo: Need to make this task run in the background and then check on job completion
#       this will allow the script to catch KILL requests etc.. and to track memory
#       usage etc.
/usr/bin/time -v $PANFISH_BASEDIR/$CHM_BIN_DIR/$CHM_TRAIN_SH "$inputImages" "$inputLabels" -m "$outputDir" $trainOpts $compiledMatlabFlag

declare chmTrainExitCode=$?

logEndTime "$RUN_CHM_TRAIN_SH" $runChmTrainStartTime $chmTrainExitCode

exit $chmTrainExitCode
