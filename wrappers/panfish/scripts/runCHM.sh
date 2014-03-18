#!/bin/bash



###########################################################
#
# Functions
#
###########################################################

#
# Given a task id this function gets job parameters set as
# the following variables
# INPUT_IMAGE
# MODEL_DIR
# CHM_OPTS
# OUTPUT_IMAGE
# If the config file does not exist or there was a problem parsing
# function returns with non zero exit code
#
function getCHMTestJobParametersForTaskFromConfig {
  local jobDir=$1
  local taskId=$2

  getParameterForTaskFromConfig "$taskId" "1" "$jobDir/$RUN_CHM_CONFIG"
  if [ $? != 0 ] ; then
    return 1
  fi
  INPUT_IMAGE=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "2" "$jobDir/$RUN_CHM_CONFIG"
  if [ $? != 0 ] ; then
    return 2
  fi
  MODEL_DIR=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "3" "$jobDir/$RUN_CHM_CONFIG"
  if [ $? != 0 ] ; then
    return 3
  fi
  CHM_OPTS=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "4" "$jobDir/$RUN_CHM_CONFIG"
  if [ $? != 0 ] ; then
    return 4
  fi
  OUTPUT_IMAGE=$TASK_CONFIG_PARAM
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
  echo "This script runs CHM on a slice/image of data.

What is run is defined by the environment variable SGE_TASK_ID.
The script uses SGE_TASK_ID and extracts the job from $RUN_CHM_CONFIG
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
if [ ! -s "$SCRIPT_DIR/$RUN_CHM_CONFIG" ] ; then
  jobFailed "No $SCRIPT_DIR/$RUN_CHM_CONFIG found"
fi


logStartTime "$RUN_CHM_SH" 

declare -r runChmStartTime=$START_TIME


# set SCRATCH variable to /tmp/chm.# or to whatever $PANFISH_SCRATCH/chm.#
# if PANFISH_SCRATCH variable is not empty
declare uuidVal=`uuidgen`
declare scratchDir="/tmp/chm.${uuidVal}"
if [ -n "$PANFISH_SCRATCH" ] ; then
  scratchDir="$PANFISH_SCRATCH/chm.${uuidVal}"
fi


getCHMTestJobParametersForTaskFromConfig "$SCRIPT_DIR" "${SGE_TASK_ID}"
if [ $? != 0 ] ; then
  logEndTime "$RUN_CHM_SH" $runChmStartTime 1
  exit 1
fi


# Parse the config for job parameters

declare -r inputImage="$PANFISH_BASEDIR/$INPUT_IMAGE"
declare -r inputImageName=`echo $inputImage | sed "s/^.*\///"`
declare -r modelDir="$PANFISH_BASEDIR/$MODEL_DIR"
declare -r chmOpts="$CHM_OPTS"
declare -r finalImage="$SCRIPT_DIR/$OUTPUT_IMAGE"

declare compiledMatlabFlag=""


makeDirectory "$scratchDir"

# Set environment variables if we are NOT using compiled matlab
# we make this determination by looking for CHM_TEST_BLOCKS_BINARY
# in the chm bin directory.  If that file is there we will use that
if [ ! -s "$PANFISH_BASEDIR/$CHM_BIN_DIR/$CHM_TEST_BINARY" ] ; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PANFISH_BASEDIR}/${MATLAB_DIR}/bin/glnxa64
  export MATLAB_BIN_DIR="${PANFISH_BASEDIR}/${MATLAB_DIR}/bin"
  export PATH=$PATH:$MATLAB_BIN_DIR
else
  compiledMatlabFlag="-M ${PANFISH_BASEDIR}/${MATLAB_DIR}"
fi

# Todo: Need to make this task run in the background and then check on job completion
#       this will allow the script to catch KILL requests etc.. and to track memory
#       usage etc.
/usr/bin/time -v $PANFISH_BASEDIR/$CHM_BIN_DIR/$CHM_TEST_SH "$inputImage" "$scratchDir" -m "$modelDir" $chmOpts -s $compiledMatlabFlag

declare chmExitCode=$?

# copy back image to final directory
if [ ! -s "$scratchDir/$inputImageName" ] ; then
  logWarning "No $scratchDir/$inputImageName found to copy"
  chmExitCode=2
else 
  /bin/cp -f "$scratchDir/$inputImageName" "$finalImage"
  if [ $? != 0 ] ; then
    logWarning "Error running /bin/cp -f \"$scratchDir/$inputImageName\" \"$finalImage\""
    chmExitCode=3
  fi
fi

# remove Scratch Directory
if [ -d "$scratchDir" ] ; then
  /bin/rm -rf "$scratchDir"
fi

logEndTime "$RUN_CHM_SH" $runChmStartTime $chmExitCode

exit $chmExitCode
