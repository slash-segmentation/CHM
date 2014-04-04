#!/bin/bash


###########################################################
#
# Functions
#
###########################################################

# 
# Given a task id this function gets merge tile parameters from
# the merge tiles configuration file.  Upon success 0 is returned
# and INPUT_TILE_DIR is set to the input tile directory and OUTPUT_IMAGE
# is set to the full path to out image
#
function getConvertJobParametersForTaskFromConfig {
  local jobDir=$1
  local taskId=$2
  local convertConfig=$3

  getParameterForTaskFromConfig "$taskId" "1" "$jobDir/$convertConfig"
  if [ $? != 0 ] ; then
    return 1
  fi
  INPUT_IMAGE=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "2" "$jobDir/$convertConfig"
  if [ $? != 0 ] ; then
    return 3
  fi
  CONVERT_OPTS=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "3" "$jobDir/$convertConfig"
  if [ $? != 0 ] ; then
    return 3
  fi
  OUTPUT_IMAGE=$TASK_CONFIG_PARAM
  return 0
}


function usage {
  echo "This script runs convert program with options specified by caller on images

What is run is defined by the environment variable SGE_TASK_ID.
The script uses SGE_TASK_ID and extracts the job from $RUN_CONVERT_CONFIG
file residing in the same directory as this script.

The script also examines $PANFISH_CHM_PROPS for configuration.
This script will return 0 exit code upon success otherwise failure. 
"

  exit 1
}

###########################################################
#
# Start of script
#
###########################################################
# Check if caller just wants to source this file for testing purposes
declare convertConfig=""
if [ $# -eq 1 ] ; then
  if [ "$1" == "source" ] ; then
    return 0
  fi
  convertConfig="$1"
else
  usage
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

if [ $? != 0 ] ; then
  echo "Error parsing properties"
  exit 2
fi

# Dont allow job to run if SGE_TASK_ID is NOT set
if [ -z "$SGE_TASK_ID" ] ; then
  usage
fi

# Bail if no config is found
if [ ! -s "$SCRIPT_DIR/$convertConfig" ] ; then
  jobFailed "No $SCRIPT_DIR/$convertConfig found"
fi

logStartTime "$RUN_CONVERT_SH" 

declare -r runConvertStartTime=$START_TIME

getConvertJobParametersForTaskFromConfig "$SCRIPT_DIR" "${SGE_TASK_ID}" "$convertConfig"
if [ $? != 0 ] ; then
  logEndTime "$RUN_CONVERT_SH" $runConvertStartTime 1
  exit 1
fi

# Parse the config for job parameters
declare -r inputImage="$PANFISH_BASEDIR/$INPUT_IMAGE"
declare -r convertOpts="$CONVERT_OPTS"
declare -r finalImage="$SCRIPT_DIR/$OUTPUT_IMAGE"

# Todo: Need to make this task run in the background and then check on job completion
#       this will allow the script to catch KILL requests etc.. and to track memory
#       usage etc.

declare cmd="$CONVERT_CMD $inputImage $convertOpts $finalImage"

logMessage "Running $cmd"

$TIME_V_CMD $cmd
convertExitCode=$?

logEndTime "$RUN_CONVERT_SH" $runConvertStartTime $convertExitCode

exit $convertExitCode
