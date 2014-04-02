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
function getMergeTilesJobParametersForTaskFromConfig {
  local jobDir=$1
  local taskId=$2

  getParameterForTaskFromConfig "$taskId" "1" "$jobDir/$RUN_MERGE_TILES_CONFIG"
  if [ $? != 0 ] ; then
    return 1
  fi
  INPUT_TILE_DIR=$TASK_CONFIG_PARAM

  getParameterForTaskFromConfig "$taskId" "2" "$jobDir/$RUN_MERGE_TILES_CONFIG"
  if [ $? != 0 ] ; then
    return 2
  fi
  OUTPUT_IMAGE=$TASK_CONFIG_PARAM
  return 0
}


function usage {
  echo "This script merges image tiles for a slice/image of data.

What is run is defined by the environment variable SGE_TASK_ID.
The script uses SGE_TASK_ID and extracts the job from $RUN_MERGE_TILES_CONFIG
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

if [ $? != 0 ] ; then
  echo "Error parsing properties"
  exit 2
fi

# Dont allow job to run if SGE_TASK_ID is NOT set
if [ -z "$SGE_TASK_ID" ] ; then
  usage
fi

# Bail if no config is found
if [ ! -s "$SCRIPT_DIR/$RUN_MERGE_TILES_CONFIG" ] ; then
  jobFailed "No $SCRIPT_DIR/$RUN_MERGE_TILES_CONFIG found"
fi


logStartTime "$RUN_MERGE_TILES_SH" 

declare -r runMergeTilesStartTime=$START_TIME

# set SCRATCH variable to /tmp/chm.# or to whatever $PANFISH_SCRATCH/chm.#
# if PANFISH_SCRATCH variable is not empty
declare uuidVal=`$UUIDGEN_CMD`
declare scratchDir="/tmp/mergetiles.${uuidVal}"
if [ -n "$PANFISH_SCRATCH" ] ; then
  scratchDir="$PANFISH_SCRATCH/mergetiles.${uuidVal}"
fi

getMergeTilesJobParametersForTaskFromConfig "$SCRIPT_DIR" "${SGE_TASK_ID}"
if [ $? != 0 ] ; then
  logEndTime "$RUN_MERGE_TILES_SH" $runMergeTilesStartTime 1
  exit 1
fi

makeDirectory "$scratchDir"

# Parse the config for job parameters
declare -r tileDir="$PANFISH_BASEDIR/$INPUT_TILE_DIR"
declare -r finalImage="$SCRIPT_DIR/$OUTPUT_IMAGE"

declare -r tempImageName=`echo $finalImage | sed "s/^.*\///"`

declare tempOutFile="$scratchDir/$tempImageName"

# Todo: Need to make this task run in the background and then check on job completion
#       this will allow the script to catch KILL requests etc.. and to track memory
#       usage etc.

declare fileList=( `find $tileDir -name "*.*" -type f | sort` )

logMessage "Found ${#fileList[@]} tiles"

if [ ${#fileList[@]} -eq 0 ] ; then
  logMessage "No tiles to merge"
  # remove Scratch Directory
  if [ -d "$scratchDir" ] ; then
    $RM_CMD -rf "$scratchDir"
  fi
  logEndTime "$RUN_MERGE_TILES_SH" $runMergeTilesStartTime 9
  exit 9
fi

declare convertExitCode=0

# If there is only one tile simply copy that to scratch
if [ ${#fileList[@]} -eq 1 ] ; then
  $CP_CMD "${fileList[0]}" "$finalImage"
  convertExitCode=$?
else
  declare cmd="$CONVERT_CMD ${fileList[0]} -compose plus"

  for i in "${fileList[@]:1:${#fileList[@]}}" ; do
    cmd="${cmd} $i -composite"
  done

  cmd="${cmd} $tempOutFile"

  logMessage "Running $cmd"

  $TIME_V_CMD $cmd
  convertExitCode=$?


  # copy back image to final directory
  if [ ! -s "$scratchDir/$tempImageName" ] ; then
    logWarning "No $scratchDir/$tempImageName found to copy"
    let convertExitCode+=2
  else 
    $CP_CMD -f "$scratchDir/$tempImageName" "$finalImage"
    if [ $? != 0 ] ; then
      logWarning "Error running $CP_CMD -f \"$scratchDir/$tempImageName\" \"$finalImage\""
      let convertExitCode+=3
    fi
  fi
fi

# remove Scratch Directory
if [ -d "$scratchDir" ] ; then
  $RM_CMD -rf "$scratchDir"
fi

logEndTime "$RUN_MERGE_TILES_SH" $runMergeTilesStartTime $convertExitCode

exit $convertExitCode
