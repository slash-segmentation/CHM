#!/bin/bash

#
# Variables used by scripts
#
#
declare -r RUN_CHM_SH="runCHM.sh"
declare -r RUN_CHM_JOB_VIA_PANFISH_SH="runCHMJobViaPanfish.sh"
declare -r HELPER_FUNCS_SH="helperfuncs.sh"
declare -r RUN_CHM_CONFIG="runCHM.sh.config"
declare -r CAST_OUT_FILE="cast.out"
declare -r CHUM_OUT_FILE="chum.out"
declare -r PANFISH_CHM_PROPS="panfishCHM.properties"
declare -r OUT_DIR_NAME="out"
declare -r CONFIG_DELIM=":::"

function parseProperties {

  local scriptDir="$1"
  
  local outputDir="$2"

  local theConfig="$outputDir/$PANFISH_CHM_PROPS"

  # Make sure we have a config file to use
  if [ ! -s "$theConfig" ] ; then
    theConfig="$scriptDir/$PANFISH_CHM_PROPS"
  fi

  if [ ! -s "$theConfig" ] ; then
    logMessage "Config $theConfig not found"
    return 1
  fi

  # if set must end with /
  PANFISH_BIN_DIR=`egrep "^panfish.bin.dir" $theConfig | sed "s/^.*= *//"`

  MATLAB_DIR=`egrep "^matlab.dir" $theConfig | sed "s/^.*= *//"`
  BATCH_AND_WALLTIME_ARGS=`egrep "^batch.and.walltime.args" $theConfig | sed "s/^.*= *//"`

  CHUMMEDLIST=`egrep "^cluster.list" $theConfig | sed "s/^.*= *//"`

  MAX_RETRIES=5
  RETRY_SLEEP=100
  CASTBINARY="${PANFISH_BIN_DIR}panfishcast"
  CHUMBINARY="${PANFISH_BIN_DIR}panfishchum"
  LANDBINARY="${PANFISH_BIN_DIR}panfishland"
  PANFISHSTATBINARY="${PANFISH_BIN_DIR}panfishstat"
  return 0
}

function getLabelForLogMessage {
  LOG_LABEL=""

  JOB_NAME=""
  if [ -n "$JOB_ID" ] ; then
     JOB_NAME="${JOB_ID}."
     LOG_LABEL="(job $JOB_ID)"
  fi

  if [ -n "$SGE_TASK_ID" ] ; then
    LOG_LABEL="(task ${JOB_NAME}${SGE_TASK_ID})"
  fi
}

#
# This function outputs an error message to standard error and exits with an exit code of 1
# The message is format of ERROR: (first argument passed to function)
#
# jobFailed($1 is program)
#
function jobFailed {
  getLabelForLogMessage
  logEcho ""
  logEcho ""
  logEcho "ERROR:  $LOG_LABEL  $1"
  logEcho ""
  exit 1
}

#
# This function outputs a warning message
# logWarning($1 is identifier,$2 is message)
#
function logWarning {
  getLabelForLogMessage
  logEcho ""
  logEcho "WARNING:  $LOG_LABEL  $1"
  logEcho ""
  
}

#
# This function outputs a log message to standard out
# logMessage ($1 is identifier,$2 is message)
#
function logMessage {
  getLabelForLogMessage 
  logEcho "$LOG_LABEL  $1"
}

# 
# Writes empty line
#
function logNewLine {
  logEcho ""
}

#
# This function runs du on path given and sets NUM_BYTES variable to size 
# of file or -1 if file is not found
#
function getSizeOfPath {
  if [ ! -e $1 ] ; then
     NUM_BYTES="-1"
     return 1
  fi

  NUM_BYTES=`du $1 -bs | sed "s/\W.*//"`  
  return 0
}

#
#
# Sets LOG_FILE
#
function getSingleCHMTaskLogFile {
  # $1 - Job Directory
  # $2 - Task Id

  JOBDIR=$1
  TASKID=$2

  if [ ! -d "$JOBDIR" ] ; then
    logWarning "$JOBDIR is not a directory"
    return 1
  fi

  if [ ! -s "$JOBDIR/$RUN_CHM_CONFIG" ] ; then
    logWarning "$JOBDIR/$RUN_CHM_CONFIG configuration file not found"
    return 1
  fi

  OUT_IMAGE_RAW=`egrep "^${TASKID}${CONFIG_DELIM}" $JOBDIR/$RUN_CHM_CONFIG | head -n 1 | sed "s/^.*${CONFIG_DELIM}//"`

  LOG_FILE_NAME=`echo $OUT_IMAGE_RAW | sed "s/^.*\///"`

  LOG_FILE="$JOBDIR/out/log/chm/${LOG_FILE_NAME}.log"

  if [ ! -s "$LOG_FILE" ]  ; then
     return 1
  fi
  return 0
}

#
#
#
function getFatalExceptionFromSingleCHMTask {
  # $1 - Job Directory
  # $2 - Task Id

  JOBDIR=$1
  TASKID=$2

  getSingleCHMTaskLogFile $JOBDIR $TASKID
  if [ $? != 0 ] ; then
    return 1
  fi

  FATAL_MESSAGE=""

  grep "Caught Fatal Exception" $LOG_FILE > /dev/null 2>&1
  if [ $? != 0 ] ; then
     return 0
  fi

  FATAL_MESSAGE=`cat $LOG_FILE | grep "Caught Fatal Exception" | sed "s/^Caught Fatal Exception://"`

  return 0
}


#
# This function parses log file from single CHM job and outputs
# its run time in seconds setting the variable RUNTIME_SECONDS
#
function getRunTimeOfSingleCHMTask {
  # $1 - Job Directory
  # $2 - Task Id
  RUNTIME_SECONDS=-1

  JOBDIR=$1
  TASKID=$2

  getSingleCHMTaskLogFile $JOBDIR $TASKID
  if [ $? != 0 ] ; then
    return 1
  fi

  grep "Running <<" $LOG_FILE > /dev/null 2>&1
  if [ $? != 0 ] ; then
     logWarning "Unable to parse runtime from $LOG_FILE for task $TASKID"
     return 1
  fi 
  RUNTIME_SECONDS=`cat $LOG_FILE | grep "Running <<" | sed "s/^.*took //" | sed "s/\..*//"`
  
  return 0
}

#
# Logs start time in seconds since epoch. Start time is also
# stored in START_TIME variable
#
# logStartTime($1 is program)
#
# Example:  
#
#   logStartTime "task 1" "hello"
#
# Output:
#
#   (task 1) hello 123354354
#
function logStartTime {
  getLabelForLogMessage
  START_TIME=`date +%s`
  logEcho "$LOG_LABEL $1 Start Time: $START_TIME"
}

#
# Logs end time duration and exit code.  
# logEndTime($1 is program, $2 is start time, $3 is the exit code to log)
#
function logEndTime {
  END_TIME=`date +%s`
  DURATION=`echo "$END_TIME-$2" | bc -l`
  getLabelForLogMessage
  logEcho "$LOG_LABEL $1 End Time: $END_TIME Duration: $DURATION Exit Code: $3"
}


#
#
#
#
function logEcho {
  if [ -n "$STDOUT" ] ; then
     THE_STD_OUT_DIR=`dirname $STDOUT`
     if [ -d "$THE_STD_OUT_DIR" ] ; then
        echo "$1" >> $STDOUT
     else
        echo "$1"
     fi
  else 
     echo "$1"
  fi
}

#
# Creates directory passed in via first argument.
# If creation fails script will exit with 1 code.
#
function makeDirectory {
  logMessage "Creating directory $1"
  /bin/mkdir -p $1
  if [ $? != 0 ] ; then
    jobFailed "ERROR: Unable to run /bin/mkdir -p $1"
  fi
}

#
# Gets the Full path of path passed in
#
#
function getFullPath {
  local thePath=$1
  local curdir=`pwd`
 
  cd $thePath 2> /dev/null
  if [ $? == 0 ] ; then
    GETFULLPATHRET=`pwd`
    cd $curdir
    return 0
  fi

  cd $curdir
 
  if [ $? == 0 ] ; then
    GETFULLPATHRET=$thePath
    return 0
  fi
  
  GETFULLPATHRET="$curdir/$thePath"
  
  return 0
}

#
# Checks for KILL.JOB.REQUEST file and if found that file
# is chummed to remote clusters before program exits
# checkForKillFile(<directory>)
#
function checkForKillFile {

    KILL_FILE=""
    if [ -e $1/KILL.JOB.REQUEST ] ; then
          KILL_FILE="$1/KILL.JOB.REQUEST"
    fi

    if [ "$KILL_FILE" != "" ] ; then
          logMessage "$KILL_FILE detected. Chumming kill file to remote cluster and exiting..."
          logMessage "Running $CHUMBINARY --path $KILL_FILE > $1/killed.chum.out"
          $CHUMBINARY --path $KILL_FILE > $1/killed.chum.out 2>&1

          logMessage "Running qdel"
          # lets also kill any running jobs if we see them by calling qdel
          qdel `cat $1/cast*out  sed "s/^.*job-array //" | sed "s/\..*//"`
          jobFailed "$KILL_FILE detected. Exiting..."
    fi
}

#
# Moves any cluster folders out of the way by renaming them with
# a #.old suffix
# moveOldClusterFolders $iteration $jobDirectory
#
function moveOldClusterFolders {
  if [ -n $CHUMMEDLIST ] ; then

     for Y in `echo "$CHUMMEDLIST" | sed "s/,/ /g"` ; do
         if [ -d "$2/$Y" ] ; then
            /bin/mv $2/$Y $2/${Y}.$1.old
         fi
     done
  fi
}

#
# gets number of jobs in config by looking at #::: of the last line
# of config file
# getNumberOfCHMJobsFromConfig $jobDirectory
#
function getNumberOfCHMJobsFromConfig {
  if [ ! -e "$1/$RUN_CHM_CONFIG" ] ; then
     logWarning "getNumberOfCHMJobsFromConfig - $1/$RUN_CHM_CONFIG not found"
     return 1
  fi
  NUMBER_JOBS=`tail -n 1 $1/$RUN_CHM_CONFIG | sed "s/${CONFIG_DELIM}.*//"`
  return 0
}

#
# Get input Image data directory
#
#
function getInputDataDirectory {
  if [ ! -e "$1/$RUN_CHM_CONFIG" ] ; then
     logWarning "getInputDataDirectory - $1/$RUN_CHM_CONFIG not found"
     return 1
  fi

  FIRST_INPUT_IMAGE=`head -n 1 $1/$RUN_CHM_CONFIG | sed "s/^.*${CONFIG_DELIM}//"`
  INPUT_DATA=`dirname $FIRST_INPUT_IMAGE`
  return 0
}


#     
# Download data in path
#
function landData {
  # $1 iteration
  # $2 cluster list
  # $3 path
  # $4 other args to pass to land
  logStartTime "LandData Iteration $1"
  LAND_RESULTS_START_TIME=$START_TIME

  LAND_OTHER_ARGS=""
  if [ $# -eq 4 ] ; then
    LAND_OTHER_ARGS=$4
  fi

  logMessage "Running $LANDBINARY --path $3 --cluster $2 $LAND_OTHER_ARGS"

  $LANDBINARY --path $3 --cluster $2 $LAND_OTHER_ARGS

  if [ $? != 0 ] ; then
    logWarning "Download failed"
    logEndTime "LandData Iteration $1" $LAND_RESULTS_START_TIME 1
    return 1
  fi

  logEndTime "LandData Iteration $1" $LAND_RESULTS_START_TIME 0
  return 0
}

#
# Upload data in path
#
function chumData {
  # $1 iteration
  # $2 cluster list
  # $3 path
  # $4 chum out file
  # $5 other args to pass to chum

  logStartTime "chumData Iteration $1"
  CHUM_DATA_START_TIME=$START_TIME

  CHUM_OTHER_ARGS=""
  if [ $# -eq 5 ] ; then
    CHUM_OTHER_ARGS=$5
  fi

  # bail if there are no clusters in the CHUMMEDLIST variable
  if [ "$2" == "" ] ; then
      logWarning "No clusters in cluster list"
      logEndTime "chumData Iteration $1" $CHUM_DATA_START_TIME 1
      return 1
  fi

  logMessage "Running $CHUMBINARY --listchummed --path $3 --cluster $2 $CHUM_OTHER_ARGS > $4 2>&1"
  $CHUMBINARY --listchummed --path $3 --cluster $2 $CHUM_OTHER_ARGS > $4 2>&1

  if [ $? != 0 ] ; then
     logWarning "Chum failed"
     logEndTime "chumData Iteration $1" $CHUM_DATA_START_TIME 1
     return 1
  fi

  # Update the CHUMMEDLIST with the clusters the job was uploaded too
  CHUMMEDLIST=`cat $4 | egrep "^chummed.clusters" | sed "s/^chummed.clusters=//"`
  
  logEndTime "chumData Iteration $1" $CHUM_DATA_START_TIME 0
  return 0
}

#
# getStatusOfJobsInCastOutFile $jobDirectory
# This function uses panfishstat to check status of jobs listed in $jobDirectory/cast.out file
# setting status in JOBSTATUS variable.
# 
#
function getStatusOfJobsInCastOutFile {
  
  JOBSTATUS="unknown"

  if [ ! -e "$1/$CAST_OUT_FILE" ] ; then
    logWarning "No $1/$CAST_OUT_FILE file found"
    return 2
  fi

  OUT=`$PANFISHSTATBINARY --statusofjobs $1/$CAST_OUT_FILE`
  if [ $? != 0 ] ; then
      logWarning "Error calling $PANFISHSTATBINARY --statusofjobs $1/$CAST_OUT_FILE will just keep waiting"
      return 1
  fi
      
  JOBSTATUS=`echo $OUT | egrep "^status=" | sed "s/^status=//" | tr \[:upper:\] \[:lower:\]`
  return 0
}

#
# Gets list of clusters either via a call to panfishcast or
# by parsing cluster.list file found under path passed in
# getInitialClusterList $jobDirectory
# This method will set CHUMMEDLIST variable
#
function getInitialClusterList {

   # If the file cluster.list exists obtain the suggested clusters from that
   # list otherwise just get the list by checking for clusters that have
   # matlab directory
   if [ -z "$CHUMMEDLIST" ] ; then
     if [ ! -d "$MATLAB_DIR" ] ; then
        logWarning "Unable to get Matlab directory"
        return 1
     fi

     if [ -n "$CHUMMEDLIST" ] ; then
       logMessage "Cluster list set to $CHUMMEDLIST"
       return 0
     fi

     logMessage "Running $CHUMBINARY --listexists --path $MATLAB_DIR"
     $CHUMBINARY --listexists --path $MATLAB_DIR > $1/check_for_matlab.out 2>&1

     if [ $? != 0 ] ; then
        logMessage "Unable to run $CHUMBINARY"
        return 1
     fi
     CHUMMEDLIST=`cat $1/check_for_matlab.out | egrep "^chummed.clusters" | sed "s/^chummed.clusters=//"`
   fi

   if [ "$CHUMMEDLIST" == "" ] ; then
     logWarning "No clusters found suitable to run on"
     return 1
   fi

   return 0
}

#
# This function creates a configuration file to run CHM on each tile of each
# image.  It is assumed the images all have the same size and need every tile
# processed.  The code will create an entry for every tile in this format:
# JOBID:::<full path to input image ie /home/foo/histeq_1.png>
# JOBID:::<chmOpts ie overlap> -t R,C 
# JOBID:::<output image path relative to job directory ie out/histeq_1.png/RxC.png>
#
function createConfig {
  local imageDir=$1
  local configFile=$2
  local tilesW=$3
  local tilesH=$4
  local tilesPerJob=$5
  local chmOpts=$6
  local imageSuffix=$7
  local cntr=0
  local outputFile=""

  # Calculate the tile sets we will be using
  local allTiles=() # simply a list of a tiles

  for c in `seq 1 $tilesW`; do
    for r in `seq 1 $tilesH`; do
      allTiles[$cntr]="-t $c,$r"
      let cntr++
    done
  done
  
  local tsCntr=0
  # Batch tiles by $tilesPerJob into new array named $tileSets
  let allTilesIndex=${#allTiles[@]}-1
 
  local tileSets=()
  while [ $allTilesIndex -ge 0 ] ; do
    for (( i=0 ; i < ${tilesPerJob} ; i++ )) ; do
      if [ $allTilesIndex -lt 0 ] ; then
        break
      fi
      tileSets[$tsCntr]="${tileSets[$tsCntr]} ${allTiles[$allTilesIndex]}"
      let allTilesIndex--
    done
    let tsCntr++
   
  done

  # Using tileSets array generate jobs for each image
  # Each job consists of 3 config lines
  # <JOBID>:::<Input image full path>
  # <JOBID>:::<CHM options and tile flags from tileSets>
  # <JOBID>:::<relative output path for image of format out/[image name]/[JOBID].[image suffix]
  let cntr=1
  for z in `find $imageDir -name "*.${imageSuffix}" -type f | sort -n` ; do
    imageName=`echo $z | sed "s/^.*\///"`
    for (( y=0 ; y < ${#tileSets[@]} ; y++ )) ; do
      echo "${cntr}${CONFIG_DELIM}${z}" >> "$configFile"
      echo "${cntr}${CONFIG_DELIM}${chmOpts} ${tileSets[$y]}" >> "$configFile"
      outputFile="${OUT_DIR_NAME}/${imageName}/${cntr}.${imageSuffix}" >> "$configFile"
      echo "${cntr}${CONFIG_DELIM}$outputFile" >> "$configFile"
      let cntr++
    done
  done

  return 0
}

# 
# Parses WxH parameter
# parseBlockParameter(String to parse, WIDTH_VARIABLE HEIGHT_VARIABLE)
# Upon success (0 return) WIDTH_VARIABLE and HEIGHT_VARIABLE will be set with values parsed out
# 
function parseWidthHeightParameter {
    local stringToParse=$1
    PARSED_WIDTH=-1
    PARSED_HEIGHT=-1
    if ! [[ $stringToParse =~ ^([0-9]+)(x([0-9]+))?$ ]]; then 
      logWarning "Unable to parse block parameter: :$stringToParse:"  
      return 1
    fi
    local -i w=${BASH_REMATCH[1]};
    local -i h=$w;

    if [[ ${#BASH_REMATCH[*]} -eq 4 ]]; then
      if [ -n "${BASH_REMATCH[3]}" ] ; then
        h=${BASH_REMATCH[3]};
      fi
    fi

    PARSED_WIDTH=$w
    PARSED_HEIGHT=$h
    return 0
}

#
# uses Image Magick identify command to get dimensions of image file
# getImageDimensions(path to image file)
# return 0 upon success and sets PARSED_WIDTH and PARSED_HEIGHT to values
function getImageDimensions {
  local image=$1

  if [ ! -f "${image}" ] ; then
     logWarning "${image} is not a file"
     return 1
  fi

  local identifyOutput=`identify -format '%wx%h' "${image}" 2>&1`
  # weird thing is in the unit tests identify never seems to kick back
  # a non zero exit code
  if [ $? -ne 0 ] ; then
     logWarning "Unable to run identify on image ${image}"
     return 1
  fi

  parseWidthHeightParameter "$identifyOutput"

  return $?
}

#
# Given a directory finds first image with matching suffix and
# obtains its dimensions.  Code returns 0 upon success otherwise failure.
# In addition, PARSED_WIDTH and PARSED_HEIGHT is set to dimensions of image
#
function getImageDimensionsFromDirOfImages {
  local imageDir=$1
  local imageSuffix=$2

  if [ ! -d "$imageDir" ] ; then
    logWarning "$imageDir is not a directory"
    return 1
  fi

  local anImage=`find $imageDir -name "*.${imageSuffix}" -type f | head -n 1` 

  if [ ! -f "$anImage" ] ; then
    logWarning "No images found in $imageDir with suffix $imageSuffix"
    return 1
  fi

  getImageDimensions "$anImage"
  
  return $?
}

#
# Given dimensions of an image along with tile size this function returns
# the number of tiles in horizontal (TILES_W) and vertical (TILES_H) 
# that are needed to cover the image
#
function calculateTilesFromImageDimensions {
  local width=$1
  local height=$2
  local blockWidth=$3
  local blockHeight=$4

  if [ $width -le 0 ] ; then
    logWarning "Width must be larger then 0"
    return 1
  fi

  if [ $height -le 0 ] ; then
    logWarning "Height must be larger then 0"
    return 1
  fi

  if [ $blockWidth -le 0 ] ; then
    logWarning "BlockWidth must be larger then 0"
    return 1
  fi  

  if [ $blockHeight -le 0 ] ; then
    logWarning "BlockHeight must be larger then 0"
    return 1
  fi

  TILES_W=`echo "scale=0;($width + $blockWidth - 1)/ $blockWidth" | bc -l`
  TILES_H=`echo "scale=0;($height + $blockHeight - 1)/ $blockHeight" | bc -l`
  return 0
}
