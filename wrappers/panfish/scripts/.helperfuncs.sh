#!/bin/bash

#
# Variables used by scripts
#
#
declare -r RUN_CHM_SH="runCHM.sh"
declare -r CHM_TEST_SH="CHM_test.sh"
declare -r RUN_CHM_VIA_PANFISH_SH="runCHMViaPanfish.sh"
declare -r HELPER_FUNCS_SH=".helperfuncs.sh"
declare -r RUN_CHM_CONFIG="runCHM.sh.config"
declare -r CHM_TEST_CAST_OUT_FILE="chm.test.cast.out"
declare -r CHUM_OUT_FILE="chum.out"
declare -r CHUM_IMAGE_OUT_FILE="chum.image.out"
declare -r CHUM_MODEL_OUT_FILE="chum.model.out"
declare -r KILLED_CHUM_OUT_FILE="killed.chum.out"
declare -r PANFISH_CHM_PROPS="panfishCHM.properties"
declare -r OUT_DIR_NAME="runchmout"
declare -r CONFIG_DELIM=":::"
declare -r JOB_ERR_DIR_NAME="joberr"
declare -r JOB_OUT_DIR_NAME="jobout"
declare -r CHM_OUT_DIR_NAME="chm"
declare -r STD_ERR_DIR_NAME="stderr"
declare -r STD_OUT_DIR_NAME="stdout"
declare -r STD_OUT_SUFFIX="stdout"
declare -r STD_ERR_SUFFIX="stderr"
declare -r JOBS_SUFFIX="jobs"
declare -r FAILED="failed"
declare -r FAILED_JOBS_TMP_FILE="${FAILED}.${JOBS_SUFFIX}.tmp"
declare -r FAILED_JOBS_FILE="${FAILED}.${JOBS_SUFFIX}"
declare -r OLD_SUFFIX="old"
declare -r OUT_SUFFIX="out"
declare -r IMAGE_TILE_DIR_SUFFIX="tiles"
declare -r KILL_JOB_REQUEST="KILL.JOB.REQUEST"
declare -r CHM_TEST_ITERATION_FILE="chm.test.iteration"
declare -r CHM_TEST_BINARY="CHM_test"
# Command defines
declare MV_CMD="/bin/mv"
declare RM_CMD="/bin/rm"

#
# Parses $PANFISH_CHM_PROPS properties file, setting several
# variables
#
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

  CHM_BIN_DIR=`egrep "^chm.bin.dir" $theConfig | sed "s/^.*= *//"`

  MAX_RETRIES=`egrep "^max.retries" $theConfig | sed "s/^.*= *//"`
  RETRY_SLEEP=`egrep "^retry.sleep" $theConfig | sed "s/^.*= *//"`

  WAIT_SLEEP_TIME=`egrep "^job.wait.sleep" $theConfig | sed "s/^.*= *//"`

  LAND_JOB_OPTS=`egrep "^land.job.options" $theConfig | sed "s/^.*= *//"`
  
  CHUM_JOB_OPTS=`egrep "^chum.job.options" $theConfig | sed "s/^.*= *//"`
  
  CHUM_IMAGE_OPTS=`egrep "^chum.image.options" $theConfig | sed "s/^.*= *//"`

  CHUM_MODEL_OPTS=`egrep "^chum.model.options" $theConfig | sed "s/^.*= *//"`

  CASTBINARY="${PANFISH_BIN_DIR}panfishcast"
  CHUMBINARY="${PANFISH_BIN_DIR}panfishchum"
  LANDBINARY="${PANFISH_BIN_DIR}panfishland"
  PANFISHSTATBINARY="${PANFISH_BIN_DIR}panfishstat"
  return 0
}

#
#
#
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
# Sets CHM_STD_OUT_FILE with path to standard out file for job
#
function getSingleCHMTestTaskStdOutFile {
  # $1 - Job Directory
  # $2 - Task Id

  local jobDir=$1
  local taskId=$2

  if [ ! -d "$jobDir" ] ; then
    logWarning "$jobDir is not a directory"
    return 1
  fi

  CHM_STD_OUT_FILE="$jobDir/$OUT_DIR_NAME/$STD_OUT_DIR_NAME/${taskId}.${STD_OUT_SUFFIX}"

  return 0
}

#
#
# Sets CHM_STD_ERR_FILE with path to standard err file for job
#
function getSingleCHMTestTaskStdErrFile {
  # $1 - Job Directory
  # $2 - Task Id

  local jobDir=$1
  local taskId=$2

  if [ ! -d "$jobDir" ] ; then
    logWarning "$jobDir is not a directory"
    return 1
  fi

  CHM_STD_ERR_FILE="$jobDir/$OUT_DIR_NAME/$STD_ERR_DIR_NAME/${taskId}.${STD_ERR_SUFFIX}"

  return 0
}


#
# Checks that a single task ran successfully by verifying an output
# image was created and that the stdout file has size greater then 0.
#
function checkSingleCHMTestTask {
  local jobDir=$1
  local taskId=$2

  getSingleCHMTestTaskStdOutFile "$jobDir" "$taskId"

  if [ $? != 0 ] ; then
    return 1
  fi

  if [ ! -s "$CHM_STD_OUT_FILE" ] ; then
    logWarning "$CHM_STD_OUT_FILE file not found"
    return 2
  fi

  local outputImageRelativePath=`egrep "^${taskId}${CONFIG_DELIM}" "$jobDir/$RUN_CHM_CONFIG" | sed "s/^.*${CONFIG_DELIM}//" | head -n 4 | tail -n 1`

  if [ ! -s "$jobDir/${outputImageRelativePath}" ] ; then
    logWarning "No output image found for task $taskId $jobDir/${outputImageRelativePath}"
    return 3
  fi

  return 0
}



#
# Verify Results
# verifyCHMTestResults(iteration,jobDir,jobs,writeToFailFile)
#
function verifyCHMTestResults {
  local iteration=$1
  local jobDir=$2
  local jobStart=$3
  local jobEnd=$4
  local writeToFailFile=$5
  local numFailedJobs=0
  local anyFailed="no"

  for Y in `seq $jobStart $jobEnd` ; do
    checkSingleCHMTestTask "$jobDir" $Y
    if [ $? != 0 ] ; then
      anyFailed="yes"
      let numFailedJobs++
      if [ "$writeToFailFile" == "yes" ] ; then
        echo "$Y" >> "$jobDir/$FAILED_JOBS_TMP_FILE"
      fi
    fi
  done
 
  local verifyExit=0

  if [ "$writeToFailFile" == "yes" ] ; then
    if [ -e "$jobDir/$FAILED_JOBS_FILE" ] ; then
      $MV_CMD "$jobDir/$FAILED_JOBS_FILE" $jobDir/${FAILED}.$(( $iteration - 1 )).${JOBS_SUFFIX}
    fi
  fi

  # make sure we have a sorted unique list of failed jobs
  if [ "$anyFailed" == "yes" ] ; then
    verifyExit=1
    if [ "$writeToFailFile" == "yes" ] ; then
      logMessage "Creating $FAILED_JOBS_FILE file"
      cat "$jobDir/$FAILED_JOBS_TMP_FILE" | sort -g | uniq > "$jobDir/$FAILED_JOBS_FILE"
      $RM_CMD -f "$jobDir/$FAILED_JOBS_TMP_FILE"
    fi
  fi

  NUM_FAILED_JOBS=$numFailedJobs
  return $verifyExit
}

#
#
#
function getFatalExceptionFromSingleCHMTestTask {
  # $1 - Job Directory
  # $2 - Task Id

  JOBDIR=$1
  TASKID=$2

  getSingleCHMTestTaskLogFile $JOBDIR $TASKID
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
function getRunTimeOfSingleCHMTestTask {
  # $1 - Job Directory
  # $2 - Task Id
  RUNTIME_SECONDS=-1

  JOBDIR=$1
  TASKID=$2

  getSingleCHMTestTaskLogFile $JOBDIR $TASKID
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
# is chummed to remote clusters before program exits unless shouldExit
# is set to anything then the function will return 2 instead
# checkForKillFile(<directory>,shouldExit)
#
function checkForKillFile {
  local jobDir=$1
  local shouldExit=$2
  local killFile="$jobDir/$KILL_JOB_REQUEST"

  # If kill file exists, chum it to remote clusters and
  # exit 
  if [ -e "$killFile" ] ; then
    if [ -z "$shouldExit" ] ; then
      logMessage "$killFile detected.  Chumming kill file to remote cluster(s) and exiting..."
    else
      logMessage "$killFile detected.  Chumming kill file to remote cluster(s)"
    fi
    
    logMessage "Running $CHUMBINARY --path $killFile --cluster $CHUMMEDLIST > $jobDir/$KILLED_CHUM_OUT_FILE"
    $CHUMBINARY --path $killFile --cluster $CHUMMEDLIST >> "$jobDir/$KILLED_CHUM_OUT_FILE"
    if [ -z "$shouldExit" ] ; then
      exit 1
    fi
    return 2
  fi
  return 0
}

#
# Moves any cluster folders out of the way by renaming them with
# a #.old suffix
# moveOldClusterFolders $iteration $jobDirectory
#
function moveOldClusterFolders {
  local iteration=$1
  local jobDir=$2
  local returnValue=0
  
  if [ -n "$CHUMMEDLIST" ] ; then
    for Y in `echo "$CHUMMEDLIST" | sed "s/,/\n/g"` ; do
      if [ -d "$jobDir/$Y" ] ; then
        $MV_CMD -f "$jobDir/$Y" "$jobDir/${Y}.$iteration.${OLD_SUFFIX}"
        if [ $? != 0 ] ; then
           returnValue=1
        fi
      fi
    done
  fi
  return $returnValue
}

#
# gets number of jobs in config by looking at #::: of the last line
# of config file
# getNumberOfCHMTestJobsFromConfig $jobDirectory
#
function getNumberOfCHMTestJobsFromConfig {
  local jobDir=$1
  if [ ! -e "$jobDir/$RUN_CHM_CONFIG" ] ; then
     return 1
  fi
  local lastLine=`tail -n 1 $jobDir/$RUN_CHM_CONFIG`
  
  if [ -z "$lastLine" ] ; then
    return 2
  fi

  NUMBER_JOBS=`echo "$lastLine" | sed "s/${CONFIG_DELIM}.*//"`
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
  local clusterList=$1   # $1 cluster list
  local thePath=$2       # $2 path
  local otherArgs=$3     # $3 other args to pass to land
  local numRetries=$4
  local retrySleep=$5

  # bail if there are no clusters
  if [ -z "$clusterList" ] ; then
      logWarning "No clusters in cluster list"
      return 1
  fi

  for y in `seq 0 $numRetries` ; do
    $LANDBINARY --path $thePath --cluster $clusterList $otherArgs

    if [ $? == 0 ] ; then
      return 0
    fi
    logWarning "Download attempt # $(($y + 1)) of $(($numRetries+1)) of $thePath failed.  Sleeping $retrySleep seconds"
    sleep $retrySleep
  done
  return 1
}

#
# Upload data in path
#
function chumData {
 local clusterList=$1 # $1 cluster list
 local thePath=$2     # $2 path
 local chumOut=$3     # $3 chum out file
 local otherArgs=$4   # $4 other args to pass to chum

  # bail if there are no clusters
  if [ -z "$clusterList" ] ; then
      logWarning "No clusters in cluster list"
      return 1
  fi

  $CHUMBINARY --listchummed --path $thePath --cluster $clusterList $otherArgs > "$chumOut" 
 
  if [ $? != 0 ] ; then
     
     logWarning "Chum of $thePath failed"
     return 2
  fi

  # Update the CHUMMEDLIST with the clusters the job was uploaded too
  CHUMMEDLIST=`cat $chumOut | egrep "^chummed.clusters" | sed "s/^chummed.clusters=//"`
  
  return 0
}

#
# getStatusOfJobsInCastOutFile $jobDirectory
# This function uses panfishstat to check status of jobs listed in $jobDirectory/cast.out file
# setting status in JOBSTATUS variable.
# 
#
function getStatusOfJobsInCastOutFile {
  local jobDir=$1
  local castOutFile=$2
 
  JOBSTATUS="unknown"

  if [ ! -e "$jobDir/$castOutFile" ] ; then
    logWarning "No $jobDir/$castOutFile file found"
    return 2
  fi

  OUT=`$PANFISHSTATBINARY --statusofjobs $jobDir/$castOutFile`
  if [ $? != 0 ] ; then
      logWarning "Error calling $PANFISHSTATBINARY --statusofjobs $jobDir/$castOutFile"
      return 1
  fi
      
  JOBSTATUS=`echo $OUT | egrep "^status=" | sed "s/^status=//" | tr \[:upper:\] \[:lower:\]`
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
function createCHMTestConfig {
  local imageDir=$1
  local configFile=$2
  local tilesW=$3
  local tilesH=$4
  local tilesPerJob=$5
  local chmOpts=$6
  local modelDir=$7
  local imageSuffix=$8
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
  # <JOBID>:::<Model directory full path>
  # <JOBID>:::<CHM options and tile flags from tileSets>
  # <JOBID>:::<relative output path for image of format out/[image name]/[JOBID].[image suffix]
  let cntr=1
  for z in `find $imageDir -name "*.${imageSuffix}" -type f | sort -n` ; do
    imageName=`echo $z | sed "s/^.*\///"`
    for (( y=0 ; y < ${#tileSets[@]} ; y++ )) ; do
      echo "${cntr}${CONFIG_DELIM}${z}" >> "$configFile"
      echo "${cntr}${CONFIG_DELIM}${modelDir}" >> "$configFile"
      echo "${cntr}${CONFIG_DELIM}${chmOpts} ${tileSets[$y]}" >> "$configFile"
      outputFile="${OUT_DIR_NAME}/${imageName}.${IMAGE_TILE_DIR_SUFFIX}/${cntr}.${imageSuffix}" >> "$configFile"
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

# 
# Creates output directories based on list of input images
# createImageOutputDirectories(output directory,image directory, image suffix)
function createImageOutputDirectories {
  local outDir=$1
  local imageDir=$2
  local imageSuffix=$3

  if [ ! -d "$outDir" ] ; then
     logWarning "Output directory $outDir is not a directory"
     return 1
  fi

  if [ ! -d "$imageDir" ] ; then
     logWarning "Image directory $imageDir is not a directory"
     return 1
  fi

  for z in `find $imageDir -name "*.${imageSuffix}" -type f` ; do
    imageName=`echo $z | sed "s/^.*\///"`
    makeDirectory "$outDir/${imageName}.${IMAGE_TILE_DIR_SUFFIX}"
  done

  return 0
}

#
# Recursively removes directory passed in and logs
# the start and completion of the task
#
function recursivelyRemoveDirectory {
  local pathToRemove=$1
  logStartTime "rm $pathToRemove"
  if [ ! -d "$pathToRemove" ] ; then
    logMessage "$pathToRemove is not a directory"
    return 1
  fi

  $RM_CMD -rf "$pathToRemove"
  local exitCode=$?
  logEndTime "rm $pathToRemove" $START_TIME $exitCode
  return $exitCode
}


#     
# Wait for jobs to complete
#
function waitForJobs {
  local iteration=$1
  local jobDir=$2
  local castOutFile=$3

  logStartTime "WaitForJobs in $castOutFile Iteration $iteration"
  local waitForJobsStartTime=$START_TIME
  local jobStatus="NA"
  local firstTime="yes"
  DOWNLOAD_DATA_REQUEST="DOWNLOAD.DATA.REQUEST"

  while [ "$jobStatus" != "done" ]
  do

      checkForKillFile "$jobDir"

      if [ "$firstTime" == "yes" ] ; then
        logMessage "Iteration $iteration job status is $jobStatus.  Sleeping $WAIT_SLEEP_TIME seconds"
        sleep $WAIT_SLEEP_TIME
      else
        firstTime="no"
      fi

      if [ -e "$jobDir/$DOWNLOAD_DATA_REQUEST" ] ; then
          $RM_CMD -f "$jobDir/$DOWNLOAD_DATA_REQUEST"
          logMessage "$DOWNLOAD_DATA_REQUEST file found.  Performing download"
          landData "$CHUMMEDLIST" "$jobDir" "$LAND_JOB_OPTS" "0" "0"
          $RM_CMD -f "$jobDir/$DOWNLOAD_DATA_REQUEST"
          logMessage "Removing $DOWNLOAD_DATA_REQUEST file"
      fi
      
      # Get status of job which will be set in JOBSTATUS 
      getStatusOfJobsInCastOutFile "$jobDir" "$castOutFile"

      statusRetVal=$?
   
      # Bail if there isn't a cast.out file cause this will loop forever otherwise
      if [ $statusRetVal -eq 2 ] ; then
         logEndTime "WaitForJobs in $castOutFile Iteration $iteration" $waitForJobsStartTime 1
         return 1
      fi

      if [ $statusRetVal -eq 0 ] ; then
        jobStatus="$JOBSTATUS"
      fi
  done
  logEndTime "WaitForJobs in $castOutFile Iteration $iteration" $waitForJobsStartTime 0
  return 0
}

function moveCastOutFile {
  local iteration=$1
  local jobDir=$2
  local castOutFileName=$3

  # Move the cast.out file out of the way
  if [ -e "$jobDir/$castOutFileName" ] ; then
     $MV_CMD -f "$jobDir/$castOutFileName" "$jobDir/${castOutFileName}.${iteration}.${OUT_SUFFIX}"
     if [ $? != 0 ] ; then
       return 1
     fi
  fi
  return 0
}

#    
# Submit job(s) via Panfish
# 
# castCHMTestJob(iteration,jobDir,jobStart,jobEnd,jobName)
# 
# iteration - numeric value denoting iteration ie 1 or 2 ...
# jobDir - directory containing job
# jobStart - either task id of starting job or a full path to file containing task ids of jobs to run
# jobEnd - either task id of last job (if jobStart is a file, this is ignored)
# jobName - Name of job to give to job scheduler must start with [A-Z|a-z] and stay away from funny chars
#           other then _
#
function castCHMTestJob {

  local iteration=$1
  local jobDir=$2
  local jobStart=$3
  local jobEnd=$4
  local jobName=$5

  local curdir=`pwd`
  cd $jobDir

  local taskArg="-t ${jobStart}-${jobEnd}"
  # Depending on the argument use either the --taskfile or -t flag
  if [ -e "$jobStart" ] ; then
    taskArg="--taskfile $jobStart"
  fi

  $CASTBINARY $taskArg -q $CHUMMEDLIST -N $jobName $BATCH_AND_WALLTIME_ARGS --writeoutputlocal -o $jobDir/${OUT_DIR_NAME}/${STD_OUT_DIR_NAME}/\$TASK_ID.${STD_OUT_SUFFIX} -e $jobDir/${OUT_DIR_NAME}/${STD_ERR_DIR_NAME}/\$TASK_ID.${STD_ERR_SUFFIX} $jobDir/${RUN_CHM_SH} > $jobDir/$CHM_TEST_CAST_OUT_FILE

  if [ $? != 0 ] ; then
      cd $curdir
      logWarning "Error calling $CASTBINARY $taskArg -q $CHUMMEDLIST -N $jobName $BATCH_AND_WALLTIME_ARGS --writeoutputlocal -o $jobDir/${OUT_DIR_NAME}/${STD_OUT_DIR_NAME}/\$TASK_ID.${STD_OUT_SUFFIX} -e $jobDir/${OUT_DIR_NAME}/${STD_ERR_DIR_NAME}/\$TASK_ID.${STD_ERR_SUFFIX} $jobDir/${RUN_CHM_SH} > $jobDir/$CHM_TEST_CAST_OUT_FILE"
      return 2
  fi

  # output in cast.out file will look like this
  # Your job-array 142.1-1:1 ("line") has been submitted
  # Your job-array 143.3-3:1 ("line") has been submitted
  # Your job-array 144.5-11:1 ("line") has been submitted

  cd $curdir
  return 0
}

#
# upload Model, Image and Job data via Panfish
#
function chumModelImageAndJobData {
  local iteration=$1
  local jobDir=$2
  local imageDir=$3
  local modelDir=$4


  # Upload model data
  chumData "$CHUMMEDLIST" "$modelDir" "$jobDir/$CHUM_MODEL_OUT_FILE" "$CHUM_MODEL_OPTS"
  if [ $? != 0 ] ; then
    logWarning "Unable to upload input model directory"
    return 1
  fi

  # Upload input image data
  chumData "$CHUMMEDLIST" "$imageDir" "$jobDir/$CHUM_IMAGE_OUT_FILE" "$CHUM_IMAGE_OPTS"
  if [ $? != 0 ] ; then
    logWarning "Unable to upload input image directory"
    return 2
  fi

  # Upload job directory
  chumData "$CHUMMEDLIST" "$jobDir" "$jobDir/$CHUM_OUT_FILE" "$CHUM_JOB_OPTS"

  if [ $? != 0 ] ; then
    logWarning "Unable to upload job directory"
    return 3
  fi

  return 0
}

#
# Moves old cluster folders out of the
# way and moves the cast.out file
# out of the way.
function moveOldDataForNewIteration {
  local iteration=$1
  local jobDir=$2
  local castOutFile=$3
  local returnValue=0

  # move all the old cluster folders if they exist out of the way
  moveOldClusterFolders "$(( $iteration - 1 ))" "$jobDir"
  if [ $? != 0 ] ; then
    logWarning "Unable to move cluster folders for previous iteration $(( $iteration - 1 ))"
    returnValue=1
  fi
  
  # move the cast file out of the way
  moveCastOutFile "$(( $iteration - 1 ))" "$jobDir" "$castOutFile"
  if [ $? != 0 ] ; then
    logWarning "Unable to move $castOutFile file for previous iteration $(( $iteration - 1 ))"
    let returnValue=$returnValue+2
  fi
  return $returnValue
}

#
#
#
function waitForDownloadAndVerifyCHMTestJobs {
  local iteration=$1
  local jobDir=$2
  local jobEnd=$3
  local castOutFile=$4
    
  # Wait for jobs to complete
  waitForJobs $iteration "$jobDir" "$castOutFile"

  if [ $? -eq 1 ] ; then
    logMessage "While checking if any jobs exist, it appears no $castOutFile exists."
  fi
  
  # Download completed job results
  landData  "$CHUMMEDLIST" "$jobDir" "$LAND_JOB_OPTS" "2" "$RETRY_SLEEP"

  if [ $? != 0 ] ; then
    logWarning "Unable to download data.  Will continue on with checking results just in case."
  fi

  checkForKillFile "$jobDir"

  # Verify Results
  verifyCHMTestResults $iteration "$jobDir" "1" "$jobEnd" "yes"
  return $?
}


#
# run jobs up to value specified with multiple retries built in
#
# runCHMTestJobs(jobDir,jobs,jobnName)
#
function runCHMTestJobs {

  local iteration=$1
  local jobDir=$2
  local imageDir=$3
  local modelDir=$4
  local jobEnd=$5
  local jobName=$6

  local altJobStart=""
  logStartTime "RunCHMTestJobs"
  local runJobsStartTime=$START_TIME

  local runJobsExit=1

  # Check if jobs have already completed successfully
  waitForDownloadAndVerifyCHMTestJobs "$iteration" "$jobDir" "$jobEnd" "$CHM_TEST_CAST_OUT_FILE"
  if [ $? == 0 ] ; then
     logEndTime "RunCHMTestJobs" $runJobsStartTime 0
     return 0
  fi

  local altJobStart="$jobDir/${FAILED_JOBS_FILE}"

  # take whatever iteration we are starting with and add max # of retries
  local maxRetries=$MAX_RETRIES
  let maxRetries=$maxRetries+$iteration

  while [ $iteration -le $maxRetries ]
  do
    # dump the current iteration to a file
    echo "$iteration" > "$jobDir/$CHM_TEST_ITERATION_FILE"

    if [ $iteration -gt 1 ] ; then
      # Move old panfish job folders out of way
      moveOldDataForNewIteration "$iteration" "$jobDir" "$CHM_TEST_CAST_OUT_FILE"
      logMessage "Iteration $iteration.  Jobs failed in previous iteration. sleeping $RETRY_SLEEP seconds before trying again"
      sleep $RETRY_SLEEP
    fi

    checkForKillFile "$jobDir"

    # Upload data to clusters
    chumModelImageAndJobData "$iteration" "$jobDir" "$imageDir" "$modelDir"
    if [ $? != 0 ] ; then
      jobFailed "Unable to upload job data"
    fi 

    checkForKillFile "$jobDir"

    # Submit job via Panfish
    castCHMTestJob $iteration "$jobDir" "$altJobStart" "1" "$jobName"

    if [ $? != 0 ] ; then
      jobFailed "Unable to submit jobs"
    fi

    waitForDownloadAndVerifyCHMTestJobs "$iteration" "$jobDir" "$jobEnd" "$CHM_TEST_CAST_OUT_FILE"
    if [ $? == 0 ] ; then
      runJobsExit=0
      break
    fi

    logMessage "Found $NUM_FAILED_JOBS failed job(s)"
    # Increment the counter
    iteration=$(( $iteration + 1 ))
  done

  if [ $runJobsExit != 0 ] ; then
    logWarning "Error running jobs...."
  fi

  logEndTime "RunCHMTestJobs" $runJobsStartTime $runJobsExit

  return $runJobsExit
}

#
# Given a task id and an offset this function
# parses a config file and returns the value from
# that line.  If there was an error a non zero return code
# is returned
# TASK_CONFIG_PARAM is the value that is set
function getParameterForTaskFromConfig {
  local taskId=$1
  local lineToParse=$2
  local configFile=$3

  local linesOfInterest=`egrep "^$taskId${CONFIG_DELIM}" $configFile`
  if [ $? != 0 ] ; then
    return 1
  fi
  
  if [ -z "$linesOfInterest" ] ; then
    return 1
  fi

 
  local numLines=`echo "$linesOfInterest" | wc -l`
  if [ $lineToParse -gt $numLines ] ; then
    return 2
  fi

  TASK_CONFIG_PARAM=`echo "$linesOfInterest" | head -n $lineToParse | tail -n 1 | sed "s/^.*${CONFIG_DELIM}//"`

  return 0
}

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
  local taskId=$1
  local jobDir=$2

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
