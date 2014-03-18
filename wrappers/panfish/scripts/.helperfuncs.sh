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
declare -r OUT_DIR_NAME="runChmOut"
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
declare -r DOWNLOAD_DATA_REQUEST="DOWNLOAD.DATA.REQUEST"
declare -r DONE_JOB_STATUS="done"

# Merge tiles defines
declare -r RUN_MERGE_TILES_SH="runMergeTiles.sh"
declare -r RUN_MERGE_TILES_CONFIG="runMergeTiles.sh.config"
declare -r MERGE_TILES_OUT_DIR_NAME="runMergeTilesOut"
declare -r MERGED_IMAGES_OUT_DIR_NAME="mergedimages"
declare -r MERGE_TILES_CAST_OUT_FILE="merge.tiles.cast.out"
declare -r MERGE_TILES_ITERATION_FILE="merge.tiles.iteration"
declare -r MERGE_TILES_FAILED_PREFIX="failed.merge.tiles"
declare -r MERGE_TILES_TMP_FILE="${MERGE_TILES_FAILED_PREFIX}.${JOBS_SUFFIX}.tmp"
declare -r MERGE_TILES_FAILED_FILE="${MERGE_TILES_FAILED_PREFIX}.${JOBS_SUFFIX}"
# Command defines
declare MV_CMD="/bin/mv"
declare RM_CMD="/bin/rm"
declare IDENTIFY_CMD="identify"
declare CONVERT_CMD="convert"
declare CP_CMD="/bin/cp"
declare TIME_V_CMD="/usr/bin/time -v"
declare UUIDGEN_CMD="uuidgen"
declare UNIQ_CMD="uniq"
declare CAT_CMD="cat"
declare SORT_CMD="sort"
declare DU_CMD="du"
declare SED_CMD="sed"
declare FIND_CMD="find"
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
  PANFISH_BIN_DIR=`egrep "^panfish.bin.dir" $theConfig | $SED_CMD "s/^.*= *//"`

  MATLAB_DIR=`egrep "^matlab.dir" $theConfig | $SED_CMD "s/^.*= *//"`
  BATCH_AND_WALLTIME_ARGS=`egrep "^batch.and.walltime.args" $theConfig | $SED_CMD "s/^.*= *//"`

  CHUMMEDLIST=`egrep "^cluster.list" $theConfig | $SED_CMD "s/^.*= *//"`

  CHM_BIN_DIR=`egrep "^chm.bin.dir" $theConfig | $SED_CMD "s/^.*= *//"`

  MAX_RETRIES=`egrep "^max.retries" $theConfig | $SED_CMD "s/^.*= *//"`
  RETRY_SLEEP=`egrep "^retry.sleep" $theConfig | $SED_CMD "s/^.*= *//"`

  WAIT_SLEEP_TIME=`egrep "^job.wait.sleep" $theConfig | $SED_CMD "s/^.*= *//"`

  LAND_JOB_OPTS=`egrep "^land.job.options" $theConfig | $SED_CMD "s/^.*= *//"`
  
  CHUM_JOB_OPTS=`egrep "^chum.job.options" $theConfig | $SED_CMD "s/^.*= *//"`
  
  CHUM_IMAGE_OPTS=`egrep "^chum.image.options" $theConfig | $SED_CMD "s/^.*= *//"`

  CHUM_MODEL_OPTS=`egrep "^chum.model.options" $theConfig | $SED_CMD "s/^.*= *//"`

  MERGE_TILES_BATCH_AND_WALLTIME_ARGS=`egrep "^mergetiles.batch.and.walltime.args" $theConfig | $SED_CMD "s/^.*= *//"`
  MERGE_TILES_CHUMMEDLIST=`egrep "^mergetiles.cluster.list" $theConfig | $SED_CMD "s/^.*= *//"`
  LAND_MERGE_TILES_OPTS=`egrep "^land.mergetiles.options" $theConfig | $SED_CMD "s/^.*= *//"`
  CHUM_MERGE_TILES_OPTS=`egrep "^chum.mergetiles.options" $theConfig | $SED_CMD "s/^.*= *//"`

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

  NUM_BYTES=`$DU_CMD $1 -bs | $SED_CMD "s/\W.*//"`  
  return 0
}


#
# Sets MERGE_TILES_STD_OUT_FILE to stdout file for MergeTiles job
#
function getSingleMergeTilesStdOutFile {
  local jobDir=$1
  local taskId=$2
  if [ ! -d "$jobDir" ] ; then
    return 1
  fi

  MERGE_TILES_STD_OUT_FILE="$jobDir/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME/${taskId}.${STD_OUT_SUFFIX}"
  return 0
}

#
# Checks a given task to see if it ran successfully
# checkSingleTask(task ie runCHM.sh or runMergeTiles.sh,jobDir,taskId)
# NEEDS TO BE IMPLEMENTED ELSEWHERE
#
function checkSingleTask {
  
  logWarning "checkSingleTask Not implemented..."
  return 10
}


#
# Uploads data to remote clusters
# chumJobData(task ie runCHM.sh or runMergeTiles.sh,jobDir,taskId)
# NEEDS TO BE IMPLEMENTED ELSEWHERE
#
function chumJobData {
  
  logWarning "chumJobData Not implemented..."
  return 10
}

#
# This function calls a user defined checkSingleTask on every
# task optionally writting out a file containining list of failed
# tasks. 
# verifyResults (task,iteration,jobDir,jobStart,jobEnd,write failed file [yes|no],failed file prefix,failed file tmp,failed file)
# 
# returns 0 if all jobs succeeded otherwise 1 
function verifyResults {
  local task=$1
  local iteration=$2
  local jobDir=$3
  local jobStart=$4
  local jobEnd=$5
  local writeToFailFile=$6
  local failedPrefix=$7
  local failedJobsTmpFile=$8
  local failedJobsFile=$9

  local numFailedJobs=0
  local anyFailed="no"

  for Y in `seq $jobStart $jobEnd` ; do
    checkSingleTask "$task" "$jobDir" "$Y"
    if [ $? != 0 ] ; then
      anyFailed="yes"
      let numFailedJobs++
      if [ "$writeToFailFile" == "yes" ] ; then
        echo "$Y" >> "$jobDir/$failedJobsTmpFile"
      fi
    fi
  done

  local verifyExit=0

  if [ "$writeToFailFile" == "yes" ] ; then
    if [ -e "$jobDir/$failedJobsFile" ] ; then
      logMessage "Renaming $jobDir/$failedJobsFile to $jobDir/${failedPrefix}.$(( $iteration - 1 )).${JOBS_SUFFIX}"
      $MV_CMD "$jobDir/$failedJobsFile" $jobDir/${failedPrefix}.$(( $iteration - 1 )).${JOBS_SUFFIX}
    fi
  fi

  # make sure we have a sorted unique list of failed jobs
  if [ "$anyFailed" == "yes" ] ; then
    verifyExit=1
    logMessage "Found $numFailedJobs failed job(s)"
    if [ "$writeToFailFile" == "yes" ] ; then
      logMessage "Creating $failedJobsFile file"
      $CAT_CMD "$jobDir/$failedJobsTmpFile" | $SORT_CMD -g | $UNIQ_CMD > "$jobDir/$failedJobsFile"
      $RM_CMD -f "$jobDir/$failedJobsTmpFile"
    fi
  fi

  NUM_FAILED_JOBS=$numFailedJobs
  return $verifyExit
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
  local chummedList=$2
  local shouldExit=$3
  local killFile="$jobDir/$KILL_JOB_REQUEST"

  if [ -z "$chummedList" ] ; then
    chummedList=$CHUMMEDLIST
  fi

  # If kill file exists, chum it to remote clusters and
  # exit 
  if [ -e "$killFile" ] ; then
    if [ -z "$shouldExit" ] ; then
      logMessage "$killFile detected.  Chumming kill file to remote cluster(s) and exiting..."
    else
      logMessage "$killFile detected.  Chumming kill file to remote cluster(s)"
    fi
    
    logMessage "Running $CHUMBINARY --path $killFile --cluster $chummedList > $jobDir/$KILLED_CHUM_OUT_FILE"
    $CHUMBINARY --path $killFile --cluster $chummedList >> "$jobDir/$KILLED_CHUM_OUT_FILE"
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
    for Y in `echo "$CHUMMEDLIST" | $SED_CMD "s/,/\n/g"` ; do
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

function getNumberOfJobsFromConfig {
  local jobDir=$1
  local configFileName=$2
  if [ ! -e "$jobDir/$configFileName" ] ; then
     return 1
  fi
  local lastLine=`tail -n 1 $jobDir/$configFileName`

  if [ -z "$lastLine" ] ; then
    return 2
  fi

  NUMBER_JOBS=`echo "$lastLine" | $SED_CMD "s/${CONFIG_DELIM}.*//"`
  return 0
}

#     
# Download data in path
#
function landData {
  local clusterList=$1   # $1 cluster list
  local thePath=$2       # $2 path
  local otherArgs=$3     # $3 other args to pass to land

  $LANDBINARY --path $thePath --cluster $clusterList $otherArgs
  return $?
}

#
# Upload data in path
#
function chumData {
  local clusterList=$1 # $1 cluster list
  local thePath=$2     # $2 path
  local chumOut=$3     # $3 chum out file
  local otherArgs=$4   # $4 other args to pass to chum

  $CHUMBINARY --listchummed --path $thePath --cluster $clusterList $otherArgs > "$chumOut" 
 
  if [ $? != 0 ] ; then
     logWarning "Chum of $thePath failed"
     return 1
  fi

  # Update the CHUMMEDLIST with the clusters the job was uploaded too
  CHUMMEDLIST=`$CAT_CMD $chumOut | egrep "^chummed.clusters" | $SED_CMD "s/^chummed.clusters=//"`
  
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
      
  JOBSTATUS=`echo $OUT | egrep "^status=" | $SED_CMD "s/^status=//" | tr \[:upper:\] \[:lower:\]`
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

  local identifyOutput=`$IDENTIFY_CMD -format '%wx%h' "${image}" 2>&1`
  # weird thing is in the unit tests identify never seems to kick back
  # a non zero exit code
  if [ $? -ne 0 ] ; then
     logWarning "Unable to run $IDENTIFY_CMD on image ${image}"
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

  local anImage=`$FIND_CMD $imageDir -name "*.${imageSuffix}" -type f | head -n 1` 

  if [ ! -f "$anImage" ] ; then
    logWarning "No images found in $imageDir with suffix $imageSuffix"
    return 1
  fi

  getImageDimensions "$anImage"
  
  return $?
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
  local chummedList=$4
  local landJobOpts=$5
  local waitSleepTime=$6

  logStartTime "WaitForJobs in $castOutFile Iteration $iteration"
  local waitForJobsStartTime=$START_TIME
  local jobStatus="NA"
  local firstTime="yes"

  while [ "$jobStatus" != "$DONE_JOB_STATUS" ]
  do

      checkForKillFile "$jobDir" "$chummedList"

      if [ "$firstTime" == "no" ] ; then
        logMessage "Iteration $iteration job status is $jobStatus.  Sleeping $waitSleepTime seconds"
        sleep $waitSleepTime
      else
        firstTime="no"
      fi

      if [ -e "$jobDir/$DOWNLOAD_DATA_REQUEST" ] ; then
          $RM_CMD -f "$jobDir/$DOWNLOAD_DATA_REQUEST"
          logMessage "$DOWNLOAD_DATA_REQUEST file found.  Performing download"
          landData "$chummedList" "$jobDir" "$landJobOpts"
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
function castJob {
  
  local task=$1
  local jobDir=$2
  local jobStart=$3
  local jobEnd=$4
  local jobName=$5
  local castOutFile=$6
  local chummedList=$7
  local batchAndWallTimeArgs=$8
  local jobOutDirName=$9
  

  local curdir=`pwd`
  cd $jobDir

  local taskArg="-t ${jobStart}-${jobEnd}"
  # Depending on the argument use either the --taskfile or -t flag
  if [ -e "$jobStart" ] ; then
    taskArg="--taskfile $jobStart"
  fi

  $CASTBINARY $taskArg -q $chummedList -N $jobName $batchAndWallTimeArgs --writeoutputlocal -o $jobDir/$jobOutDirName/${STD_OUT_DIR_NAME}/\$TASK_ID.${STD_OUT_SUFFIX} -e $jobDir/$jobOutDirName/${STD_ERR_DIR_NAME}/\$TASK_ID.${STD_ERR_SUFFIX} $jobDir/$task > $jobDir/$castOutFile

  if [ $? != 0 ] ; then
      cd $curdir
      logWarning "Error calling $CASTBINARY $taskArg -q $chummedList -N $jobName $batchAndWallTimeArgs --writeoutputlocal -o $jobDir/$jobOutDirName/${STD_OUT_DIR_NAME}/\$TASK_ID.${STD_OUT_SUFFIX} -e $jobDir/$jobOutDirName/${STD_ERR_DIR_NAME}/\$TASK_ID.${STD_ERR_SUFFIX} $jobDir/$task > $jobDir/$castOutFile"
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


function waitForDownloadAndVerifyJobs {
  local task=$1
  local iteration=$2
  local jobDir=$3
  local jobEnd=$4
  local castOutFile=$5
  local chummedList=$6
  local landJobOpts=$7
  local waitSleepTime=$8
  local failedPrefix=$9
  shift  # using shift cause only 1st 9 vars are accesible via $#
  local failedJobsTmpFile=$9
  shift
  local failedJobsFile=$9

  # Wait for jobs to complete
  waitForJobs $iteration "$jobDir" "$castOutFile" "$chummedList" "$landJobOpts" "$waitSleepTime"

  if [ $? -eq 1 ] ; then
    logMessage "While checking if any jobs exist, it appears no $castOutFile exists."
  fi
 
  # Download completed job results
  landData  "$chummedList" "$jobDir" "$landJobOpts"

  if [ $? != 0 ] ; then
    logWarning "Unable to download data.  Will continue on with checking results just in case."
  fi

  checkForKillFile "$jobDir" "$chummedList"

  # Verify Results
  verifyResults "$task" "$iteration" "$jobDir" "1" "$jobEnd" "yes" "$failedPrefix" "$failedJobsTmpFile" "$failedJobsFile"
  return $?
}

#
# run jobs up to value specified with multiple retries built in
#
# runJobs(jobDir,jobs,jobnName)
#
function runJobs {

  local task=$1
  local iteration=$2
  local jobDir=$3
  local jobEnd=$4
  local jobName=$5
  local castOutFile=$6
  local chummedList=$7
  local landJobOpts=$8
  local failedPrefix=$9
  shift  # using shift cause only 1st 9 vars are accesible via $#
  local failedJobsTmpFile=$9
  shift  
  local failedJobsFile=$9
  shift
  local maxIterationRetries=$9
  shift
  local iterationSleep=$9
  shift
  local iterationFile=$9
  shift
  local waitSleep=$9
  shift
  local batchAndWallTimeArgs=$9
  shift
  local jobOutDirName=$9

  local altJobStart=""

  logStartTime "$task"
  local runJobsStartTime=$START_TIME

  local runJobsExit=1

  logMessage "Checking for already running jobs and previously completed jobs" 
  # Check if jobs have already completed successfully
  waitForDownloadAndVerifyJobs "$task" "$iteration" "$jobDir" "$jobEnd" "$castOutFile" "$chummedList" "$landJobOpts" "$waitSleep" "$failedPrefix" "$failedJobsTmpFile" "$failedJobsFile"
  if [ $? == 0 ] ; then
     logEndTime "$task" $runJobsStartTime 0
     return 0
  fi

  local altJobStart="$jobDir/$failedJobsFile"

  # take whatever iteration we are starting with and add max # of retries
  local maxRetries=$maxIterationRetries
  let maxRetries=$maxRetries+$iteration

  while [ $iteration -le $maxRetries ]
  do
    # dump the current iteration to a file
    echo "$iteration" > "$jobDir/$iterationFile"

    if [ $iteration -gt 1 ] ; then
      # Move old panfish job folders out of way
      moveOldDataForNewIteration "$iteration" "$jobDir" "$castOutFile"
      logMessage "Iteration $iteration.  Jobs failed in previous iteration. sleeping $iterationSleep seconds before trying again"
      sleep $iterationSleep
    fi

    checkForKillFile "$jobDir" "$chummedList"

    logMessage "Uploading data for $task job(s)"
    # Upload data to clusters
    chumJobData "$task" "$iteration" "$jobDir"
    if [ $? != 0 ] ; then
      logWarning "Unable to upload data for $task job(s)"
      return 10
    fi 

    checkForKillFile "$jobDir" "$chummedList"

    # Submit job via Panfish
    castJob "$task" "$jobDir" "$altJobStart" "1" "$jobName" "$castOutFile" "$chummedList" "$batchAndWallTimeArgs" "$jobOutDirName"

    if [ $? != 0 ] ; then
      logWarning "Unable to submit jobs"
      return 11
    fi

    waitForDownloadAndVerifyJobs "$task" "$iteration" "$jobDir" "$jobEnd" "$castOutFile" "$chummedList" "$landJobOpts" "$waitSleep" "$failedPrefix" "$failedJobsTmpFile" "$failedJobsFile"
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

  logEndTime "$task" $runJobsStartTime $runJobsExit

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

  TASK_CONFIG_PARAM=`echo "$linesOfInterest" | head -n $lineToParse | tail -n 1 | $SED_CMD "s/^.*${CONFIG_DELIM}//"`

  return 0
}

# 
# getNextIteration
#
function getNextIteration {
  local jobDir=$1
  local iterationFileName=$2
  NEXT_ITERATION=1
  if [ -s "$jobDir/$iterationFileName" ] ; then
    iteration=`$CAT_CMD $jobDir/$iterationFileName`
    let iteration++
    if [ $? != 0 ] ; then
      return 2
    fi
    NEXT_ITERATION=$iteration
    return 0
  fi
  return 1
}
