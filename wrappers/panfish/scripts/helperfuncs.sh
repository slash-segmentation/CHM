#!/bin/sh

#
# Variables used by scripts
#
#
RUN_CHM_CONFIG="runCHM.sh.config"

CAST_OUT_FILE="cast.out"
CHUM_OUT_FILE="chum.out"
CLUSTER_LIST_FILE="cluster.list"
JOB_PROPERTIES_FILE="job.properties"

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

  OUT_IMAGE_RAW=`egrep "^${TASKID}:::" $JOBDIR/$RUN_CHM_CONFIG | head -n 1 | sed "s/^.*::://"`

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
  CURDIR=`pwd`
 
  cd $1 2> /dev/null
  if [ $? == 0 ] ; then
    GETFULLPATHRET=`pwd`
    cd $CURDIR
    return 0
  fi

  cd $CURDIR

  echo $1 | egrep "^/"
 
  if [ $? == 0 ] ; then
    GETFULLPATHRET=$1
    return 0
  fi
  
  GETFULLPATHRET="$CURDIR/$1"
  
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
  NUMBER_JOBS=`tail -n 1 $1/$RUN_CHM_CONFIG | sed "s/:::.*//"`
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

  FIRST_INPUT_IMAGE=`head -n 1 $1/$RUN_CHM_CONFIG | sed "s/^.*::://"`
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
   if [ -e "$1/$CLUSTER_LIST_FILE" ] ; then
     CHUMMEDLIST=`cat $1/$CLUSTER_LIST_FILE`
   else
     getMatlabDirectory $1
     if [ $? != 0 ] ; then
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
# Gets the matlab directory by parsing
# job.properties file in $1 directory passed in
#
function getMatlabDirectory {
  # $1 jobDirectory
  MATLAB_DIR="matlab"


  if [ ! -s "$1/$JOB_PROPERTIES_FILE" ] ; then
     logWarning "No $1/JOB_PROPERTIES_FILE file found"
     return 1
  fi

  MATLAB_DIR=`egrep "^matlab.dir *= *" "$1/$JOB_PROPERTIES_FILE" | sed "s/^matlab.dir *= *//"`
  return 0
}

function writeJobProperties {
  # $1 jobDirectory

  echo "matlab.dir=$MATLAB_DIR" > "$1/$JOB_PROPERTIES_FILE"
  return 0
}


