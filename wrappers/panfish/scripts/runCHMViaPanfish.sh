#!/bin/sh

function usage() {

  echo -e "Run CHM via Panfish.

This program runs CHM on grid compute resources via Panfish.  The
jobs are defined via runCHM.sh.config file that must reside
in the same directory as this script.

runCHMJobViaPanfish.sh <optional arguments>

Optional Arguments:
  -h           Displays this help.
  -C           Check if job is completed and exit 0 if yes otherwise
               no. (Cannot be used with any other flag.)
  -D           Download/Land results from remote clusters and exit.
               (Cannot be used with any other flag.)
  -U           Upload/Chum job data to remote clusters and exit.
               (Cannot be used with any other flag.)
  -S           Check if any jobs are running.  Exit 0 if no, 2 if
               yes and 1 if there was an error.
               (Cannot be used with any other flag.)
  -n           Specifies job name to pass to give to SGE.  No
               funny characters other then _ and only a-z|A-Z
               for the first character
"
  exit 1
}

#######################################################################
#
# Functions
#
#######################################################################

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
function checkSingleTask {
  local task=$1
  local jobDir=$2
  local taskId=$3

  getSingleCHMTestTaskStdOutFile "$jobDir" "$taskId"

  if [ $? != 0 ] ; then
    return 1
  fi

  if [ ! -s "$CHM_STD_OUT_FILE" ] ; then
    return 2
  fi

  getParameterForTaskFromConfig "$taskId" "4" "$jobDir/$RUN_CHM_CONFIG"
  if [ $? != 0 ] ; then
    return 3
  fi

  if [ ! -s "$jobDir/$TASK_CONFIG_PARAM" ] ; then
    return 4
  fi

  return 0
}

#
# upload Model, Image and Job data via Panfish
#
function chumJobData {
  local task=$1
  local iteration=$2
  local jobDir=$3

  # parse the first job for image directory
  getParameterForTaskFromConfig "1" "1" "$jobDir/$RUN_CHM_CONFIG"
  if [ $? != 0 ] ; then
    return 1
  fi
  local imageDir=`dirname $TASK_CONFIG_PARAM`

  # parse first job for model directory
  getParameterForTaskFromConfig "1" "2" "$jobDir/$RUN_CHM_CONFIG"
  if [ $? != 0 ] ; then
    return 2
  fi
  local modelDir=$TASK_CONFIG_PARAM

  
  # Upload model data
  chumData "$CHUMMEDLIST" "$modelDir" "$jobDir/$CHUM_MODEL_OUT_FILE" "$CHUM_MODEL_OPTS"
  if [ $? != 0 ] ; then
    logWarning "Unable to upload input model directory"
    return 3
  fi

  # Upload input image data
  chumData "$CHUMMEDLIST" "$imageDir" "$jobDir/$CHUM_IMAGE_OUT_FILE" "$CHUM_IMAGE_OPTS"
  if [ $? != 0 ] ; then
    logWarning "Unable to upload input image directory"
    return 4
  fi

  # Upload job directory
  chumData "$CHUMMEDLIST" "$jobDir" "$jobDir/$CHUM_OUT_FILE" "$CHUM_JOB_OPTS"

  if [ $? != 0 ] ; then
    logWarning "Unable to upload job directory"
    return 5
  fi

  return 0
}

#
# function called when USR2 signal is caught
#
on_usr2() {
  echo "USR2 signal caught exiting.." > $OUTPUT_DIR/KILL.JOB.REQUEST
  checkForKillFile $OUTPUT_DIR
  jobFailed "USR2 signal caught exiting.."
}

# trap usr2 signal cause its what gets sent by SGE when qdel is called
trap 'on_usr2' USR2


###########################################################
#
# Start of program
#
###########################################################

# Check if caller just wants to source this file for testing purposes
if [ $# -eq 1 ] ; then
  if [ "$1" == "source" ] ; then
    return 0
  fi
fi



declare CHECK_MODE="false"
declare DOWNLOAD_MODE="false"
declare UPLOAD_MODE="false"
declare STATUS_MODE="false"

declare CHM_JOB_NAME="chm_job"
# get the directory where the script resides
declare SCRIPT_DIR=`dirname $0`

while getopts ":CDSUhn:" o; do
  case "${o}" in
    h)
      usage
      ;;
    n)
      CHM_JOB_NAME="${OPTARG}"
      ;;
    C)
      CHECK_MODE="true"
      ;;
    D)
      DOWNLOAD_MODE="true"
      ;;
    U)
      UPLOAD_MODE="true"
      ;;
    S)
      STATUS_MODE="true"
      ;;
    *)
      echo "Invalid argument"
      usage
      ;;
    esac
done

if [ ! -s "$SCRIPT_DIR/.helperfuncs.sh" ] ; then
  echo "No $SCRIPT_DIR/.helperfuncs.sh found"
  exit 2
fi

# load the helper functions
. $SCRIPT_DIR/.helperfuncs.sh

getFullPath "$SCRIPT_DIR"
declare OUTPUT_DIR="$GETFULLPATHRET"

# Parse the configuration file
parseProperties "$SCRIPT_DIR" "$OUTPUT_DIR"

if [ $? != 0 ] ; then
  jobFailed "There was a problem parsing the $PANFISH_CHM_PROPS file"
fi

logEcho ""
logStartTime "Full run"
declare -i modeStartTime=$START_TIME
logEcho ""

# Get the number of jobs we will be running
getNumberOfJobsFromConfig "$OUTPUT_DIR" "$RUN_CHM_CONFIG"
if [ $? != 0 ] ; then
  jobFailed "Error obtaining number of jobs from $RUN_CHM_CONFIG file"
fi

#######################################################################
#
# If user passed in -S that means only run check and exit
#
#######################################################################
if [ "$STATUS_MODE" == "true" ] ; then
  logMessage "Getting status of running/pending jobs..."
  getStatusOfJobsInCastOutFile "$OUTPUT_DIR" "$CHM_TEST_CAST_OUT_FILE"
  if [ $? != 0 ] ; then
    logMessage "Unable to get status of jobs"
    logEndTime "Full run" $modeStartTime 1
    exit 1
  fi
  if [ "$JOBSTATUS" == "$DONE_JOB_STATUS" ] ; then
    logMessage "No running/pending jobs found."
    logEndTime "Full run" $modeStartTime 0
    exit 0
  fi 
  logMessage "Job status returned $JOBSTATUS  Job(s) still running."
  logEndTime "Full run" $modeStartTime 2
  exit 2
fi


#######################################################################
#
# If user passed in -C that means only run check and exit
#
#######################################################################
if [ "$CHECK_MODE" == "true" ] ; then
  logMessage "Checking results..."
  verifyResults "$RUN_CHM_SH" "1" "$OUTPUT_DIR" "1" "${NUMBER_JOBS}" "no" "$FAILED" "$FAILED_JOBS_TMP_FILE" "$FAILED_JOBS_FILE"
  if [ $? != 0 ] ; then
     logMessage "$NUM_FAILED_JOBS out of ${NUMBER_JOBS} job(s) failed."
     logEndTime "Full run" $modeStartTime 1
     exit 1
  fi
  logMessage "All ${NUMBER_JOBS} job(s) completed successfully."
  logEndTime "Full run" $modeStartTime 0
  exit 0
fi

#######################################################################
#
# If user passed in -D that means only download data and exit
#
#######################################################################
if [ "$DOWNLOAD_MODE" == "true" ] ; then
  logMessage "Downloading/Landing data..."
  landData "$CHUMMEDLIST" "$OUTPUT_DIR" "$LAND_JOB_OPTS" "0" "0"
  if [ $? != 0 ] ; then
     logWarning "Unable to retreive data"
     logEndTime "Full run" $modeStartTime 1
     exit 1
  fi
  logMessage "Download successful."
  logEndTime "Full run" $modeStartTime 0
  exit 0
fi

# Verify we have a MATLAB_DIR set to a directory
if [ ! -d "$MATLAB_DIR" ] ; then
  jobFailed "Unable to get path to matlab directory: $MATLAB_DIR"
fi

logEcho ""


#######################################################################
#
# If user passed in -U that means only upload data and exit
#
#######################################################################
if [ "$UPLOAD_MODE" == "true" ] ; then
  logMessage "Uploading/Chumming data..."
  chumJobData "$RUN_CHM_SH" "1" "$OUTPUT_DIR"
  if [ $? != 0 ] ; then
     logWarning "Unable to upload data"
     logEndTime "Full run" $modeStartTime 1
     exit 1
  fi
  logMessage "Upload successful."
  logEndTime "Full run" $modeStartTime 0
  exit 0
fi

# If a iteration file exists set iteration to
# the value from that file +1
getNextIteration "$OUTPUT_DIR" "$CHM_TEST_ITERATION_FILE"
if [ $? -eq 2 ] ; then
  iteration=1
else
  iteration=$NEXT_ITERATION
  logMessage "Setting iteration to $iteration"
fi

# Chum, submit, and wait for jobs to complete
runJobs "$RUN_CHM_SH" "$iteration" "$OUTPUT_DIR" "${NUMBER_JOBS}" "$CHM_JOB_NAME" "$CHM_TEST_CAST_OUT_FILE" "$CHUMMEDLIST" "$LAND_JOB_OPTS" "$FAILED" "$FAILED_JOBS_TMP_FILE" "$FAILED_JOBS_FILE" "$MAX_RETRIES" "$WAIT_SLEEP_TIME" "$CHM_TEST_ITERATION_FILE" "$RETRY_SLEEP" "$BATCH_AND_WALLTIME_ARGS" "$OUT_DIR_NAME" 

runJobsExit=$?
if [ "$runJobsExit" != 0 ] ; then
  logEndTime "Full run" $modeStartTime $runJobsExit
  jobFailed "Error running CHMTest"
fi

logEcho ""
logMessage "CHMTest successfully run."

# need to write out a this phase is happy file so the next step can skip all the checks

logEndTime "Full run" $modeStartTime $runJobsExit

exit $runJobsExit
