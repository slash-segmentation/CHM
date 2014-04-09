#!/bin/sh

function usage() {

  echo -e "Run CHM Train via Panfish.

This program runs CHM Train on grid compute resources via Panfish.  The
jobs are defined via runCHMTrain.sh.config file that must reside
in the same directory as this script.

runCHMTrainViaPanfish.sh <optional arguments>

Optional Arguments:
  -h           Displays this help.
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
# Chums data to remote clusters
#
function chumJobData {
  local task=$1
  local iteration=$2
  local jobDir=$3

  # parse the first job for image directory
  getParameterForTaskFromConfig "1" "1" "$jobDir/$RUN_CHM_TRAIN_CONFIG"
  if [ $? != 0 ] ; then
    return 1
  fi
  local imageDir=$TASK_CONFIG_PARAM

  # parse first job for label directory
  getParameterForTaskFromConfig "1" "2" "$jobDir/$RUN_CHM_TRAIN_CONFIG"
  if [ $? != 0 ] ; then
    return 2
  fi
  local labelDir=$TASK_CONFIG_PARAM

  # Upload image directory
  chumData "$CHM_TRAIN_CHUMMEDLIST" "$imageDir" "$CHM_TRAIN_CHUM_OUT" "$CHUM_CHM_TRAIN_IMAGE_OPTS"

  if [ $? != 0 ] ; then
    return 3
  fi
 
  # Upload label directory
  chumData "$CHM_TRAIN_CHUMMEDLIST" "$labelDir" "$CHM_TRAIN_CHUM_OUT" "$CHUM_CHM_TRAIN_LABEL_OPTS"
  if [ $? != 0 ] ; then
    return 4
  fi

  # Upload job directory
  chumData "$CHM_TRAIN_CHUMMEDLIST" "$jobDir" "$CHM_TRAIN_CHUM_OUT" "$CHUM_CHM_TRAIN_OPTS"
  if [ $? != 0 ] ; then
    return 5
  fi

  return 0
}

#
# Checks that a single Merge Tiles task ran successfully by verifying
# an output image was created and std out file has size greater then 0
#
function checkSingleTask {
  local task=$1
  local jobDir=$2
  local taskId=$3

  if [ ! -s "${jobDir}/$CHM_TRAIN_OUT_DIR_NAME/$STD_OUT_DIR_NAME/${taskId}.${STD_OUT_SUFFIX}" ] ; then
    return 1
  fi

  getParameterForTaskFromConfig "$taskId" "4" "$jobDir/$RUN_CHM_TRAIN_CONFIG"
  if [ $? != 0 ] ; then
    return 2
  fi

  if [ ! -s "${jobDir}/${TASK_CONFIG_PARAM}/param.mat" ] ; then
    return 3
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

declare CHM_TRAIN_JOB_NAME="chmtrain_job"

# get the directory where the script resides
declare SCRIPT_DIR=`dirname $0`

while getopts ":hn:" o; do
  case "${o}" in
    h)
      usage
      ;;
    n)
      CHM_TRAIN_JOB_NAME="${OPTARG}"
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
logStartTime "CHM Train"
declare -i modeStartTime=$START_TIME
logEcho ""

# Get the number of jobs we will be running
getNumberOfJobsFromConfig "$OUTPUT_DIR" "$RUN_CHM_TRAIN_CONFIG"
if [ $? != 0 ] ; then
  jobFailed "Error obtaining number of jobs from $RUN_CHM_TRAIN_CONFIG file"
fi

# set iteration to 1 initially
iteration="1"

# If a iteration file exists set iteration to
# the value from that file +1
getNextIteration "$OUTPUT_DIR" "$CHM_TRAIN_ITERATION_FILE"
if [ $? == 0 ] ; then
  iteration=$NEXT_ITERATION
  logMessage "$CHM_TRAIN_ITERATION_FILE file found setting iteration to $iteration"
fi

# Chum, submit, and wait for jobs to complete
runJobs "$RUN_CHM_TRAIN_SH" "$iteration" "$OUTPUT_DIR" "${NUMBER_JOBS}" "$CHM_TRAIN_JOB_NAME" "$CHM_TRAIN_CAST_OUT_FILE" "$CHM_TRAIN_CHUMMEDLIST" "$LAND_CHM_TRAIN_OPTS" "$CHM_TRAIN_FAILED_PREFIX" "$CHM_TRAIN_TMP_FILE" "$CHM_TRAIN_FAILED_FILE" "$MAX_RETRIES" "$WAIT_SLEEP_TIME" "$CHM_TRAIN_ITERATION_FILE" "$RETRY_SLEEP" "$CHM_TRAIN_BATCH_AND_WALLTIME_ARGS" "$CHM_TRAIN_OUT_DIR_NAME"

runJobsExit=$?
if [ "$runJobsExit" != 0 ] ; then
  logWarning "Error running CHM Train"
  logEndTime "CHM Train" $modeStartTime $runJobsExit
  exit $runJobsExit
fi

logEcho ""
logMessage "CHM Train successfully run."

# need to write out a this phase is happy file so the next step can skip all the checks

logEndTime "CHM Train" $modeStartTime $runJobsExit

exit $runJobsExit
