#!/bin/sh

function usage() {

  echo -e "Run CHM via Panfish.

This program runs CHM on grid compute resources via Panfish.  The
jobs are defined via runCHM.sh.config file that must reside
in the same directory as this script.

runCHMJobViaPanfish.sh <optional arguments>

Optional Arguments:
  -h           Displays this help.
  -n           Specifies job name to pass to give to SGE.  No
               funny characters other then _ and only a-z|A-Z
               for the first character
"
  exit 1
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

declare CHM_JOB_NAME="chm_job"
# get the directory where the script resides
declare SCRIPT_DIR=`dirname $0`

while getopts ":hn:" o; do
  case "${o}" in
    h)
      usage
      ;;
    n)
      CHM_JOB_NAME="${OPTARG}"
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
getNumberOfCHMTestJobsFromConfig $OUTPUT_DIR
if [ $? != 0 ] ; then
  jobFailed "Error obtaining number of jobs from $RUN_CHM_CONFIG file"
fi

# Verify we have a MATLAB_DIR set to a directory
if [ ! -d "$MATLAB_DIR" ] ; then
  jobFailed "Unable to get path to matlab directory: $MATLAB_DIR"
fi

logEcho ""

# Get the parameters for the first job cause we need to 
# know where the image and model directories are
getCHMTestJobParametersForTaskFromConfig "1" "$OUTPUT_DIR"
if [ $? != 0 ] ; then
  jobFailed "Error parsing the first job from config"
fi

# set image and model directories
imageDir=`dirname $INPUT_IMAGE`
modelDir="$MODEL_DIR"

# set iteration to 1 initially
iteration="1"

# If a iteration file exists set iteration to
# the value from that file +1
if [ -s "$OUTPUT_DIR/$CHM_TEST_ITERATION_FILE" ] ; then
  iteration=`cat $OUTPUT_DIR/$CHM_TEST_ITERATION_FILE`
  let iteration++
  logMessage "$CHM_TEST_ITERATION_FILE file found setting iteration to $iteration"
fi

# Chum, submit, and wait for jobs to complete
runCHMTestJobs "$iteration" "$OUTPUT_DIR" "$imageDir" "$modelDir" "${NUMBER_JOBS}" "$CHM_JOB_NAME"

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
