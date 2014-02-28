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
declare SCRIPTS_SUBDIR="$SCRIPT_DIR/scripts"

while getopts ":h:n" o; do
  case "${o}" in
    h)
      usage
      ;;
    n)
      CHM_JOB_NAME="${OPTARG}"
      ;;
    *)
      jobFailed "Invalid argument: ${o}."
      ;;
    esac
done

# load the helper functions
if [ -s "$SCRIPT_DIR/.helperfuncs.sh" ] ; then
  . $SCRIPT_DIR/.helperfuncs.sh
else 
  . $SCRIPTS_SUBDIR/.helperfuncs.sh
fi

getFullPath "$SCRIPT_DIR"
declare OUTPUT_DIR="$GETFULLPATHRET"


# Parse the configuration file
parseProperties "$SCRIPT_DIR" "$OUTPUT_DIR"

if [ $? != 0 ] ; then
  jobFailed "There was a problem parsing the properties"
fi

logEcho ""
logStartTime "Full run"
declare -i modeStartTime=$START_TIME
logEcho ""


getNumberOfCHMJobsFromConfig $OUTPUT_DIR
if [ $? != 0 ] ; then
  jobFailed "Error obtaining number of jobs"
fi

# Verify we have a MATLAB_DIR set to a directory
if [ ! -d "$MATLAB_DIR" ] ; then
  jobFailed "Unable to get path to matlab directory: $MATLAB_DIR"
fi

logEcho ""
  
# get Image Directory

# get model directory

# get last iteration
# need to find the last iteration and increment by one
# this also requires the MAX_RETRIES to be shifted in the runCHMTestJobs 

runCHMTestJobs "1" "$OUTPUT_DIR" "$imageDir" "$modelDir" "${NUMBER_JOBS}" "$CHM_JOB_NAME"

runJobsExit=$?
logEcho ""

# need to write out a this phase is happy file so the next step can skip all the checks

logEndTime "$MODE mode" $modeStartTime $runJobsExit

exit $runJobsExit


