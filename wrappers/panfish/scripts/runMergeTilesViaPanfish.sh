#!/bin/sh

function usage() {

  echo -e "Run Merge Tiles via Panfish.

This program runs Merge Tiles on grid compute resources via Panfish.  The
jobs are defined via runMergeTiles.sh.config file that must reside
in the same directory as this script.

runMergeTilesViaPanfish.sh <optional arguments>

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
# Checks that a single Merge Tiles task ran successfully by verifying
# an output image was created and std out file has size greater then 0
#
function checkSingleMergeTilesTask {
  local jobDir=$1
  local taskId=$2

  getSingleMergeTilesStdOutFile "$jobDir" "$taskId"

  if [ $? != 0 ] ; then
    return 1
  fi

  if [ ! -s "$MERGE_TILES_STD_OUT_FILE" ] ; then
    return 2
  fi

  getParameterForTaskFromConfig "$taskId" "2" "$jobDir/$RUN_MERGE_TILES_CONFIG"
  if [ $? != 0 ] ; then
    return 3
  fi

  if [ ! -s "$jobDir/$TASK_CONFIG_PARAM" ] ; then
    return 4
  fi

  return 0
}

# 
# Creates merge tiles configuration file
# 
# createMergeTilesConfig(jobdir)
#
# Code examines the output directory of chm looking for all
# directories with tiles suffix, generating a job for each
# directory and setting output file name to be the tiles suffix 
# directory minus the tiles suffix.
# Code will also remove any preexisting config file and error out
# with return code of 1 if there is a problem removing the config
# file.
# Format of output
# #:::<full path to tile directory for a given image>
# #:::<full path where output image should be written>
#
function createMergeTilesConfig {
  local jobDir=$1

  local outConfig="${jobDir}/$RUN_MERGE_TILES_CONFIG"
  local cntr=1

  # bail if the job directory does not exist
  if [ ! -d "$jobDir" ] ; then
    return 1
  fi

  # remove the config if it exists already
  if [ -e "$outConfig" ] ; then
    $RM_CMD -f "$outConfig"
    if [ $? != 0 ] ; then
      return 2
    fi
  fi

  # another is to look in runchmout (OUT_DIR_NAME) directory and for every <IMAGE>.tiles dir make a
  # job and set output to be <IMAGE> in destination runmergetilesout folder
  for y in `find "$jobDir/${OUT_DIR_NAME}" -maxdepth 1 -name "*.${IMAGE_TILE_DIR_SUFFIX}" -type d | sort -g` ; do
    outImage=`echo $y | sed "s/^.*\///" | sed "s/\.${IMAGE_TILE_DIR_SUFFIX}//"`
    echo "${cntr}${CONFIG_DELIM}${y}" >> "$outConfig"
    echo "${cntr}${CONFIG_DELIM}$MERGE_TILES_OUT_DIR_NAME/$MERGED_IMAGES_OUT_DIR_NAME/${outImage}" >> "$outConfig"
    let cntr++
  done

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

declare CHECK_MODE="false"
declare DOWNLOAD_MODE="false"
declare UPLOAD_MODE="false"
declare STATUS_MODE="false"

declare CHM_JOB_NAME="mergetiles_job"
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
logStartTime "Merge Tiles"
declare -i modeStartTime=$START_TIME
logEcho ""

# Get the number of jobs we will be running
getNumberOfJobsFromConfig "$OUTPUT_DIR" "$RUN_MERGE_TILES_CONFIG"
if [ $? != 0 ] ; then
  jobFailed "Error obtaining number of jobs from $RUN_MERGE_TILES_CONFIG file"
fi

#######################################################################
#
# If user passed in -S that means only run check and exit
#
#######################################################################
if [ "$STATUS_MODE" == "true" ] ; then
  logMessage "Getting status of running/pending jobs..."
  getStatusOfJobsInCastOutFile "$OUTPUT_DIR" "$MERGE_TILES_CAST_OUT_FILE"
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
  logEndTime "Merge Tiles" $modeStartTime 2
  exit 2
fi


#######################################################################
#
# If user passed in -C that means only run check and exit
#
#######################################################################
if [ "$CHECK_MODE" == "true" ] ; then
  logMessage "Checking results..."
  verifyMergeTilesResults "1" "$OUTPUT_DIR" "1" "${NUMBER_JOBS}"
  if [ $? != 0 ] ; then
     logMessage "$NUM_FAILED_JOBS out of ${NUMBER_JOBS} job(s) failed."
     logEndTime "Merge Tiles" $modeStartTime 1
     exit 1
  fi
  logMessage "All ${NUMBER_JOBS} job(s) completed successfully."
  logEndTime "Merge Tiles" $modeStartTime 0
  exit 0
fi

#######################################################################
#
# If user passed in -D that means only download data and exit
#
#######################################################################
if [ "$DOWNLOAD_MODE" == "true" ] ; then
  logMessage "Downloading/Landing data..."
  landData "$CHUMMEDLIST" "$OUTPUT_DIR" "$LAND_MERGE_TILES_OPTS" "0" "0"
  if [ $? != 0 ] ; then
     logWarning "Unable to retreive data"
     logEndTime "Merge Tiles" $modeStartTime 1
     exit 1
  fi
  logMessage "Download successful."
  logEndTime "Merge Tiles" $modeStartTime 0
  exit 0
fi

logEcho ""

#######################################################################
#
# If user passed in -U that means only upload data and exit
#
#######################################################################
if [ "$UPLOAD_MODE" == "true" ] ; then
  logMessage "Uploading/Chumming data..."
  chumMergeTilesData "1" "$OUTPUT_DIR"
  if [ $? != 0 ] ; then
     logWarning "Unable to upload data"
     logEndTime "Merge Tiles" $modeStartTime 1
     exit 1
  fi
  logMessage "Upload successful."
  logEndTime "Merge Tiles" $modeStartTime 0
  exit 0
fi

# set iteration to 1 initially
iteration="1"

# If a iteration file exists set iteration to
# the value from that file +1
getNextIteration "$OUTPUT_DIR" "$MERGE_TILES_ITERATION_FILE"
if [ $? == 0 ] ; then
  iteration=$NEXT_ITERATION
  logMessage "$MERGE_TILES_ITERATION_FILE file found setting iteration to $iteration"
fi

# Chum, submit, and wait for jobs to complete
runMergeTilesJobs "$iteration" "$OUTPUT_DIR" "${NUMBER_JOBS}" "$CHM_JOB_NAME"

runJobsExit=$?
if [ "$runJobsExit" != 0 ] ; then
  logEndTime "Merge Tiles" $modeStartTime $runJobsExit
  jobFailed "Error running Merge Tiles"
fi

logEcho ""
logMessage "Merge Tiles successfully run."

# need to write out a this phase is happy file so the next step can skip all the checks

logEndTime "Merge Tiles" $modeStartTime $runJobsExit

exit $runJobsExit
