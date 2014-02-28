#!/bin/bash


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


###########################################################
#
# Functions
#
###########################################################


function usage {
  echo "This script runs CHM on a slice/image of data.

What is run is defined by the environment variable SGE_TASK_ID.
The script uses SGE_TASK_ID and extracts the job from $RUN_CHM_CONFIG
file residing in the same directory as this script.

The script also examines $PANFISH_CHM_PROPS for configuration that is
needed such as path to matlab.  This script will return 0 exit
code upon success otherwise failure. 
 
"
   
  exit 1
}


###########################################################
#
# Start of script
#
###########################################################

# Dont allow job to run if SGE_TASK_ID is NOT set
if [ -z "$SGE_TASK_ID" ] ; then
  usage
fi

# Bail if no config is found
if [ ! -s "$SCRIPT_DIR/$RUN_CHM_CONFIG" ] ; then
  jobFailed "No $SCRIPT_DIR/$RUN_CHM_CONFIG found"
fi


logStartTime "$RUN_CHM_SH" 

declare -r runChmStartTime=$START_TIME


# set SCRATCH variable to /tmp/chm.# or to whatever $PANFISH_SCRATCH/chm.#
# if PANFISH_SCRATCH variable is not empty
declare uuidVal=`uuidgen`
declare scratchDir="/tmp/chm.${uuidVal}"
if [ -n "$PANFISH_SCRATCH" ] ; then
  scratchDir="$PANFISH_SCRATCH/chm.${uuidVal}"
fi

makeDirectory "$scratchDir"

# Parse the config for job parameters
declare -r inputImage=$PANFISH_BASEDIR/`egrep "^${SGE_TASK_ID}:::" $SCRIPT_DIR/$RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 1`
declare -r inputImageName=`echo $inputImage | sed "s/^.*\///"`
declare -r modelDir=$PANFISH_BASEDIR/`egrep "^${SGE_TASK_ID}:::" $SCRIPT_DIR/$RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 2 | tail -n 1`
declare -r chmOpts=`egrep "^${SGE_TASK_ID}:::" $SCRIPT_DIR/$RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 3 | tail -n 1`

declare -r finalImage=${SCRIPT_DIR}/`egrep "^${SGE_TASK_ID}:::" $SCRIPT_DIR/$RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 4 | tail -n 1`

# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PANFISH_BASEDIR}/${MATLAB_DIR}/bin/glnxa64
export MATLAB_BIN_DIR="${PANFISH_BASEDIR}/${MATLAB_DIR}/bin"
export PATH=$PATH:$MATLAB_BIN_DIR

# Todo: Need to make this task run in the background and then check on job completion
#       this will allow the script to catch KILL requests etc.. and to track memory
#       usage etc.
/usr/bin/time -v $PANFISH_BASEDIR/$CHM_BIN_DIR/$CHM_TEST_SH "$inputImage" "$scratchDir" -m "$modelDir" $chmOpts -s

declare chmExitCode=$?

# copy back image to final directory
if [ ! -s "$scratchDir/$inputImageName" ] ; then
  logWarning "No $scratchDir/$inputImageName found to copy"
  chmExitCode=2
else 
  /bin/cp -f "$scratchDir/$inputImageName" "$finalImage"
  if [ $? != 0 ] ; then
    logWarning "Error running /bin/cp -f \"$scratchDir/$inputImageName\" \"$finalImage\""
    chmExitCode=3
  fi
fi

# remove Scratch Directory
if [ -d "$scratchDir" ] ; then
  /bin/rm -rf "$scratchDir"
fi

logEndTime "$RUN_CHM_SH" $runChmStartTime $chmExitCode

exit $chmExitCode
