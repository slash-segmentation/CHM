#!/bin/bash

declare -r CREATE_BASIC_MODE="basictrain"
declare -r SOURCE_MODE="source"
function usage() {

  echo -e "Create CHM Train job.

This program packages and runs CHM Train on grid compute resources via Panfish.

$0 <mode> <run_folder> <arguments & optional arguments>
  mode         Mode to run.  This script has several modes
               $CREATE_BASIC_MODE -- Creates the job (requires -l,i)
               $SOURCE_MODE -- Allows this script to be sourced.  Used
               for unit testing.
  run_folder   The job folder used by all the modes

Arguments:
  -l label_folder  The folder that contains the label images.  
  -i image_folder  Folder containing input images.  Images should be either in
                   tiff (.tiff|.tif) or png (.png) format and should be 8/16 bit
                   grayscale

Optional Arguments:
  -S Nstage       The number of stages of training to perform. Must be >=2.
                  Default is 2.
  -L Nlevel       The number of levels of training to perform. Must be >=1.
                  Default is 4.
"
  exit 1
}

# 
# Creates CHM train output directories
# createCHMTrainOutputDirectories(jobDir)
#
# Creates this directory structure
# <jobdir>/
#          runCHMTrainOut/
#                           trainedmodel/
#                           stderr/
#                           stdout/
#
# If directory does not exist 1 is returned and if any of the
# make dir calls fails the function exits completely
#
function createCHMTrainOutputDirectories {
  local jobDir=$1

  makeDirectory "$jobDir/$CHM_TRAIN_OUT_DIR_NAME/$CHM_TRAIN_TRAINEDMODEL_OUT_DIR_NAME"

  makeDirectory "$jobDir/$CHM_TRAIN_OUT_DIR_NAME/$STD_OUT_DIR_NAME"

  makeDirectory "$jobDir/$CHM_TRAIN_OUT_DIR_NAME/$STD_ERR_DIR_NAME"

  return 0
}

# 
# Creates CHM Train configuration file
# 
# createCHMTrainConfig(jobdir)
#
# Format of output
# #:::<full path to input images>
# #:::<full path to input labels>
# #:::<options ie stage and level to pass to CHM_train>
# #:::<output directory relative to jobdir to write trained model to>
#
function createCHMTrainConfig {
  local inputImages=$1
  local inputLabels=$2
  local trainOpts=$3
  local outConfig=$4

  # remove the config if it exists already
  if [ -e "$outConfig" ] ; then
    $RM_CMD -f "$outConfig"
    if [ $? != 0 ] ; then
      return 1
    fi
  fi

  local cntr=1

  echo "${cntr}${CONFIG_DELIM}${inputImages}" > $outConfig
  echo "${cntr}${CONFIG_DELIM}${inputLabels}" >> $outConfig
  echo "${cntr}${CONFIG_DELIM}${trainOpts}" >> $outConfig
  echo "${cntr}${CONFIG_DELIM}${CHM_TRAIN_OUT_DIR_NAME}/${CHM_TRAIN_TRAINEDMODEL_OUT_DIR_NAME}" >> $outConfig

  return 0
}


#
# Creates job directory using model folder passed in
# runCreateBasicTrainMode(<output/job directory>,<images dir>,<labels dir>,<train opts>)
#
function runCreateBasicTrainMode {
  local scriptDir=$1
  local scriptsSubDir=$2
  local outputDir=$3
  local imagesDir=$4
  local labelsDir=$5
  local trainOpts=$6

  # try to make CHM Train output directories
  createCHMTrainOutputDirectories "$outputDir"
 
  logMessage "Copying scripts to $outputDir"
  # Copy over runCHMTrain.sh script
  $CP_CMD "$scriptsSubDir/$RUN_CHM_TRAIN_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running $CP_CMD \"$scriptsSubDir/$RUN_CHM_TRAIN_SH\" \"$outputDir/.\""
  fi

  # Copy over runCHMTrainViaPanfish.sh script
  $CP_CMD "$scriptsSubDir/$RUN_CHM_TRAIN_VIA_PANFISH_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running $CP_CMD \"$scriptsSubDir/$RUN_CHM_TRAIN_VIA_PANFISH_SH\" \"$outputDir/.\""
  fi

  # Copy the helperfuncs.sh
  $CP_CMD "$scriptsSubDir/$HELPER_FUNCS_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running $CP_CMD \"$scriptsSubDir/$HELPER_FUNCS_SH\" \"$outputDir/.\""
  fi

  # Copy the properties
  $CP_CMD "$scriptDir/$PANFISH_CHM_PROPS" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running $CP_CMD \"$scriptDir/$PANFISH_CHM_PROPS\" \"$outputDir/.\""
  fi

  local configFile="$outputDir/$RUN_CHM_TRAIN_CONFIG"

  # Delete any existing configuration file
  $RM_CMD -f "$configFile"

  logMessage "Generating $configFile configuration file"

  # create the config needed to run train
  createCHMTrainConfig "$imagesDir" "$labelsDir" "$trainOpts" "$configFile"

  return $?
}

###########################################################
#
# Start of program
#
###########################################################

declare STAGE="2"
declare LEVEL="4"
declare LABELS_FOLDER=""
declare IMAGES_FOLDER=""

# get the directory where the script resides
declare SCRIPT_DIR=`dirname $0`
declare SCRIPTS_SUBDIR="$SCRIPT_DIR/scripts"


# if source mode is set on command line simply
# exit
if [ $# -eq 1 ] ; then
  if [ "$1" == "$SOURCE_MODE" ] ; then
    return 0
  fi
fi

if [ $# -lt 2 ] ; then
  usage;
fi

declare  MODE=$1

# load the helper functions
if [ -s "$SCRIPT_DIR/.helperfuncs.sh" ] ; then
  . $SCRIPT_DIR/.helperfuncs.sh
else
  . $SCRIPTS_SUBDIR/.helperfuncs.sh
fi

logEcho ""
logStartTime "$MODE mode"
declare -i MODE_START_TIME=$START_TIME
logEcho ""

# if job directory does not exist create it
if [ ! -d "$2" ] ; then
  makeDirectory "$2"
  if [ $? != 0 ] ; then
    jobFailed "Error creating directory $2"
  fi
fi


getFullPath "$2"

declare OUTPUT_DIR="$GETFULLPATHRET"
shift 2

while getopts ":l:i:S:L:" o; do
  case "${o}" in
    l)
      getFullPath "${OPTARG}"
      LABELS_FOLDER="$GETFULLPATHRET"

      if [ ! -d "${LABELS_FOLDER}" ]; then 
        jobFailed "Labels folder is not a directory."
      fi
      ;;
    S)
      STAGE="${OPTARG}"
      ;;
    L)
      LEVEL="${OPTARG}"
      ;;
    i)
      if [ ! -d "${OPTARG}" ]; then 
        jobFailed "Image folder is not a directory."
      fi
      getFullPath "${OPTARG}"
      IMAGES_FOLDER="$GETFULLPATHRET"
      ;;
    *)
      jobFailed "Invalid argument: ${o}."
      ;;
    esac
done

# Parse the properties file
parseProperties "$SCRIPT_DIR" "$OUTPUT_DIR"

if [ $? != 0 ] ; then
  jobFailed "There was a problem parsing the properties"
fi

# 
# Create mode
#
if [ "$MODE" == "$CREATE_BASIC_MODE" ] ; then

  if [ -z "$LABELS_FOLDER" ] ; then
    jobFailed "Label images directory must be specified via -l flag"
  fi

  if [ -z "$IMAGES_FOLDER" ] ; then
    jobFailed "Images directory must be specified via -i flag"
  fi

  if [ ! -e "$CHM_BIN_DIR" ] ; then
    jobFailed "CHM bin dir $CHM_BIN_DIR does not exist"
  fi

  runCreateBasicTrainMode "$SCRIPT_DIR" "$SCRIPTS_SUBDIR" "$OUTPUT_DIR" "$IMAGES_FOLDER" "$LABELS_FOLDER" "-S $STAGE -L $LEVEL"
  declare createBasicTrainExit=$?

  if [ $createBasicTrainExit -eq 0 ] ; then
    logEcho ""
    logMessage "Next step is to run the program by running the following command:"
    logEcho ""
    logMessage "cd \"$OUTPUT_DIR\";./$RUN_CHM_TRAIN_VIA_PANFISH_SH $OUTPUT_DIR"
    logEcho ""
  fi

  logEcho ""
  logEndTime "$MODE mode" $MODE_START_TIME $createBasicTrainExit
  logEcho ""
  exit $createBasicTrainExit
fi

# implies an unsupported mode
jobFailed "$MODE not supported.  Invoke $0 for list of valid options."

