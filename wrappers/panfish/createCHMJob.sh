#!/bin/bash

declare -r CREATE_PRETRAIN_MODE="createpretrained"

function usage() {

  echo -e "Create CHM job.

This program packages and runs CHM on grid compute resources via Panfish.

$0 <mode> <run_folder> <arguments & optional arguments>
  mode         Mode to run.  This script has several modes
               $CREATE_PRETRAIN_MODE -- Creates the job (requires -m,-b,-i)
  run_folder   The job folder used by all the modes

Arguments:
  -m model_folder  The folder that contains the model data.
                   (contains param.mat and MODEL_level#_stage#.mat)
  -b block_size    Process images in blocks of this size instead of all at once.
                   Can be given as a single value (used for both height and
                   width) or a WxH value. This can reduce processing time and
                   memory usage along with increasing quality. The block size
                   should be exactly the size of the training images.
  -i image_folder  Folder containing input images.  Images should be either in
                   tiff (.tiff|.tif) or png (.png) format and should be 8/16 bit 
                   grayscale

Optional Arguments:
  -o overlap_size  Only allowed with -b. Specifies how much the blocks should
                   overlap (default is none). Like -b this supports a single
                   value or a WxH value. The value used will depend on the size
                   of the structures being segmenting but at most 50 pixels
                   seems necessary.  (Default: 50 pixels)
  -T tiles_per_job Sets the number of tiles to be processed per job.  
                   If left unset value defaults to 1.
"
  exit 1
}

#
# Creates job directory using model folder passed in
# runCreatePreTrainedMode(<output/job directory>,<model directory>,<image directory>,<chm opts>)
#
function runCreatePreTrainedMode {

  local outputDir=$1
  local imageDir=$2
  local chmOpts=$3
  local modelDir=$4
  local blocksW=$5
  local blocksH=$6
  local overlapW=$7
  local overlapH=$8
  local tilesPerJob=$9

  # try to make the output directory and
  # stderr and stdout directories
  makeDirectory "$outputDir/$OUT_DIR_NAME/$STD_ERR_DIR_NAME"
  makeDirectory "$outputDir/$OUT_DIR_NAME/$STD_OUT_DIR_NAME"
 
  logMessage "Copying scripts to $outputDir"
  # Copy over runCHM.sh script
  /bin/cp "$SCRIPTS_SUBDIR/$RUN_CHM_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$outputDir/$RUN_CHM_SH\" \"$outputDir/.\""
  fi

  # Copy the helperfuncs.sh
  /bin/cp "$SCRIPTS_SUBDIR/$HELPER_FUNCS_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$SCRIPTS_SUBDIR/$HELPER_FUNCS_SH\" \"$outputDir/.\""
  fi

  # Copy run script over
  /bin/cp "$SCRIPTS_SUBDIR/$RUN_CHM_VIA_PANFISH_SH" "$outputDir/."
  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$SCRIPTS_SUBDIR/$RUN_CHM_VIA_PANFISH_SH\" \"$outputDir/.\""
  fi

  # Copy the properties
  /bin/cp "$SCRIPT_DIR/$PANFISH_CHM_PROPS" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$SCRIPT_DIR/$PANFISH_CHM_PROPS\" \"$outputDir/.\""
  fi

  local configFile="$outputDir/$RUN_CHM_CONFIG"

  logMessage "Getting dimensions of images"

  # Delete any existing configuration file
  /bin/rm -f "$configFile"

  getImageDimensionsFromDirOfImages "$imageDir" "png"

  if [ $? != 0 ] ; then
    # try looking at the tiff
    getImageDimensionsFromDirOfImages "$imageDir" "tiff"
    if [ $? != 0 ] ; then
      getImageDimensionsFromDirOfImages "$imageDir" "tif"
      if [ $? != 0 ] ; then
        jobFailed "Unable to get dimensions of an image in $imageDir"
      fi
    fi
  fi
  local imageWidth=$PARSED_WIDTH
  local imageHeight=$PARSED_HEIGHT

  let blocksW=`echo "scale=0;$blocksW-(2*$overlapW)" | bc -l`
  let blocksH=`echo "scale=0;$blocksH-(2*$overlapH)" | bc -l`


  logMessage "Images are ${imageWidth}x${imageHeight} in size"

  calculateTilesFromImageDimensions "$imageWidth" "$imageHeight" "$blocksW" "$blocksH"

  if [ $? != 0 ] ; then
    jobFailed "Unable to calculate tiles needed to process"
  fi

  local tilesW=$TILES_W
  local tilesH=$TILES_H
  
  logMessage "Tile dimensions: ${tilesW}x${tilesH} and each job will be processing $tilesPerJob tile(s)"
  
  logMessage "Generating $configFile configuration file"

  createCHMTestConfig "$imageDir" "$configFile" "$tilesW" "$tilesH" "$tilesPerJob" " $chmOpts" "$modelDir" "png"
  createCHMTestConfig "$imageDir" "$configFile" "$tilesW" "$tilesH" "$tilesPerJob" " $chmOpts" "$modelDir" "tiff"
  createCHMTestConfig "$imageDir" "$configFile" "$tilesW" "$tilesH" "$tilesPerJob" " $chmOpts" "$modelDir" "tif"

  for Y in `echo png tif tiff` ; do
    createImageOutputDirectories  "$outputDir/${OUT_DIR_NAME}" "$imageDir" "$Y"
  done  

  return $?
}

###########################################################
#
# Start of program
#
###########################################################

declare OVERLAP=""
declare MODEL_FOLDER=""
declare IMAGE_FOLDER=""

# get the directory where the script resides
declare SCRIPT_DIR=`dirname $0`
declare SCRIPTS_SUBDIR="$SCRIPT_DIR/scripts"


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


getFullPath "$2"

declare OUTPUT_DIR="$GETFULLPATHRET"
declare -i BLOCK_W=0
declare -i BLOCK_H=0
declare BLOCK=""
declare OVERLAP=""
declare -i OVERLAP_W=0
declare -i OVERLAP_H=0
declare MODEL_FOLDER=""
declare IMAGE_FOLDER=""
declare -i TILES_PER_JOB=1
shift 2

while getopts ":m:b:o:i:T:" o; do
  case "${o}" in
    m)
      getFullPath "${OPTARG}"
      MODEL_FOLDER="$GETFULLPATHRET"

      if [ ! -d "${MODEL_FOLDER}" ]; then 
        jobFailed "Model folder is not a directory."
      fi
      ;;
    b)
      parseWidthHeightParameter "${OPTARG}"
      if [ $? -ne 0 ] ; then
        jobFailed "Invalid block parameter"
      fi
      BLOCK_W=$PARSED_WIDTH
      BLOCK_H=$PARSED_HEIGHT
      BLOCK=" -b ${BLOCK_W}x${BLOCK_H}"
      ;;
    o)
      parseWidthHeightParameter "${OPTARG}"
      if [ $? -ne 0 ] ; then
        jobFailed "Invalid block parameter"
      fi
      OVERLAP_W=$PARSED_WIDTH
      OVERLAP_H=$PARSED_HEIGHT
      OVERLAP=" -o ${OVERLAP_W}x${OVERLAP_H}"
      ;;
    i)
      if [ ! -d "${OPTARG}" ]; then 
        jobFailed "Image folder is not a directory."
      fi
      getFullPath "${OPTARG}"
      IMAGE_FOLDER="$GETFULLPATHRET"
      ;;
    T)
      TILES_PER_JOB="${OPTARG}"
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

logEcho ""
logStartTime "$MODE mode"
declare -i MODE_START_TIME=$START_TIME
logEcho ""

# 
# Create mode
#
if [ "$MODE" == "$CREATE_PRETRAIN_MODE" ] ; then

  if [ -z "$MODEL_FOLDER" ] ; then
    jobFailed "Model must be specified via -m flag"
  fi

  if [ -z "$IMAGE_FOLDER" ] ; then
    jobFailed "Image Folder must be specified via -i flag"
  fi

  if [ ! -e "$CHM_BIN_DIR" ] ; then
    jobFailed "CHM bin dir $CHM_BIN_DIR does not exist"
  fi

  runCreatePreTrainedMode "$OUTPUT_DIR" "$IMAGE_FOLDER" "$OVERLAP $BLOCK" "$MODEL_FOLDER" $BLOCK_W $BLOCK_H $OVERLAP_W $OVERLAP_H $TILES_PER_JOB
  declare createPreTrainExit=$?

  if [ $createPreTrainExit -eq 0 ] ; then
    logEcho ""
    logMessage "Next step is to do a test run by running the following command:"
    logEcho ""
    logMessage "cd \"$OUTPUT_DIR\";./$RUN_CHM_VIA_PANFISH_SH $OUTPUT_DIR"
    logEcho ""
  fi

  logEcho ""
  logEndTime "$MODE mode" $MODE_START_TIME $createPreTrainExit
  logEcho ""
  exit $createPreTrainExit
fi

# implies an unsupported mode
jobFailed "$MODE not supported.  Invoke $0 for list of valid options."

