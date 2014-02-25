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
                   tiff (.tiff) or png (.png) format and should be 8/16 bit 
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
  local modelDir=$2
  local imageDir=$3
  local chmOpts=$4
  local blocksW=$5
  local blocksH=$6
  local tilesPerJob=$7

  # try to make the output directory
  makeDirectory $outputDir
 
  # Copy over CHM folder
  local inputChmFolder="$SCRIPT_DIR/CHM"
  
  logMessage "Copying over $inputChmFolder to $outputDir"
  
  /bin/cp -dR "$inputChmFolder" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp -dR \"$inputChmFolder\" \"$outputDir/.\""
  fi

  # Create out directory under $OUTPUT_DIR/CHM folder
  makeDirectory "$outputDir/CHM/out"

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
  /bin/cp "$SCRIPTS_SUBDIR/$RUN_CHM_JOB_VIA_PANFISH_SH" "$outputDir/."
  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$SCRIPTS_SUBDIR/$RUN_CHM_JOB_VIA_PANFISH_SH\" \"$outputDir/.\""
  fi

  # Copy the properties
  /bin/cp "$SCRIPT_DIR/$PANFISH_CHM_PROPS" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$SCRIPT_DIR/$PANFISH_CHM_PROPS\" \"$outputDir/.\""
  fi

  getSizeOfPath $modelDir
  logMessage "Copying Model/Training Data $modelDir which is $NUM_BYTES bytes in size to $outputDir/CHM/out/"
  # Copy training data into output folder under CHM folder
  /bin/cp -r "$modelDir/"* "$outputDir/CHM/out/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp -r \"$modelDir/\"* \"$outputDir/CHM/out/.\""
  fi

  getSizeOfPath "$outputDir/CHM"
  
  # create out directory under $OUTPUT_DIR
  makeDirectory "$outputDir/out/log/chm"
  makeDirectory "$outputDir/out/log/joberr"
  makeDirectory "$outputDir/out/log/jobout"

  local configFile="$outputDir/$RUN_CHM_CONFIG"

  logMessage "Getting dimensions of images"

  # Delete any existing configuration file
  /bin/rm -f "$configFile"

  getImageDimensionsFromDirOfImages "$imageDir" "png"

  if [ $? != 0 ] ; then
    # try looking at the tiff
    getImageDimensionsFromDirOfImages "$imageDir" "tiff"
    if [ $? != 0 ] ; then
      jobFailed "Unable to get dimensions of an image in $imageDir"
    fi
  fi
  local imageWidth=$PARSED_WIDTH
  local imageHeight=$PARSED_HEIGHT

  logMessage "Images are ${imageWidth}x${imageHeight} in size"

  calculateTilesFromImageDimensions "$imageWidth" "$imageHeight" "$blocksW" "$blocksH"

  if [ $? != 0 ] ; then
    jobFailed "Unable to calculate tiles needed to process"
  fi

  local tilesW=$TILES_W
  local tilesH=$TILES_H
  
  logMessage "Tile dimensions: ${tilesW}x${tilesH} and each job will be processing $tilesPerJob tile(s)"
  
  logMessage "Generating $configFile configuration file"

  createConfig "$imageDir" "$configFile" "$tilesW" "$tilesH" "$tilesPerJob" " $chmOpts" "png"
  createConfig "$imageDir" "$configFile" "$tilesW" "$tilesH" "$tilesPerJob" " $chmOpts" "tiff"

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

declare -l MODE=$1

# load the helper functions
if [ -s "$SCRIPT_DIR/helperfuncs.sh" ] ; then
  . $SCRIPT_DIR/helperfuncs.sh
else
  . $SCRIPTS_SUBDIR/helperfuncs.sh
fi


getFullPath "$2"

declare OUTPUT_DIR="$GETFULLPATHRET"
declare -i BLOCK_W=0
declare -i BLOCK_H=0
declare BLOCK=""
declare OVERLAP=""
declare MODEL_FOLDER=""
declare IMAGE_FOLDER=""
declare -i TILES_PER_JOB=1
shift 2

while getopts ":m:b:o:i:T:" o; do
  case "${o}" in
    m)
      MODEL_FOLDER=${OPTARG}
      if [ ! -d "$MODEL_FOLDER" ]; then 
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
      OVERLAP="-o ${OPTARG}"
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

  if [ ! -d "$SCRIPT_DIR/CHM" ] ; then
    jobFailed "No CHM folder found.  Is this script in a job folder?"
  fi

  runCreatePreTrainedMode "$OUTPUT_DIR" "$MODEL_FOLDER" "$IMAGE_FOLDER" "$OVERLAP $BLOCK" $BLOCK_W $BLOCK_H $TILES_PER_JOB
  declare createPreTrainExit=$?

  if [ $createPreTrainExit -eq 0 ] ; then
    logEcho ""
    logMessage "Next step is to do a test run by running the following command:"
    logEcho ""
    logMessage "cd \"$OUTPUT_DIR\";./$RUN_CHM_JOB_VIA_PANFISH_SH $TESTRUN_MODE $OUTPUT_DIR"
    logEcho ""
  fi

  logEcho ""
  logEndTime "$MODE mode" $MODE_START_TIME $createPreTrainExit
  logEcho ""
  exit $createPreTrainExit
fi

# implies an unsupported mode
jobFailed "$MODE not supported.  Invoke $0 for list of valid options."

