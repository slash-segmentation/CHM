#!/bin/bash

declare -r CREATE_PRETRAIN_MODE="createpretrained"
declare -r SOURCE_MODE="source"
function usage() {

  echo -e "Create CHM job.

This program packages and runs CHM on grid compute resources via Panfish.

$0 <mode> <run_folder> <arguments & optional arguments>
  mode         Mode to run.  This script has several modes
               $CREATE_PRETRAIN_MODE -- Creates the job (requires -m,-b,-i)
               $SOURCE_MODE -- Allows this script to be sourced.  Used
               for unit testing.
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
  -h               Don't histogram-equalize the testing images to the training
                   image histogram (if provided in the model). This option
                   should only be used if the testing data has already been
                   equalized.

"
  exit 1
}

# 
# Creates merge tiles output directories
# createImageOutputDirectories(jobDir)
#
# Creates this directory structure
# <jobdir>/
#          runmergetilesout/
#                           mergedimages/
#                           stderr/
#                           stdout/
#
# If directory does not exist 1 is returned and if any of the
# make dir calls fails the function exits completely
#
function createMergeTilesOutputDirectories {
  local jobDir=$1

  if [ ! -d "$jobDir" ] ; then
     return 1
  fi

  makeDirectory "$jobDir/$MERGE_TILES_OUT_DIR_NAME/$MERGED_IMAGES_OUT_DIR_NAME"

  makeDirectory "$jobDir/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME"

  makeDirectory "$jobDir/$MERGE_TILES_OUT_DIR_NAME/$STD_ERR_DIR_NAME"

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
# Creates output directories based on list of input images
# createImageOutputDirectories(output directory,image directory, image suffix)
function createImageOutputDirectories {
  local outDir=$1
  local imageDir=$2
  local imageSuffix=$3

  if [ ! -d "$outDir" ] ; then
     logWarning "Output directory $outDir is not a directory"
     return 1
  fi

  if [ ! -d "$imageDir" ] ; then
     logWarning "Image directory $imageDir is not a directory"
     return 1
  fi

  for z in `find $imageDir -name "*.${imageSuffix}" -type f` ; do
    imageName=`echo $z | sed "s/^.*\///"`
    makeDirectory "$outDir/${imageName}.${IMAGE_TILE_DIR_SUFFIX}"
  done

  return 0
}


#
# Given dimensions of an image along with tile size this function returns
# the number of tiles in horizontal (TILES_W) and vertical (TILES_H) 
# that are needed to cover the image
#
function calculateTilesFromImageDimensions {
  local width=$1
  local height=$2
  local blockWidth=$3
  local blockHeight=$4

  if [ $width -le 0 ] ; then
    logWarning "Width must be larger then 0"
    return 1
  fi

  if [ $height -le 0 ] ; then
    logWarning "Height must be larger then 0"
    return 1
  fi

  if [ $blockWidth -le 0 ] ; then
    logWarning "BlockWidth must be larger then 0"
    return 1
  fi

  if [ $blockHeight -le 0 ] ; then
    logWarning "BlockHeight must be larger then 0"
    return 1
  fi

  TILES_W=`echo "scale=0;($width + $blockWidth - 1)/ $blockWidth" | bc -l`
  TILES_H=`echo "scale=0;($height + $blockHeight - 1)/ $blockHeight" | bc -l`
  return 0
}

#
# This function creates a configuration file to run CHM on each tile of each
# image.  It is assumed the images all have the same size and need every tile
# processed.  The code will create an entry for every tile in this format:
# JOBID:::<full path to input image ie /home/foo/histeq_1.png>
# JOBID:::<chmOpts ie overlap> -t R,C 
# JOBID:::<output image path relative to job directory ie out/histeq_1.png/RxC.png>
#
function createCHMTestConfig {
  local imageDir=$1
  local configFile=$2
  local tilesW=$3
  local tilesH=$4
  local tilesPerJob=$5
  local chmOpts=$6
  local modelDir=$7
  local imageSuffix=$8
  local cntr=0
  local outputFile=""

  # Calculate the tile sets we will be using
  local allTiles=() # simply a list of a tiles

  for c in `seq 1 $tilesW`; do
    for r in `seq 1 $tilesH`; do
      allTiles[$cntr]="-t $c,$r"
      let cntr++
    done
  done

  local tsCntr=0
  # Batch tiles by $tilesPerJob into new array named $tileSets
  let allTilesIndex=${#allTiles[@]}-1

  local tileSets=()
  while [ $allTilesIndex -ge 0 ] ; do
    for (( i=0 ; i < ${tilesPerJob} ; i++ )) ; do
      if [ $allTilesIndex -lt 0 ] ; then
        break
      fi
      tileSets[$tsCntr]="${tileSets[$tsCntr]} ${allTiles[$allTilesIndex]}"
      let allTilesIndex--
    done
    let tsCntr++

  done
  # Using tileSets array generate jobs for each image
  # Each job consists of 3 config lines
  # <JOBID>:::<Input image full path>
  # <JOBID>:::<Model directory full path>
  # <JOBID>:::<CHM options and tile flags from tileSets>
  # <JOBID>:::<relative output path for image of format out/[image name]/[JOBID].[image suffix]
  let cntr=1
  for z in `find $imageDir -name "*.${imageSuffix}" -type f | sort -n` ; do
    imageName=`echo $z | sed "s/^.*\///"`
    for (( y=0 ; y < ${#tileSets[@]} ; y++ )) ; do
      echo "${cntr}${CONFIG_DELIM}${z}" >> "$configFile"
      echo "${cntr}${CONFIG_DELIM}${modelDir}" >> "$configFile"
      echo "${cntr}${CONFIG_DELIM}${chmOpts} ${tileSets[$y]}" >> "$configFile"
      outputFile="${OUT_DIR_NAME}/${imageName}.${IMAGE_TILE_DIR_SUFFIX}/${cntr}.${imageSuffix}" >> "$configFile"
      echo "${cntr}${CONFIG_DELIM}$outputFile" >> "$configFile"
      let cntr++
    done
  done

  return 0
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


  # try to make merge tiles directories
  createMergeTilesOutputDirectories "$outputDir"
 
  logMessage "Copying scripts to $outputDir"
  # Copy over runCHM.sh script
  /bin/cp "$SCRIPTS_SUBDIR/$RUN_CHM_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$outputDir/$RUN_CHM_SH\" \"$outputDir/.\""
  fi

  # Copy over runMergeTiles.sh script
  /bin/cp "$SCRIPTS_SUBDIR/$RUN_MERGE_TILES_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$outputDir/$RUN_MERGE_TILES_SH\" \"$outputDir/.\""
  fi

  # Copy over runMergeTilesViaPanfish.sh script
  /bin/cp "$SCRIPTS_SUBDIR/$RUN_MERGE_TILES_VIA_PANFISH_SH" "$outputDir/."

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp \"$outputDir/$RUN_MERGE_TILES_VIA_PANFISH_SH\" \"$outputDir/.\""
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


  for Y in `echo png tif tiff` ; do
    createCHMTestConfig "$imageDir" "$configFile" "$tilesW" "$tilesH" "$tilesPerJob" " $chmOpts" "$modelDir" "$Y"
    createImageOutputDirectories  "$outputDir/${OUT_DIR_NAME}" "$imageDir" "$Y"
  done  

  # create the config needed to merge the tiles back
  createMergeTilesConfig "$outputDir"

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
declare -i BLOCK_W=0
declare -i BLOCK_H=0
declare BLOCK=""
declare OVERLAP=""
declare -i OVERLAP_W=0
declare -i OVERLAP_H=0
declare MODEL_FOLDER=""
declare IMAGE_FOLDER=""
declare HIST_FLAG=""
declare -i TILES_PER_JOB=1
shift 2

while getopts ":hm:b:o:i:T:" o; do
  case "${o}" in
    h)
      HIST_FLAG="-h"
      ;;
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

  runCreatePreTrainedMode "$OUTPUT_DIR" "$IMAGE_FOLDER" "$OVERLAP $BLOCK $HIST_FLAG" "$MODEL_FOLDER" $BLOCK_W $BLOCK_H $OVERLAP_W $OVERLAP_H $TILES_PER_JOB
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

