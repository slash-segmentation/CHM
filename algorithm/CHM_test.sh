#!/bin/bash

usage()
{
    echo "CHM Image Testing Phase Script.
    
CHM_test <input_files> <output_folder> <optional arguments>
  input_files     The input files to use. See below for the specification.
  output_folder   The folder to save the generated images to
                  The images will have the same name and type as the input
                  files but be placed in this folder
	
Optional Arguments:
  -m model_folder The folder that contains the model data
                  (contains param.mat and MODEL_level#_stage#.mat)
  -b block_size   Process images in blocks of this size instead of all at once.
                  Can be given as a single value (used for both height and
                  width) or a WxH value. This can reduce processing time and
                  memory usage along with increasing quality. The block size
                  should be exactly the size of the training images. TIFF
                  images are particularly faster and use less memory with this
                  method. TIFF images that have a width and height that are
                  multiples of block_size-2*overlap_size use a lot less memory
                  but will create uncompressed outputs. When using blocks,
                  parallelism is done on blocks instead of images.
  -o overlap_size Only allowed with -b. Specifies how much the blocks should
                  overlap (default is none). Like -b this supports a single
                  value or a WxH value. The value used will depend on the size
                  of the structures being segmenting but 100-200 pixels seem
                  good in general.
  -s              Single-thread / non-parallel. Normally the images/blocks are
                  processed in parallel using all available physical cores. 

Input Files Specification
For the many-files version the input files can be specified in multiple ways.
It needs to be one of the these or a comma-separated list of these:
 * path to a folder            - all PNGs in that folder
 * path to a file              - only that file 
 * path with numerical pattern - get all files matching the pattern
     pattern must have #s in it and end with a semicolon and number range
     the #s are replaced by the values at the end with leading zeros
     example: in/####.png;5-15 would do in/0005.png through in/0015.png
     note: the semicolon needs to be escaped or in double quotes in some shells
 * path with wildcard pattern  - get all files matching the pattern
     pattern has * in it which means any number of any characters
     example: in/*.tif does all TIFF images in that directory
     note: the asterisk needs to be escaped or in double quotes in some shells" 1>&2;
  exit 1;
}

# Parse and minimally check arguments
if [[ $# < 2 ]]; then usage; fi
INPUT=$1;
OUTPUT=$2;
if [[ -f $OUTPUT ]]; then echo "Output directory already exists as a file." 2>&1; echo; usage; fi;
MODEL_FOLDER=./temp/;
SINGLE_THREAD=; # normally blank, "-nojvm" when single-threaded which disables parellism (along with other unnecessary things)
declare -i COMMA=0;
declare -i BLOCK_SIZE_X=0;
declare -i BLOCK_SIZE_Y=0;
declare -i OVERLAP_SIZE_X=0;
declare -i OVERLAP_SIZE_Y=0;
shift 2
while getopts ":sm:b:o:" o; do
  case "${o}" in
    s)
      SINGLE_THREAD=-nojvm;
      ;;
    m)
      MODEL_FOLDER=${OPTARG};
      if [ ! -d "$MODEL_FOLDER" ]; then echo "Model folder is not a directory." 2>&1; echo; usage; fi;
      ;;
    b)
      COMMA=`expr index "${OPTARG}" "x"`;
      if [[ $COMMA != 0 ]]; then
        BLOCK_SIZE_X=${OPTARG:0:$COMMA-1};
        BLOCK_SIZE_Y=${OPTARG:$COMMA};
      else
        BLOCK_SIZE_X=${OPTARG};
        BLOCK_SIZE_Y=${OPTARG};
      fi;
      if (( $BLOCK_SIZE_X <= 0 || $BLOCK_SIZE_Y <= 0 )); then echo "Invalid block size." 2>&1; echo; usage; fi;
      ;;
    o)
      COMMA=`expr index "${OPTARG}" "x"`;
      if [[ $COMMA != 0 ]]; then
        OVERLAP_SIZE_X=${OPTARG:0:$COMMA-1};
        OVERLAP_SIZE_Y=${OPTARG:$COMMA};
      else
        OVERLAP_SIZE_X=${OPTARG};
        OVERLAP_SIZE_Y=${OPTARG};
      fi;
      if (( $OVERLAP_SIZE_X < 0 || $OVERLAP_SIZE_Y < 0 )); then echo "Invalid overlap size." 2>&1; echo; usage; fi;
      ;;
    *)
      echo "Invalid argument." 2>&1; echo; 
      usage;
      ;;
    esac
done
if [[ ($OVERLAP_SIZE_X != 0 || $OVERLAP_SIZE_Y != 0) && $BLOCK_SIZE_X == 0 ]]; then echo "Overlap size can only be used with block size." 2>&1; echo; usage; fi;


# We need to add the path with the script in it to the MATLAB path
# This is a bit complicated since this script is actually a symlink
# See stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
if [ -n "$MATLABPATH" ]; then
  MATLABPATH_ORIGINAL=$MATLABPATH
  export MATLABPATH="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )":$MATLABPATH
else
  export MATLABPATH="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"
fi


# Run the main matlab script
if [[ $BLOCK_SIZE_X != 0 ]]; then
  matlab -nodisplay -singleCompThread ${SINGLE_THREAD} -r "run_from_shell('CHM_test_blocks(''${INPUT}'',''${OUTPUT}'',[${BLOCK_SIZE_Y} ${BLOCK_SIZE_X}],[${OVERLAP_SIZE_Y} ${OVERLAP_SIZE_X}],''${MODEL_FOLDER}'');');";
else
  matlab -nodisplay -singleCompThread ${SINGLE_THREAD} -r "run_from_shell('CHM_test(''${INPUT}'',''${OUTPUT}'',''${MODEL_FOLDER}'');');";
fi
matlab_err=$?;


# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
