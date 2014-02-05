#!/bin/bash

usage()
{
    echo "CHM Image Testing Phase Script.
    
$0 <input_files> <output_folder> <optional arguments>
  input_files     The input files to use. See below for the specification.
  output_folder   The folder to save the generated images to
                  The images will have the same name and type as the input
                  files but be placed in this folder
	
Optional Arguments:
  -m model_folder The folder that contains the model data. Default is ./temp/.
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
                  of the structures being segmenting but at most 50 pixels
                  seems necessary.
  -M matlab_dir   MATLAB 2011b directory. If not given will look for a MCR_DIR
                  environmental variable. If that doesn't exist then an attempt
                  will be made using 'which'.

Input Files Specification
The input files can be specified in multiple ways. It needs to be one of the
these or a comma-separated list of these:
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
if [[ $# -lt 2 ]]; then usage; fi
INPUT=$1;
OUTPUT=$2;
if [[ -f $OUTPUT ]]; then echo "Output directory already exists as a file." 1>&2; echo; usage; fi;
MODEL_FOLDER=./temp/;
declare -i COMMA=0;
declare -i BLOCK_SIZE_X=0;
declare -i BLOCK_SIZE_Y=0;
declare -i OVERLAP_SIZE_X=0;
declare -i OVERLAP_SIZE_Y=0;
MATLAB_FOLDER=;
shift 2
while getopts ":sm:b:o:M:" o; do
  case "${o}" in
    s)
      echo "Warning: argument -s is ignored for compiled version, it is always single-threaded" 1>&2;
      ;;
    m)
      MODEL_FOLDER=${OPTARG};
      if [ ! -d "$MODEL_FOLDER" ]; then echo "Model folder is not a directory." 1>&2; echo; usage; fi;
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
      if (( $BLOCK_SIZE_X <= 0 || $BLOCK_SIZE_Y <= 0 )); then echo "Invalid block size." 1>&2; echo; usage; fi;
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
      if (( $OVERLAP_SIZE_X < 0 || $OVERLAP_SIZE_Y < 0 )); then echo "Invalid overlap size." 1>&2; echo; usage; fi;
      ;;
    M)
      MTLB_FLDR=${OPTARG};
      if [ ! -d "$MTLB_FLDR" ]; then echo "MATLAB folder is not a directory." 1>&2; echo; usage; fi;
      ;;
    *)
      echo "Invalid argument." 1>&2; echo; 
      usage;
      ;;
    esac
done
if [[ ($OVERLAP_SIZE_X != 0 || $OVERLAP_SIZE_Y != 0) && $BLOCK_SIZE_X == 0 ]]; then echo "Overlap size can only be used with block size." 1>&2; echo; usage; fi;


# Find MATLAB or MATLAB Compiler Runtime and add some paths to the LD_LIBRARY_PATH
if [[ -z $MTLB_FLDR ]]; then
    if [[ -z $MCR_DIR ]]; then
        MTLB_FLDR=`which MATLAB 1>/dev/null 2>&1`
        if [[ $? != 0 ]]; then echo "Unable to find MATLAB or MATLAB Compiler Runtime." 1>&2; echo; usage; fi;
        MTLB_FLDR=$( dirname $MTLB_FLDR )
    elseif [ ! -d "$MCR_DIR" ]; then echo "MCR_DIR is not a directory." 1>&2; echo; usage;
    else; MTLB_FLDR=$MCR_DIR; fi
fi
if [[ ! -d $MTLB_FLDR/bin/glnxa64 ]] || [[ ! -d $MTLB_FLDR/runtime/glnxa64 ]] || [[ ! -d $MTLB_FLDR/sys/os/glnxa64 ]]; then
    echo "Unable to find MATLAB or MATLAB Compiler Runtime (thought we found $MTLB_FLDR but that wasn't it)." 1>&2; echo; usage; fi;
fi
if [ -z $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=$MTLB_FLDR/bin/glnxa64:$MTLB_FLDR/runtime/glnxa64:$MTLB_FLDR/sys/os/glnxa64;
else
    export LD_LIBRARY_PATH=$MTLB_FLDR/bin/glnxa64:$MTLB_FLDR/runtime/glnxa64:$MTLB_FLDR/sys/os/glnxa64:$LD_LIBRARY_PATH;
fi


# Setup caching
export MCR_CACHE_ROOT=/tmp/mcr_cache_root_$USER
mkdir -p $MCR_CACHE_ROOT


# Find where the bash script actually is so we can find the wrapped program
# This is a bit complicated since this script is actually a symlink
# See stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done


# Run the main matlab script
if [[ $BLOCK_SIZE_X != 0 ]]; then
  $SOURCE/CHM_test_blocks "${INPUT}" "${OUTPUT}" "[${BLOCK_SIZE_Y} ${BLOCK_SIZE_X}]" "[${OVERLAP_SIZE_Y} ${OVERLAP_SIZE_X}]" "${MODEL_FOLDER}";
else
  $SOURCE/CHM_test "${INPUT}" "${OUTPUT}" "${MODEL_FOLDER}";
fi


# Done
exit $?;
