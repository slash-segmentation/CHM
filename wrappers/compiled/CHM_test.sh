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
  -t tile_pos     Only allowed with -b. Specifies that only the given
                  blocks/tiles be processed by CHM while all others simply
                  output black. Each tile is given as C,R (e.g 2,1 would be the
                  tile in the second column 2 and first row). Can process
                  multiple tiles by using multiple -t arguments. The tiles are
                  defined by multiples of block_size-2*overlap_size. A tile
                  position out of range will be ignored. If not included then
                  all tiles will be processed.
  -M matlab_dir   MATLAB or MCR directory. If not given will look for a MCR_DIR
                  environmental variable. If that doesn't exist then an attempt
                  will be made using 'which'. It must be the same version used
                  to compile the scripts.

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

# Functions for parsing arguments
get_size() # takes string to parse, if 0s are not allowed, and the width and height variable names (e.g. get_size ${OPTARG} 1 OVERLAP_W OVERLAP_H ) 
{
    if ! [[ $1 =~ ^([0-9]+)(x([0-9]+))?$ ]]; then echo "Invalid size: $1. Expecting single number or WxH." 1>&2; echo; usage; fi;
    local -i W=${BASH_REMATCH[1]};
    local -i H=$W;
    if [[ ! -z ${BASH_REMATCH[3]} ]]; then
        H=${BASH_REMATCH[3]};
    fi;
    if [[ $2 -ne 0 && ( $W -eq 0 || $H -eq 0 ) ]]; then echo "Invalid size: $1. Neither dimension can be 0." 1>&2; echo; usage; fi;
    eval $3=$W
    eval $4=$H
}
get_pos() # takes string to parse and the X and Y variable names (e.g. get_pos ${OPTARG} TILE_COL TILE_ROW )
{
    if ! [[ $1 =~ ^(-?[0-9]+),(-?[0-9]+)$ ]]; then echo "Invalid position: $1. Expecting COL,ROW." 1>&2; echo; usage; fi;
    local -i C=${BASH_REMATCH[1]};
    local -i R=${BASH_REMATCH[2]};
    eval $2=$C
    eval $3=$R
}

# Parse and minimally check arguments
if [[ $# -lt 2 ]]; then usage; fi
INPUT=$1;
OUTPUT=$2;
if [[ -f $OUTPUT ]]; then echo "Output directory already exists as a file." 1>&2; echo; usage; fi;
MODEL_FOLDER=./temp/;
declare -i BLOCK_W=0;
declare -i BLOCK_H=0;
declare -i OVERLAP_W=0;
declare -i OVERLAP_H=0;
declare -i TILE_ROW=0; # temporary variables
declare -i TILE_COL=0;
TILES=; # the tiles to do as row1 col1;row2 col2;...
MATLAB_FOLDER=;
shift 2
while getopts ":sm:b:o:t:M:" o; do
  case "${o}" in
    s)
      echo "Warning: argument -s is ignored for compiled version, it is always single-threaded" 1>&2;
      ;;
    m)
      MODEL_FOLDER=${OPTARG};
      if [ ! -d "$MODEL_FOLDER" ]; then echo "Model folder is not a directory." 1>&2; echo; usage; fi;
      ;;
    b)
      get_size "${OPTARG}" 1 BLOCK_W BLOCK_H
      ;;
    o)
      get_size "${OPTARG}" 0 OVERLAP_W OVERLAP_H
      ;;
    t)
      get_pos "${OPTARG}" TILE_COL TILE_ROW
      if [[ -z $TILES ]]; then
        TILES="$TILE_ROW $TILE_COL"
      else
        TILES="$TILES;$TILE_ROW $TILE_COL"
      fi
      ;;
    M)
      MTLB_FLDR=${OPTARG};
      if [ ! -d "$MTLB_FLDR" ]; then echo "MATLAB folder is not a directory." 1>&2; echo; usage; fi;
      ;;
    *)
      echo "Invalid argument: ${o}." 1>&2; echo;
      usage;
      ;;
    esac
done
if [[ ($OVERLAP_W -ne 0 || $OVERLAP_H -ne 0) && $BLOCK_W -eq 0 ]]; then echo "Overlap size can only be used with block size." 1>&2; echo; usage; fi;
if [[ ! -z $TILES && $BLOCK_W -eq 0 ]]; then echo "Tile position can only be used with block size." 2>&1; echo; usage; fi;


# Find MATLAB or MATLAB Compiler Runtime and add some paths to the LD_LIBRARY_PATH
if [[ -z $MTLB_FLDR ]]; then
    if [[ -z $MCR_DIR ]]; then
        MTLB_FLDR=`which matlab 2>/dev/null`
        if [[ $? -ne 0 ]]; then echo "Unable to find MATLAB or MATLAB Compiler Runtime." 1>&2; echo; usage; fi;
        while [ -h "$MTLB_FLDR" ]; do
            DIR="$( cd -P "$( dirname "$MTLB_FLDR" )" && pwd -P )"
            MTLB_FLDR="$(readlink "$MTLB_FLDR")"
            [[ $MTLB_FLDR != /* ]] && MTLB_FLDR="$DIR/$MTLB_FLDR"
        done
        MTLB_FLDR=`dirname "$( cd -P "$( dirname "$MTLB_FLDR" )" && pwd -P )"`
    elif [ ! -d "$MCR_DIR" ]; then echo "MCR_DIR is not a directory." 1>&2; echo; usage;
    else MTLB_FLDR=$MCR_DIR; fi
fi
if [[ ! -d $MTLB_FLDR/bin/glnxa64 ]] || [[ ! -d $MTLB_FLDR/runtime/glnxa64 ]] || [[ ! -d $MTLB_FLDR/sys/os/glnxa64 ]]; then
    echo "Unable to find MATLAB or MATLAB Compiler Runtime (thought we found $MTLB_FLDR but that wasn't it)." 1>&2; echo; usage;
fi
if [ -z $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=$MTLB_FLDR/bin/glnxa64:$MTLB_FLDR/runtime/glnxa64:$MTLB_FLDR/sys/os/glnxa64;
else
    export LD_LIBRARY_PATH=$MTLB_FLDR/bin/glnxa64:$MTLB_FLDR/runtime/glnxa64:$MTLB_FLDR/sys/os/glnxa64:$LD_LIBRARY_PATH;
fi


# Setup caching
if [ -z $MCR_CACHE_ROOT ]; then
    export MCR_CACHE_ROOT=/tmp/mcr_cache_root_$USER
    mkdir -p $MCR_CACHE_ROOT
fi


# Find where the bash script actually is so we can find the wrapped program
# This is a bit complicated since this script is actually a symlink
# See stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SOURCE="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"


# Run the main matlab script
if [[ $BLOCK_W -ne 0 ]]; then
  $SOURCE/CHM_test_blocks "${INPUT}" "${OUTPUT}" "[${BLOCK_H} ${BLOCK_W}]" "[${OVERLAP_H} ${OVERLAP_W}]" "${MODEL_FOLDER}" "[${TILES}]";
else
  $SOURCE/CHM_test "${INPUT}" "${OUTPUT}" "${MODEL_FOLDER}";
fi


# Done
exit $?;
