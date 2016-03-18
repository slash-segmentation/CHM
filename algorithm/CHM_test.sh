#!/bin/bash

usage()
{
    echo "CHM Image Testing Phase Script.  @@VERSION@@
    
$0 <input_files> <output_folder> <optional arguments>
  input_files     The input files to use. See below for the specification.
  output_folder   The folder to save the generated images to.
                  The images will have the same name and type as the input
                  files but be placed in this folder.
	
Optional Arguments:
  -m model_folder The folder that contains the model data. Default is ./temp/.
                  (contains param.mat and MODEL_level#_stage#.mat)
  -b block_size   Set the block size to use as WxH. By default the block size
                  is the the same as the size of the training images (which is
                  believed to be optimal). Old models did not include the size
                  of the training images so this argument must always be given
                  in that case.
  -o overlap_size Specifies how much the blocks should overlap given as WxH.
                  Default is 50x50. The value used will depend on the size
                  of the structures being segmenting but at most 75 pixels
                  seems necessary.
  -t tile_pos     Specifies that only the given blocks/tiles be processed by
                  CHM while all others simply output black. Each tile is given
                  as C,R (e.g 2,1 would be the tile in the second column 2 and
                  first row). Can process multiple tiles by using multiple -t
                  arguments. The tiles are defined by multiples of
                  block_size-2*overlap_size. A tile position out of range will
                  be ignored. If not included then all tiles will be processed.
  -h              Don't histogram-equalize the testing images to the training
                  image histogram (if provided in the model). This option
                  should only be used if the testing data has already been
                  equalized.
  -s              Single-thread / non-parallel. Without this each block is
                  done in parallel using all available physical cores after an
                  initial few tiles are down by them selves (for each image).

Input Files Specification
The input files can be specified in multiple ways. It needs to be one of the
these or a comma-separated list of these:
 * path to a folder            - all PNGs and TIFFs in that folder
 * path to a file              - only that file 
 * path with numerical pattern - get all files matching the pattern
     pattern must have #s in it and end with a semicolon and number range
     the #s are replaced by the values at the end with leading zeros
     example: in/####.png;5-15 would do in/0005.png through in/0015.png
     note: the semicolon needs to be escaped or in double quotes in some shells
 * path with wildcard pattern  - get all files matching the pattern
     pattern has * in it which means any number of any characters
     example: in/lbl_*.tif does all TIFF images starting with lbl_ in 'in'
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
if [[ $# -gt 2 ]] && [ "${3:0:1}" != "-" ]; then
  echo "You provided more than 2 required arguments. Did you accidently use a glob expression without escaping the asterisk?" 1>&2; echo; usage; 
fi
INPUT=$1;
OUTPUT=$2;
if [[ -f $OUTPUT ]]; then echo "Output directory already exists as a file." 1>&2; echo; usage; fi;
MODEL_FOLDER=./temp/;
SINGLE_THREAD=; # normally blank, "-nojvm" when single-threaded which disables parellism (along with other unnecessary things)
HIST_EQ=true;
declare -i BLOCK_W=0; # temporary variables
declare -i BLOCK_H=0;
BLOCKSIZE=\'\'auto\'\'; # or [${BLOCK_H} ${BLOCK_W}]
declare -i OVERLAP_W=50;
declare -i OVERLAP_H=50;
declare -i TILE_ROW=0; # temporary variables
declare -i TILE_COL=0;
TILES=; # the tiles to do as row1 col1;row2 col2;...
shift 2
while getopts ":shm:b:o:t:" o; do
  case "${o}" in
    s)
      SINGLE_THREAD=-nojvm;
      ;;
    h)
      HIST_EQ=false;
      ;;
    m)
      MODEL_FOLDER=${OPTARG};
      if [ ! -d "$MODEL_FOLDER" ]; then echo "Model folder is not a directory." 1>&2; echo; usage; fi;
      ;;
    b)
      get_size "${OPTARG}" 1 BLOCK_W BLOCK_H;
      BLOCKSIZE="[${BLOCK_H} ${BLOCK_W}]";
      ;;
    o)
      get_size "${OPTARG}" 0 OVERLAP_W OVERLAP_H;
      ;;
    t)
      get_pos "${OPTARG}" TILE_COL TILE_ROW;
      if [[ -z $TILES ]]; then
        TILES="$TILE_ROW $TILE_COL";
      else
        TILES="$TILES;$TILE_ROW $TILE_COL";
      fi
      ;;
    *)
      echo "Invalid argument: ${o}." 1>&2; echo;
      usage;
      ;;
    esac
done


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
matlab -nodisplay -singleCompThread ${SINGLE_THREAD} -r "run_from_shell('CHM_test(''${INPUT}'',''${OUTPUT}'',$BLOCKSIZE,[${OVERLAP_H} ${OVERLAP_W}],''${MODEL_FOLDER}'',[${TILES}],''${HIST_EQ}'');');";
matlab_err=$?;


# Cleanup
stty sane >/dev/null 2>&1 # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
