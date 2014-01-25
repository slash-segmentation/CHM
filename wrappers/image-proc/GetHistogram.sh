#!/bin/bash

usage()
{
    echo "Usage:
$0 <image_files> [<output_file>]
  image_files     The path(s) to the grayscale image(s) to read.
  output_file     The output file to write the histogram to. If multiple images
                  are given then the output is a sum over all of the images
                  read. If not included will be written to standard out
                  (without regular MATLAB output).
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
if [[ $# < 1 || $# > 2 ]]; then usage; fi;
INPUT="''$1''";
if [[ $# > 1 ]]; then
  OUTPUT="''$2''";
else
  OUTPUT=1;
fi


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
if [[ $OUTPUT = 1 ]]; then
  matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('WriteIntTextFile(GetHistogram(GetInputFiles(${INPUT})),${OUTPUT});');" | grep -o -E ^[0-9]+;
else
  matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('WriteIntTextFile(GetHistogram(GetInputFiles(${INPUT})),${OUTPUT});');";
fi
matlab_err=$?;


# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
