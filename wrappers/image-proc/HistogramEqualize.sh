#!/bin/bash

usage()
{
    echo "Usage:
$0 <image_in> [<histogram_file>] <image_out>
  image_in        The path to the grayscale image to read.
  histogram_file  The file to read that contains the histogram to equalize to.
                  Should be a file with 256 integers on seperate lines. Use -
                  to read from stdin. If not included at all, the histogram
                  will be equalized to a uniform histogram.
  image_out       The path to the grayscale image to write." 1>&2;
  exit 1;
}

# Parse and minimally check arguments
if [[ $# < 2 || $# > 3 ]]; then usage; fi;
INPUT="''$1''";
if [[ $# > 2 ]]; then
  if [[ $2 = '-' ]]; then
    X=`cat | tr -s [:space:] ' ' | sed -e 's/^ *//g' -e 's/ *$//g'`;
    echo "$X" | grep -x -E "([0-9]+ ){255}[0-9]+" >/dev/null 2>&1;
    if [[ $? -ne 0 ]]; then echo "STDIN was not 256 integers" 1>&2; echo; usage; fi;
    HISTOGRAM=",[${X}]"
  else
    HISTOGRAM=",ReadIntTextFile(''$2'')";
  end
  OUTPUT="''$3''";
else
  OUTPUT="''$2''";
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
matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('imwrite(HistogramEqualize(${INPUT}${HISTOGRAM}),${OUTPUT});');";
matlab_err=$?;


# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
