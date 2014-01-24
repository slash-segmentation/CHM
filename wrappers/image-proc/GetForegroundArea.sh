#!/bin/bash

usage()
{
    echo "Usage:
$0 <image_path> [<bg_color>]
  image_path      The path to the grayscale image to use.
  bg_color        The background color to use. If not given then it will be
                  automatically determined by looking for strips of solid color
                  along the sides.

On success, the program has the exit code 0 and outputs the line: "#,#,#,#"
Where the numbers are the top-left and bottom-right corners of the foreground
area. On failure the exit code is 1." 1>&2;
  exit 1;
}

# Parse and minimally check arguments
ARGS=;
if [[ $# < 1 || $# > 2 ]]; then usage;
elif [[ $# > 1 ]]; then ARGS="''$1'',$2";
elif [[ $# > 0 ]]; then ARGS="''$1''";fi

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
matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('[t,l,b,r]=GetForegroundArea($ARGS);fprintf(''%d %d %d %d\n'',t,l,b,r);');" | grep -m 1 -x -E '[0-9]+ [0-9]+ [0-9]+ [0-9]+';
matlab_err=$?;


# Cleanup
stty sane # restore terminal settings
if [ -n "$MATLABPATH_ORIGINAL" ]; then export MATLABPATH=$MATLABPATH_ORIGINAL; fi

exit $matlab_err
