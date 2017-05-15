#!/bin/bash

declare image_name="@@IMAGE_NAME@@"
declare version="@@VERSION@@"

declare pychm_version="@@PYCHM_VERSION@@"
declare segtools_version="@@SEGTOOLS_VERSION@@"

if [ $# -eq 0 ] ; then
  echo "$image_name <train|test|verify> <images> <labels> (options)"
  echo "$image_name -m <python module> (options)"
  echo ""
  echo "This singularity image runs PyCHM ($pychm_version) train or test"
  echo "using included segtools ($segtools_version)"
  echo ""
  echo "  The mode is determined by the first argument "
  echo "    <-m|<file>|train|test|verify>"
  echo "  If first argument is:"
  echo ""
  echo "    -m    -- Invokes python passing -m and all other args"
  echo "    <file> -- Runs PyCHM Test in Probability Map Viewer Mode"
  echo "    train -- Runs PyCHM Train"
  echo "    test  -- Runs PyCHM Test"
  echo "    verify -- Runs a quick test to verify PyCHM Train & Test work"
  echo "    version -- Outputs version of PyCHM singularity image"
  echo ""
  echo "  For more information on how this image was created please visit:"
  echo "  https://github.com/slash-segmentation/CHM/pychm_singularity"
  echo ""
  exit 1
fi

declare mode=$1

# remove the first argument
shift

if [ "$mode" == "version" ] ; then
  echo "$version"
  exit 0
fi

P_CMD="python -s"

export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:${LD_LIBRARY_PATH}

# to make sure we are calling correct python
# in case someone set an alternate one in their PATH
# environment
export PATH=/usr/bin:$PATH

# if first argument is -m then assume
# user wants to invoke a python module
# just pass this to python
if [ "$mode" == "-m" ] ; then
  exec $P_CMD -m "$@"
fi

if [ "$mode" == "verify" ] ; then
  echo ""
  echo "Running segtools check"
  echo ""
  $P_CMD -m pysegtools.imstack_main --check
  if [ $# -ne 1 ] ; then
    echo "verify mode requires <directory> as second argument to run test PyCHM train job"
    exit 1
  fi
  echo ""
  echo "Running small PyCHM train job which writes output to $1/foo.out" 
  echo ""
  $0 train $1/foo.out "[ /pychmtestdata/images/*.png ]" "[ /pychmtestdata/labels/*.png ]" -S 2 -L 1 
  echo ""
  echo "Running small PyCHM test job which writes output to $1/foo.png"
  echo ""
  exec $0 test $1/foo.out /pychmtestdata/5.png $1/foo.png
fi

if [ "$mode" == "train" ] ; then
  exec $P_CMD -m chm.train "$@"
fi

if [ "$mode" == "test" ] ; then
  exec $P_CMD -m chm.test "$@"
fi

if [ -f "$mode" ] ; then
  if [ $# -ne 2 ] ; then
    echo ""
    echo "$0 <input tile> <output tile> <model directory>"
    echo "In this mode PyCHM test is run on <input tile> image file using the model"
    echo "specified by <model directory>. The overlap is "
    echo "set to 0x0 and block/tile size is set to size of <input tile> image."
    echo "After CHM is run, the Image Magick convert command is used to threshold the"
    echo "image using the flag -threshold 30% to zero out intensities below this" 
    echo "threshold. The resulting image is stored in <output tile> image file as a png"
    echo ""
    exit 3
  fi

  input="$mode"
  output="$1"
  outputdir=`dirname $output`
  model="$2"
  echo "Running $0 test $model $input $output"
  $0 test "$model" "$input" "$output"
  outimagetmp="${output}.tmp.png"
  echo "Running convert: $output -threshold 30% $outimagetmp"
  convert "$output" -threshold 30% "$outimagetmp"
  echo "Running mv $outimagetmp $output"
  mv "$outimagetmp" "$output"
  exit $?
fi


echo "Invalid mode: $mode: Run $image_name with no arguments for help"
exit 6
