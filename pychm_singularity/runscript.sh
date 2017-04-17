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
  echo "  The mode is determined by the first argument <-m|train|test|verify>"
  echo "  If first argument is:"
  echo ""
  echo "    -m    -- Invokes python passing -m and all other args"
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

export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:${LD_LIBRARY_PATH}

# if first argument is -m then assume
# user wants to invoke a python module
# just pass this to python
if [ "$mode" == "-m" ] ; then
  exec python -m "$@"
fi

if [ "$mode" == "verify" ] ; then
  echo ""
  echo "Running segtools check"
  echo ""
  python -m pysegtools.imstack_main --check
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
  exec python -m chm.train "$@"
fi

if [ "$mode" == "test" ] ; then
  exec python -m chm.test "$@"
fi

echo "Invalid mode: $mode: Run $image_name with no arguments for help"
exit 6
