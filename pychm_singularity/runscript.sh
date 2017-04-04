#!/bin/bash

declare image_name="@@IMAGE_NAME@@"
declare version="@@VERSION@@"

declare pychm_version="@@PYCHM_VERSION@@"
declare segtools_version="@@SEGTOOLS_VERSION@@"

if [ $# -eq 0 ] ; then
  echo "$image_name <train|test|verify> <images> <labels> (options)"
  echo ""
  echo "This singularity image runs PyCHM ($pychm_version) train or test"
  echo "using included segtools ($segtools_version)"
  echo ""
  echo "  The mode is determined by the first argument <train|test|verify>"
  echo "  If first argument is:"
  echo ""
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

if [ "$mode" == "verify" ] ; then
  exec python -m pysegtools.imstack_main --check
  exit $?
fi

if [ "$mode" == "train" ] ; then
  exec python -m chm.train "$@"
  exit $?
fi

if [ "$mode" == "test" ] ; then
  exec python -m chm.test "$@"
  exit $?
fi

echo "Inavlid mode: $mode: Run $image_name with no arguments for help"
exit 6
