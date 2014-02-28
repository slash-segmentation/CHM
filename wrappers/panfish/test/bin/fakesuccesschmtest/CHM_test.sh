#!/bin/bash

# echo arguments
echo $*

if [ -z "$SKIP_COPY" ] ; then
  # create fake output image in temp directory
  fakeImageName=`echo $1 | sed "s/^.*\///"`
  echo "fakey" > ${2}/$fakeImageName
fi

exit 0
