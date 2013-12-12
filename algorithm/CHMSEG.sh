#!/bin/csh -vx

if ( $#argv < 4 ) then
  echo 'usage: ' $0 ' trainfolder labelfolder testfolder outputfolder'
  exit
endif


set trainfolder = $1
set labelfolder = $2
set testfolder = $3
set testout = $4

set trainF="'${trainfolder}'"
set labelF="'${labelfolder}'"
set testF="'${testfolder}'"
set testO="'${testout}'"


/ncmir/local.linux.amd64/bin/matlab -nodisplay -singleCompThread -r 'TrainScript('${trainF},${labelF},${testF},${testO}'); quit'


