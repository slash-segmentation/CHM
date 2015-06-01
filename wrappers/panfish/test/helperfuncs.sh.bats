#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/helperfuncs" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  export TESTIMAGE="${BATS_TEST_DIRNAME}/createchmjob/600by400.png" 
  export TESTIMAGE_DIR="${BATS_TEST_DIRNAME}/createchmjob"
  export HELPERFUNCS="$THE_TMP/.helperfuncs.sh"
  export PANFISH_TEST_BIN="${BATS_TEST_DIRNAME}/bin/panfish"
  unset SGE_TASK_ID
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  echo "end of tear down" 1>&2
}


#
# parseProperties() tests
#
@test "parseProperties() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test where no properties file is found
  run parseProperties "$THE_TMP" "$THE_TMP/s"

  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "  Config $THE_TMP/panfishCHM.properties not found" ]


  echo "panfish.bin.dir=/xx" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=/matlab" >> "$THE_TMP/panfishCHM.properties"
  echo "image.magick.bin.dir=/foo" >> "$THE_TMP/panfishCHM.properties"
  echo "image.magick.convert.opts=-limit 4gb -define registry:temporary-path=/data/tmp" >> "$THE_TMP/panfishCHM.properties"
  echo "batch.and.walltime.args=batch" >> "$THE_TMP/panfishCHM.properties"
  echo "cluster.list=foo.q" >> "$THE_TMP/panfishCHM.properties"
  echo "chm.bin.dir=/bin/chm" >> "$THE_TMP/panfishCHM.properties"
  echo "max.retries=3" >> "$THE_TMP/panfishCHM.properties"
  echo "retry.sleep=10" >> "$THE_TMP/panfishCHM.properties"
  echo "job.wait.sleep=20" >> "$THE_TMP/panfishCHM.properties"
  echo "land.job.options=ljo" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.job.options=cjo" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.image.options=-b --deletebefore -x *.*" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.model.options=--exclude *.foo" >> "$THE_TMP/panfishCHM.properties"

 
  # test with valid complete config
  parseProperties "$THE_TMP" "$THE_TMP/s"
  
  [ "$PANFISH_BIN_DIR" == "/xx" ]
  [ "$MATLAB_DIR" == "/matlab" ]
  [ "$BATCH_AND_WALLTIME_ARGS" == "batch" ]
  [ "$CHUMMEDLIST" == "foo.q" ]
  [ "$CHM_BIN_DIR" == "/bin/chm" ]
  [ "$MAX_RETRIES" == "3" ]
  [ "$RETRY_SLEEP" == "10" ]
  [ "$WAIT_SLEEP_TIME" == "20" ]

  [ "$LAND_JOB_OPTS" == "ljo" ]
  [ "$CHUM_JOB_OPTS" == "cjo" ]
  [ "$CHUM_IMAGE_OPTS" == "-b --deletebefore -x *.*" ]
  [ "$CHUM_MODEL_OPTS" == "--exclude *.foo" ]

  [ "$CASTBINARY" == "/xxpanfishcast" ]
  [ "$CHUMBINARY" == "/xxpanfishchum" ]
  [ "$LANDBINARY" == "/xxpanfishland" ]
  [ "$PANFISHSTATBINARY" == "/xxpanfishstat" ]
  [ "$IMAGE_MAGICK_BIN_DIR" == "/foo" ] 
  [ "$IMAGE_MAGICK_CONVERT_OPTS" == "-limit 4gb -define registry:temporary-path=/data/tmp" ] 
  [ "$IDENTIFY_CMD" == "/foo/identify" ] 
  [ "$CONVERT_CMD" == "/foo/convert" ]
}

#
# Test parse Properties with image magick opts unset
#
@test "parseProperties() tests with image magick opts unset" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test where no properties file is found
  run parseProperties "$THE_TMP" "$THE_TMP/s"

  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "  Config $THE_TMP/panfishCHM.properties not found" ]

  echo "image.magick.bin.dir=" > "$THE_TMP/panfishCHM.properties"
  echo "image.magick.convert.opts=" >> "$THE_TMP/panfishCHM.properties"


  # test with valid complete config
  parseProperties "$THE_TMP" "$THE_TMP/s"

  [ "$IMAGE_MAGICK_BIN_DIR" == "" ]
  [ "$IMAGE_MAGICK_CONVERT_OPTS" == "" ]
  [ "$IDENTIFY_CMD" == "identify" ]
  [ "$CONVERT_CMD" == "convert" ]
}


#
# parseWidthHeightParameter() with various values
#
@test "parseWidthHeightParameter() with various values" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  parseWidthHeightParameter "200x100"
 
  [ $? -eq 0 ] 
  [ "$PARSED_WIDTH" -eq 200 ]
  [ "$PARSED_HEIGHT" -eq 100 ]

  parseWidthHeightParameter "20"

  [ $? -eq 0 ]
  [ "$PARSED_WIDTH" -eq 20 ]
  [ "$PARSED_HEIGHT" -eq 20 ]

}

# 
# getImageDimensions() tests
#
@test "getImageDimensions() tests" {
  run which identify

  if [ "$status" -ne 0 ] ; then
    skip "No Image Magick identify command found"
  fi 

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  getImageDimensions "$TESTIMAGE"
  [ $? -eq 0 ]
  [ "$PARSED_WIDTH" -eq 600 ]
  [ "$PARSED_HEIGHT" -eq 400 ]

  # try on file that does not exist
  run getImageDimensions "$THE_TMP"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP is not a file" ]

  # try on non image file
  run getImageDimensions "$HELPERFUNCS"

  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == "WARNING:    Unable to parse block parameter:"* ]]

}

@test "getImageDimensionsFromDirOfImages() tests" {
  run which identify

  if [ "$status" -ne 0 ] ; then
    skip "No Image Magick identify command found"
  fi

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  run getImageDimensionsFromDirOfImages "$TESTIMAGE" "png"
  
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $TESTIMAGE is not a directory" ]

  # try on file that does not exist
  run getImageDimensionsFromDirOfImages "$THE_TMP/doesnotexist" "png"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP/doesnotexist is not a directory" ]

  mkdir -p "$THE_TMP/emptydir" 1>&2 

  # try on directory with no images
  run getImageDimensionsFromDirOfImages "$THE_TMP/emptydir" "png"

  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    No images found in $THE_TMP/emptydir with suffix png" ]


  getImageDimensionsFromDirOfImages "$TESTIMAGE_DIR" "png"

  [ $? -eq 0 ] 
  [ "$PARSED_WIDTH" -eq 600 ]
  [ "$PARSED_HEIGHT" -eq 400 ]
}

#
# verifyResults() tests
#
@test "verifyResults() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

# checkSingleTask that is always happy
echo "function checkSingleTask {
  echo "\$*" >> "\$THE_TMP/checkSingleTask.out"
  return 0
}
" > "$THE_TMP/myscript.sh"

  . "$THE_TMP/myscript.sh"

  # Test with 1 job successful no file
  verifyResults "foo" "1" "$THE_TMP" "1" "1" "no" "failed" "fail.tmp.jobs" "failed.jobs"
  [ "$?" -eq 0 ] 
  [ "$NUM_FAILED_JOBS" == "0" ]
  aLine=`cat "$THE_TMP/checkSingleTask.out"`
  [ "$aLine" == "foo $THE_TMP 1" ]
  
  /bin/rm -f "$THE_TMP/checkSingleTask.out" 1>&2
  # Test with 3 jobs successful with file
  verifyResults "foo" "1" "$THE_TMP" "1" "3" "yes" "failed" "fail.tmp.jobs" "failed.jobs"
  [ $? -eq 0 ]
  [ "$NUM_FAILED_JOBS" -eq 0 ]
  [ ! -e "$THE_TMP/failed.jobs" ] 
  numLines=`wc -l "$THE_TMP/checkSingleTask.out" | sed "s/ .*//"`
  [ $numLines -eq 3 ]

# checkSingleTask that always fails
echo "function checkSingleTask {
  return 1
}
" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"

  # Test with 1 job failure no file
  run verifyResults "foo" "1" "$THE_TMP" "1" "3" "no" "failed" "fail.tmp.jobs" "failed.jobs"
  [ "$status" -eq 1 ]
  [ ! -e "$THE_TMP/failed.jobs" ]


# checkSingleTask that fails on first job
echo "function checkSingleTask {
  local task=\$1
  local jobDir=\$2
  local taskId=\$3
  if [ \$taskId -eq 1 ] ; then
    return 1
  fi
  return 0
}
" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"

  echo "hello" > "$THE_TMP/failed.jobs"
  # Test with 1 job failure with file and failed file exists 
  run verifyResults "foo" "1" "$THE_TMP" "1" "3" "yes" "failed" "fail.tmp.jobs" "failed.jobs"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ -s "$THE_TMP/failed.jobs" ]
  [ -s "$THE_TMP/failed.0.jobs" ]

  aLine=`head -n 1 "$THE_TMP/failed.jobs"` 
  [ "$aLine" == "1" ]  
  numLines=`wc -l "$THE_TMP/failed.jobs" | sed "s/ .*//"`
  [ $numLines -eq 1 ]

}


#
# getSizeOfPath() tests
#
@test "getSizeOfPath() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test non existant file
  run getSizeOfPath "$THE_TMP/doesnotexist"
  [ "$status" -eq 1 ]

  # Test on file
  echo "hi" > "$THE_TMP/hi"
  getSizeOfPath "$THE_TMP/hi"
 
  [ $? -eq 0 ]
  [ "$NUM_BYTES" -ge 3 ]

  getSizeOfPath "$THE_TMP"

  [ $? -eq 0 ]
  [ "$NUM_BYTES" -ge 20000 ]

}

#
# getFullPath() tests
#
@test "getFullPath() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  
  # Test full path on path that is already full
  getFullPath "$THE_TMP"
  
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP" ]

  curdir=`pwd`
  cd $THE_TMP
  
  # try .
  getFullPath "."
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP" ]

  # try relative path
  mkdir -p "$THE_TMP/foo"
  getFullPath "foo"
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP/foo" ]

  # try another relative path
  getFullPath "foo/.."
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP" ]

  cd $curdir

}

#
# moveOldClusterFolders() tests
#
@test "moveOldClusterFolders() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where CHUMMEDLIST is empty
  export CHUMMEDLIST=""
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]

  # Test with CHUMMEDLIST not empty (one entry) but no folders exist
  export CHUMMEDLIST="foo_shadow.q"
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]

  # Test with CHUMMEDLIST not empty (two entries) but no folders exist
  export CHUMMEDLIST="foo_shadow.q,blah_shadow.q"
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]

  
  # Test with CHUMMEDLIST not empty (one entry) and folders exist
  export CHUMMEDLIST="foo_shadow.q"
  mkdir -p "$THE_TMP/foo_shadow.q" 1>&2
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]
  [ ! -d "$THE_TMP/foo_shadow.q" ]
  [ -d "$THE_TMP/foo_shadow.q.1.old" ] 


  # Test with CHUMMEDLIST not empty (two entries) and folders exist
  export CHUMMEDLIST="foo_shadow.q,blah_shadow.q"
  mkdir -p "$THE_TMP/foo_shadow.q" 1>&2
  mkdir -p "$THE_TMP/blah_shadow.q" 1>&2
  run moveOldClusterFolders "3" "$THE_TMP"
  [ "$status" -eq 0 ]
  ls -la "$THE_TMP" 1>&2
  [ ! -d "$THE_TMP/foo_shadow.q" ]
  [ -d "$THE_TMP/foo_shadow.q.3.old" ] 
  [ ! -d "$THE_TMP/blah_shadow.q" ]
  [ -d "$THE_TMP/blah_shadow.q.3.old" ]
 
  # Test where destination foo_shadow.q.3.old already exists
  mkdir -p "$THE_TMP/foo_shadow.q" 1>&2
  run moveOldClusterFolders "3" "$THE_TMP"
  [ "$status" -eq 0 ]
  ls -la "$THE_TMP" 1>&2
  [ ! -d "$THE_TMP/foo_shadow.q" ]
  [ -d "$THE_TMP/foo_shadow.q.3.old" ]
  [ ! -d "$THE_TMP/blah_shadow.q" ]
  [ -d "$THE_TMP/blah_shadow.q.3.old" ]
}

#
# checkForKillFile() tests
#
@test "checkForKillFile() tests" {
    
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test where there is no kill file
  run checkForKillFile "$THE_TMP"
  [ "$status" -eq 0 ]

  export CHUMBINARY="echo"
  export CHUMMEDLIST="foo.q"
  touch "$THE_TMP/KILL.JOB.REQUEST"
  # test where there is a kill file
  run checkForKillFile "$THE_TMP"

  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "  $THE_TMP/KILL.JOB.REQUEST detected.  Chumming kill file to remote cluster(s) and exiting..." ]
  [ "${lines[1]}" == "  Running echo --path $THE_TMP/KILL.JOB.REQUEST --cluster foo.q > $THE_TMP/killed.chum.out" ]

  [ -s "$THE_TMP/killed.chum.out" ]
  aLine=`head -n 1 "$THE_TMP/killed.chum.out"`
  [ "$aLine" == "--path /tmp/helperfuncs/KILL.JOB.REQUEST --cluster foo.q" ]

  # test where there is a kill file and we request to just have code return instead of exiting
  touch "$THE_TMP/KILL.JOB.REQUEST"
  run checkForKillFile "$THE_TMP" "blah.q" "dontexit"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "  $THE_TMP/KILL.JOB.REQUEST detected.  Chumming kill file to remote cluster(s)" ]
  [ "${lines[1]}" == "  Running echo --path $THE_TMP/KILL.JOB.REQUEST --cluster blah.q > $THE_TMP/killed.chum.out" ]
  
}

# 
# chumData() tests
#
@test "chumData() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2

  export CHUMBINARY="/bin/false"
  # Test where chum fails
  run chumData "clist" "$THE_TMP" "$THE_TMP/chum.out" "blah"
  [ "$status" -eq 1 ] 
  
  # Test where chum succeeds check CHUMMEDLIST variable
  echo "0,chummed.clusters=hello,," > "$THE_TMP/panfish/panfishchum.tasks"
  export CHUMBINARY="$THE_TMP/panfish/panfishchum"
  chumData "clist.q" "$THE_TMP" "$THE_TMP/chum.out" "--exclude ha"
  [ "$?" -eq 0 ]
  [ "$CHUMMEDLIST" == "hello" ]

  # Test where chum succeeds verify args passed in
  export CHUMBINARY="echo"
  run chumData "clist.q" "$THE_TMP" "$THE_TMP/chum.out" "--exclude ha"
  [ "$?" -eq 0 ]
  aLine=`head -n 1 "$THE_TMP/chum.out"`
  [ "$aLine" == "--listchummed --path $THE_TMP --cluster clist.q --exclude ha" ]


}

#
# landData() tests
#
@test "landData() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  export LANDBINARY="/bin/false"

  # Test where land fails
  run landData "clist" "$THE_TMP" "blah"
  [ "$status" -eq 1 ]
  
  export LANDBINARY="echo"
  # Test where land succeeds
  run landData "clist" "$THE_TMP" "--exclude yo"
  [ "$status" -eq 0 ] 
  [ "${lines[0]}" == "--path $THE_TMP --cluster clist --exclude yo" ]

}

#
# getStatusOfJobsInCastOutFile() tests
# 
@test "getStatusOfJobsInCastOutFile() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2

  # Test with no cast.out file
  run getStatusOfJobsInCastOutFile "$THE_TMP" "cast.out"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    No $THE_TMP/cast.out file found" ]
 
  # Test where panfishstat binary outputs error 
  export PANFISHSTATBINARY="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run getStatusOfJobsInCastOutFile "$THE_TMP" "cast.out"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Error calling /bin/false --statusofjobs $THE_TMP/cast.out" ]

  # Test with successful panfishstat run
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  echo "0,status=hello,," > "$THE_TMP/panfish/panfishstat.tasks"  
  getStatusOfJobsInCastOutFile "$THE_TMP" "cast.out"
  [ "$?" -eq 0 ]
  [ "$JOBSTATUS" == "hello" ]
}

#
# waitForJobs() tests
#
@test "waitForJobs() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  echo "hello" > "$THE_TMP/cast.out"

  # Test where job finishes 1st time
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  run waitForJobs "3" "$THE_TMP" "cast.out" "foo.q" "--exclude foo" "0" "0" "0"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 3 Start Time: "* ]]
  [[ "${lines[1]}" == " WaitForJobs in cast.out Iteration 3 End Time: "* ]]
  

  # Test where getStatus errors 1st time and works second going to run state and finishes 3rd time
  echo "1,status=uh,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
   
  run waitForJobs "2" "$THE_TMP" "cast.out" "foo.q" "--exclude foo" "0" "0" "0"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 2 Start Time: "* ]]
  [ "${lines[1]}" == "WARNING:    Error calling $PANFISHSTATBINARY --statusofjobs $THE_TMP/cast.out" ]
  [ "${lines[2]}" == "  Iteration 2 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[3]}" == "  Iteration 2 job status is running.  Sleeping 0 seconds" ]
  [[ "${lines[4]}" == " WaitForJobs in cast.out Iteration 2 End Time: "* ]]

  
  # Test where a kill file is found
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  export CHUMBINARY="/bin/true"
  echo "die" > "$THE_TMP/KILL.JOB.REQUEST"
  run waitForJobs "3" "$THE_TMP" "cast.out" "foo.q" "--exclude foo" "0" "0" "0"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 3 Start Time: "* ]]
  [ "${lines[1]}" == "  $THE_TMP/KILL.JOB.REQUEST detected.  Chumming kill file to remote cluster(s) and exiting..." ]
  echo "${lines[2]}" 1>&2
  [ "${lines[2]}" == "  Running /bin/true --path $THE_TMP/KILL.JOB.REQUEST --cluster foo.q > $THE_TMP/killed.chum.out" ]

  # Test where DOWNLOAD.DATA.REQUEST file shows up and job then finishes after
  echo "yo" > "$THE_TMP/chm.cast.out"
  export LANDBINARY="/bin/echo"
  echo "0,status=running,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,,touch $THE_TMP/DOWNLOAD.DATA.REQUEST" >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"  
  /bin/rm -f $THE_TMP/KILL.JOB.REQUEST

  run waitForJobs "3" "$THE_TMP" "chm.cast.out" "foo.q" "--exclude hi.* --exclude bye.*" "0" "0" "0"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " WaitForJobs in chm.cast.out Iteration 3 Start Time: "* ]]
  [ "${lines[1]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [ "${lines[2]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [ "${lines[3]}" == "  DOWNLOAD.DATA.REQUEST file found.  Performing download" ]
  [ "${lines[4]}" == "--path $THE_TMP --cluster foo.q --exclude hi.* --exclude bye.*" ]
  [ "${lines[5]}" == "  Removing DOWNLOAD.DATA.REQUEST file" ]
  [ "${lines[6]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [[ "${lines[7]}" == " WaitForJobs in chm.cast.out Iteration 3 End Time: "* ]] 
  [ ! -e "$THE_TMP/DOWNLOAD.DATA.REQUEST" ]

}

#
# moveCastOutFile() tests
#
@test "moveCastOutFile() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where there isn't a cast.out file
  run moveCastOutFile "1" "$THE_TMP" "hi"
  [ "$status" -eq 0 ]

  # Test where there is a cast.out file
  echo "hi" > "$THE_TMP/cast.out"
  run moveCastOutFile "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/cast.out.1.out" ]
  [ ! -e "$THE_TMP/cast.out" ]

  # Test where there is a cast.1.out file
  echo "bye" > "$THE_TMP/cast.out"
  run moveCastOutFile "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/cast.out.1.out" ]
  [ ! -e "$THE_TMP/cast.out" ]
  aLine=`head -n 1 "$THE_TMP/cast.out.1.out"`
  [ "$aLine" == "bye" ]

  # Test where mv command fails
  export MV_CMD="/bin/false"
  echo "yo" > "$THE_TMP/cast.out" "cast.out"
  run moveCastOutFile "1" "$THE_TMP"
  [ "$status" -eq 1 ]
}

#
# moveOldDataForNewIteration() tests
#
@test "moveOldDataForNewIteration() tests" {
    
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
 
  # test where CHUMMEDLIST is empty and no cast.out file
  unset CHUMMEDLIST
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
 
  # test where there is an error moving the old cluster folders
  export MV_CMD="/bin/false"
  export CHUMMEDLIST="foo.q"
  mkdir -p "$THE_TMP/foo.q" 1>&2
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cluster folders for previous iteration 0" ]

  # test where there is an error moving cluster folders and moving cast.out file
  export MV_CMD="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 3 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cluster folders for previous iteration 0" ]
  [ "${lines[1]}" == "WARNING:    Unable to move cast.out file for previous iteration 0" ]

  unset CHUMMEDLIST
  export MV_CMD="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cast.out file for previous iteration 0" ]

  

}

#
# castJob() tests
#
@test "castJob() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where cast fails
  export CASTBINARY="/bin/false"
  /bin/rm -f "$THE_TMP/cast.out" 1>&2
  thecurdir=`pwd`
  run castJob "$RUN_CHM_SH" "$THE_TMP" "1" "2" "foo" "cast.out" "hi.q" "--batchy bye" "runchmout"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Error calling /bin/false -t 1-2 -q hi.q -N foo --batchy bye --writeoutputlocal -o $THE_TMP/runchmout/stdout/\$TASK_ID.stdout -e $THE_TMP/runchmout/stderr/\$TASK_ID.stderr $THE_TMP/runCHM.sh > $THE_TMP/cast.out" ]
  newcurdir=`pwd`
  [ "$thecurdir" == "$newcurdir" ]


  # Test where cast succeeds
  export CASTBINARY="/bin/echo"
  /bin/rm -f "$THE_TMP/cast.out" 1>&2
  thecurdir=`pwd`
  run castJob "$RUN_CHM_SH" "$THE_TMP" "1" "2" "foo" "cast.out" "hi.q" "--batchy bye" "runchmout"
  [ "$status" -eq 0 ]
  newcurdir=`pwd`
  [ "$thecurdir" == "$newcurdir" ]
  [ -s "$THE_TMP/cast.out" ]
  cat "$THE_TMP/cast.out" 1>&2
  aLine=`cat "$THE_TMP/cast.out"`
  [ "$aLine" == "-t 1-2 -q hi.q -N foo --batchy bye --writeoutputlocal -o /tmp/helperfuncs/runchmout/stdout/\$TASK_ID.stdout -e /tmp/helperfuncs/runchmout/stderr/\$TASK_ID.stderr /tmp/helperfuncs/runCHM.sh" ]

  # Test where jobStart is a file
  export CASTBINARY="/bin/echo"
  /bin/rm -f "$THE_TMP/cast.out" 1>&2
  echo "hi" > "$THE_TMP/tasks"
  thecurdir=`pwd`
  run castJob "$RUN_CHM_SH" "$THE_TMP" "$THE_TMP/tasks" "" "foo" "cast.out" "hi.q" "--batchy bye" "runchmout"
  [ "$status" -eq 0 ]
  newcurdir=`pwd`
  [ "$thecurdir" == "$newcurdir" ]
  [ -s "$THE_TMP/cast.out" ] 
  cat "$THE_TMP/cast.out" 1>&2
  aLine=`cat "$THE_TMP/cast.out"`
  [ "$aLine" == "--taskfile $THE_TMP/tasks -q hi.q -N foo --batchy bye --writeoutputlocal -o /tmp/helperfuncs/runchmout/stdout/\$TASK_ID.stdout -e /tmp/helperfuncs/runchmout/stderr/\$TASK_ID.stderr /tmp/helperfuncs/runCHM.sh" ]

}


# 
# waitForDownloadAndVerifyJobs() tests
#
@test "waitForDownloadAndVerifyJobs tests" {
 
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  
  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export LANDBINARY="$THE_TMP/panfish/panfishland"
  
  # no cast.out file and land fails and verify fails 

  echo "1,error,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "1,error,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "1,error,," >> "$THE_TMP/panfish/panfishland.tasks"
 
  run waitForDownloadAndVerifyJobs "$RUN_CHM_SH" "1" "$THE_TMP" "3" "cast.out" "foo.q" "--exclude" "0" "failed" "failed.jobs.tmp" "failed.jobs"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 1 Start Time: "* ]]
  [ "${lines[3]}" == "  While checking if any jobs exist, it appears no cast.out exists." ]
  [ "${lines[5]}" == "WARNING:    Unable to download data.  Will continue on with checking results just in case." ] 
 

  # all good but verify fails
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  echo "hi" > "$THE_TMP/cast.out"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,error,," > "$THE_TMP/panfish/panfishland.tasks"
 
  run waitForDownloadAndVerifyJobs "$RUN_CHM_SH" "2" "$THE_TMP" "1" "cast.out" "foo.q" "--exclude" "0" "failed" "failed.jobs.tmp" "failed.jobs"
 
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 2 Start Time: "* ]]
  [[ "${lines[1]}" == *" Exit Code: 0" ]]
  [ "${lines[4]}" == "  Renaming $THE_TMP/failed.jobs to $THE_TMP/failed.1.jobs" ]
  [ "${lines[5]}" == "  Found 1 failed job(s)" ]
  [ "${lines[6]}" == "  Creating failed.jobs file" ]
  [ -s "$THE_TMP/failed.jobs" ]

  # all good
  echo "function checkSingleTask {
          return 0
  }" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"

  echo "hi" > "$THE_TMP/cast.out"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,error,," > "$THE_TMP/panfish/panfishland.tasks"
  run waitForDownloadAndVerifyJobs "$RUN_CHM_SH" "3" "$THE_TMP" "1" "cast.out" "foo.q" "--exclude" "0" "failed" "failed.jobs.tmp" "failed.jobs"
  echo "$output" 1>&2
  echo "status $status" 1>&2
  [ "$status" -eq 0 ]

}

#
# runJobs() tests
# 
@test "runJobs() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2

  export CHUMBINARY="$THE_TMP/panfish/panfishchum"
  export CASTBINARY="$THE_TMP/panfish/panfishcast"
  export LANDBINARY="$THE_TMP/panfish/panfishland"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"

  # Test jobs already completed successfully
  
  echo "function checkSingleTask {
          return 0
  }" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"

  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
 
  run runJobs "$RUN_CHM_SH" "1" "$THE_TMP" "3" "myjob" "cast.out" "foo.q" "--exclude" "failed" "failed.jobs.tmp" "failed.jobs" "1" "0" "my.iteration" "0" "--batchy" "outdir"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " runCHM.sh Start Time: "* ]]
  [ "${lines[1]}" == "  Checking for already running jobs and previously completed jobs" ]
  [[ "${lines[2]}" == " WaitForJobs in cast.out Iteration 1 Start Time: "* ]]
  [ "${lines[3]}" == "WARNING:    No $THE_TMP/cast.out file found" ]
  [[ "${lines[4]}" == " WaitForJobs in cast.out Iteration 1 End Time: "* ]]
  [ "${lines[5]}" == "  While checking if any jobs exist, it appears no cast.out exists." ]
  [ "${lines[6]}" == "landcall" ]
  [[ "${lines[7]}" == *" Exit Code: 0" ]]

  # Test failure to chum data
  echo "function checkSingleTask {
          return 1
  }" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"

  /bin/rm -f "$THE_TMP/${OUT_DIR_NAME}/1.png/1.png" 1>&2
  echo "0,download,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "1,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  run runJobs "$RUN_CHM_SH" "1" "$THE_TMP" "3" "myjob" "cast.out" "foo.q" "--exclude" "failed" "failed.jobs.tmp" "failed.jobs" "1" "0" "my.iteration" "0" "--batchy" "outdir"
  [ "$status" -eq 10 ]
  [[ "${lines[0]}" == " runCHM.sh Start Time: "* ]]
  [ "${lines[9]}" == "  Uploading data for runCHM.sh job(s)" ]
  [ "${lines[11]}" == "WARNING:    Unable to upload data for runCHM.sh job(s)" ]
  [ -s "$THE_TMP/failed.jobs" ]
  aLine=`head -n 1 "$THE_TMP/failed.jobs"`
  [ "$aLine" == "1" ]
   
  # Test failure to submit jobs
   
  echo "function chumJobData {
     echo \$*
     return 0
  }
  function checkSingleTask {
          return 1
  }" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"

  echo "0,download,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,chumerror,," > "$THE_TMP/panfish/panfishcast.tasks"

  run runJobs "$RUN_CHM_SH" "1" "$THE_TMP" "3" "myjob" "cast.out" "foo.q" "--exclude" "failed" "failed.jobs.tmp" "failed.jobs" "1" "0" "my.iteration" "0" "--batchy" "outdir"
  [ "$status" -eq 11 ]
  [ "${lines[11]}" == "runCHM.sh 1 $THE_TMP" ]
  [ "${lines[13]}" == "WARNING:    Unable to submit jobs" ]

  # Test success 1st time
  echo "function chumJobData {
     return 0
  }
  function checkSingleTask {
     \$THE_TMP/panfish/panfish
     return \$?
  }" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"
  echo "1,,," > "$THE_TMP/panfish/panfish.tasks"
  echo "1,,," >> "$THE_TMP/panfish/panfish.tasks"
  echo "1,,," >> "$THE_TMP/panfish/panfish.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfish.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfish.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfish.tasks"

  echo "0,cast,," > "$THE_TMP/panfish/panfishcast.tasks" 
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"

  run runJobs "$RUN_CHM_SH" "2" "$THE_TMP" "3" "myjob" "cast.out" "foo.q" "--exclude" "failed" "failed.jobs.tmp" "failed.jobs" "1" "0" "my.iteration" "0" "--batchy" "outdir"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " runCHM.sh Start Time: "* ]]
  [ -s "$THE_TMP/cast.out" ]
  [[ "${lines[14]}" == *" Exit Code: 0" ]]
  
  # Test 1st iteration fail then success
  echo "function chumJobData {
     return 0
  }
  function checkSingleTask {
     \$THE_TMP/panfish/panfish
     return \$?
  }" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"
  echo "1,,," > "$THE_TMP/panfish/panfish.tasks"
  echo "1,,," >> "$THE_TMP/panfish/panfish.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfish.tasks"
  echo "0,cast,," > "$THE_TMP/panfish/panfishcast.tasks"
  echo "0,cast,," >> "$THE_TMP/panfish/panfishcast.tasks"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"

  run runJobs "$RUN_CHM_SH" "2" "$THE_TMP" "1" "myjob" "cast.out" "foo.q" "--exclude" "failed" "failed.jobs.tmp" "failed.jobs" "1" "0" "my.iteration" "0" "--batchy" "outdir"
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " runCHM.sh Start Time: "* ]]
  echo "${lines[16]}" 1>&2 
  [ "${lines[16]}" == "  Iteration 3.  Jobs failed in previous iteration. sleeping 0 seconds before trying again" ]
  [[ "${lines[22]}" == *" Exit Code: 0" ]]

  
  # Test where failures exceed max retries
  echo "function chumJobData {
     return 0
  }
  function checkSingleTask {
     return 1
  }" > "$THE_TMP/myscript.sh"
  . "$THE_TMP/myscript.sh"

  echo "0,cast,," > "$THE_TMP/panfish/panfishcast.tasks"
  echo "0,cast,," >> "$THE_TMP/panfish/panfishcast.tasks"
  echo "0,cast,," >> "$THE_TMP/panfish/panfishcast.tasks"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"


  run runJobs "$RUN_CHM_SH" "2" "$THE_TMP" "1" "myjob" "cast.out" "foo.q" "--exclude" "failed" "failed.jobs.tmp" "failed.jobs" "1" "0" "my.iteration" "0" "--batchy" "outdir"
  [ "$status" -eq 1 ]
}

#
# getParameterForTaskFromConfig() tests
#
@test "getParameterForTaskFromConfig() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  
  # Test where initial egrep fails
  run getParameterForTaskFromConfig "1" "1" "$THE_TMP/doesnotexist"
  [ "$status" -eq 1 ] 

  # Test where line to parse is larger then lines found
  echo "1${CONFIG_DELIM}a" > "$THE_TMP/config"
  echo "1${CONFIG_DELIM}b" >> "$THE_TMP/config"
  echo "1${CONFIG_DELIM}c" >> "$THE_TMP/config"
  echo "2${CONFIG_DELIM}d" >> "$THE_TMP/config"
  run getParameterForTaskFromConfig "1" "4" "$THE_TMP/config"
  [ "$status" -eq 2 ]



  # Test valid extraction from 1st line
  getParameterForTaskFromConfig "1" "1" "$THE_TMP/config"
  [ "$?" -eq 0 ] 
  [ "$TASK_CONFIG_PARAM" == "a" ]
  

  # Test valid extraction from 2nd line
  getParameterForTaskFromConfig "1" "2" "$THE_TMP/config"
  [ "$?" -eq 0 ]
  [ "$TASK_CONFIG_PARAM" == "b" ]

  # Test valid extraction from 3rd line
  getParameterForTaskFromConfig "1" "1" "$THE_TMP/config"
  [ "$?" -eq 0 ]
  [ "$TASK_CONFIG_PARAM" == "a" ]


  # Test valid extraction from 4th line
  echo "2${CONFIG_DELIM}e" >> "$THE_TMP/config"
  echo "2${CONFIG_DELIM}f" >> "$THE_TMP/config"
  echo "2${CONFIG_DELIM} g -g ?" >> "$THE_TMP/config"
  getParameterForTaskFromConfig "2" "4" "$THE_TMP/config"
  [ "$?" -eq 0 ]
  [ "$TASK_CONFIG_PARAM" == " g -g ?" ]
}

#
# getNumberOfJobsFromConfig() tests
#
@test "getNumberOfJobsFromConfig() tests" {
 
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/rm -f "$THE_TMP/runCHM.sh.config" 1>&2
  # Test where there isnt a config
  run getNumberOfJobsFromConfig "$THE_TMP" "runCHM.sh.config"
  echo "$output and status $status" 1>&2
  [ "$status" -eq 1 ]

  # Test on empty file
  touch "$THE_TMP/runCHM.sh.config"
  run getNumberOfJobsFromConfig "$THE_TMP" "runCHM.sh.config"
  [ "$status" -eq 2 ]

  # Test on file without proper job #${CONFIG_DELIM} prefix on last line
  echo "yikes" > "$THE_TMP/runCHM.sh.config"
  getNumberOfJobsFromConfig "$THE_TMP" "runCHM.sh.config"
  [ "$?" -eq 0 ]
  [ "$NUMBER_JOBS" == "yikes" ]

  # Test on file with 1 job
  echo "1${CONFIG_DELIM}a" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}b" >> "$THE_TMP/runCHM.sh.config"
  getNumberOfJobsFromConfig "$THE_TMP" "runCHM.sh.config"
  [ "$?" -eq 0 ]
  [ "$NUMBER_JOBS" -eq 1 ]


  # Test on file with 4 jobs
  echo "4${CONFIG_DELIM}asdf" >> "$THE_TMP/runCHM.sh.config"
  getNumberOfJobsFromConfig "$THE_TMP" "runCHM.sh.config"
  [ "$?" -eq 0 ]
  [ "$NUMBER_JOBS" -eq 4 ]

}

#
# getNextIteration() tests
#
@test "getNextIteration() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test no iteration file
  run getNextIteration "$THE_TMP" "foo"
  [ "$status" -eq 1 ]
 
  # test with iteration file
  echo "3" > "$THE_TMP/foo"
  getNextIteration "$THE_TMP" "foo"
  [ "$?" -eq 0 ] 
  [ "$NEXT_ITERATION" -eq 4 ]

  # test with empty file
  echo "" > "$THE_TMP/foo"
  run getNextIteration "$THE_TMP" "foo"
  [ "$status" -eq 2 ]

}

#
# isExitCodeInStdOutFileZero() tests
#
@test "isExitCodeInStdOutFileZero() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test no stdout file
  run isExitCodeInStdOutFileZero "$THE_TMP/foo"
  [ "$status" -eq 12 ]

  echo "blah" > "$THE_TMP/foo"
  
  # test no Exit Code in file
  run isExitCodeInStdOutFileZero "$THE_TMP/foo"
  [ "$status" -eq 13 ]

  # test 0 exit code
  echo "blah" > "$THE_TMP/foo"
  echo "(task 1079206.1) runCHMTrain.sh End Time: 1411046427 Duration: 52523 Exit Code: 0" >> "$THE_TMP/foo"
  run isExitCodeInStdOutFileZero "$THE_TMP/foo"
  [ "$status" -eq 0 ]

  # test non zero exit code ie Exit Code: 255
  echo "blah" > "$THE_TMP/foo"
  echo "(task 1079206.1) runCHMTrain.sh End Time: 1411046427 Duration: 52523 Exit Code: 255" >> "$THE_TMP/foo"
  run isExitCodeInStdOutFileZero "$THE_TMP/foo"
  [ "$status" -eq 14 ]

  # test empty exit code ie Exit Code: 
  echo "blah" > "$THE_TMP/foo"
  echo "(task 1079206.1) runCHMTrain.sh End Time: 1411046427 Duration: 52523 Exit Code: " >> "$THE_TMP/foo"
  run isExitCodeInStdOutFileZero "$THE_TMP/foo"
  [ "$status" -eq 14 ]
}
