#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runCHMTrainViaPanfish" 1>&2
  /bin/mkdir -p "$THE_TMP/panfish" 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/runCHMTrainViaPanfish.sh "$THE_TMP/." 1>&2
    
  export RUN_CHM_TRAIN_VIA_PANFISH="$THE_TMP/runCHMTrainViaPanfish.sh"
  chmod a+x $RUN_CHM_TRAIN_VIA_PANFISH
  unset SGE_TASK_ID
  /bin/cp -a "${BATS_TEST_DIRNAME}/bin/panfish" "${THE_TMP}/panfish/."
 
  export HELPERFUNCS="$THE_TMP/.helperfuncs.sh"
 
  export FAKEHELPERFUNCS="$THE_TMP/fakehelperfuncs"
  # Make fake base helperfuncs used by some tests
  echo "
  function logEcho {
    echo \$*
    return 0
  }
  function logMessage {
    echo \$*
    return 0
  }
  function logWarning {
    echo \$*
    return 0
  }
  function jobFailed {
    echo \$*
    exit 1
  }
  function logStartTime {
    echo \$*
    return 0
  }
  function logEndTime {
    echo \$*
    return 0
  }
" > "$FAKEHELPERFUNCS"

}

teardown(){
  /bin/rm -rf "$THE_TMP"
   echo "teardown" 1>&2
}

#
# checkSingleTask
#
@test "checkSingleTask() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  . $RUN_CHM_TRAIN_VIA_PANFISH source

  # Test where jobdir does not exist
  run checkSingleTask "$RUN_CHM_TRAIN_SH" "$THE_TMP/doesnotexist" "1"
  [ "$status" -eq 1 ]

  mkdir -p "$THE_TMP/$CHM_TRAIN_OUT_DIR_NAME/$STD_OUT_DIR_NAME"
  # Test where stdout file does not exist
  run checkSingleTask "$RUN_CHM_TRAIN_SH" "$THE_TMP" "1"
  [ "$status" -eq 1 ]

  # test where stdout file size is 0
  touch "$THE_TMP/$CHM_TRAIN_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  run checkSingleTask "$RUN_CHM_TRAIN_SH" "$THE_TMP" "1"
  [ "$status" -eq 1 ]


  # test where there is a problem parsing config
  echo "hi" > "$THE_TMP/$CHM_TRAIN_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  run checkSingleTask "$RUN_CHM_TRAIN_SH" "$THE_TMP" "1"
  [ "$status" -eq 2 ]


  # test where there is no output file
  echo "1${CONFIG_DELIM}yo" > "$THE_TMP/$RUN_CHM_TRAIN_CONFIG"
  echo "1${CONFIG_DELIM}foo.png" >> "$THE_TMP/$RUN_CHM_TRAIN_CONFIG"
  echo "1${CONFIG_DELIM}foo.png" >> "$THE_TMP/$RUN_CHM_TRAIN_CONFIG"
  echo "1${CONFIG_DELIM}$CHM_TRAIN_OUT_DIR_NAME/$CHM_TRAIN_TRAINEDMODEL_OUT_DIR_NAME" >> "$THE_TMP/$RUN_CHM_TRAIN_CONFIG"

  run checkSingleTask "$RUN_CHM_TRAIN_SH" "$THE_TMP" "1"
  echo "$status is" 1>&2
  [ "$status" -eq 3 ]

  # test success
  mkdir -p "$THE_TMP/$CHM_TRAIN_OUT_DIR_NAME/$CHM_TRAIN_TRAINEDMODEL_OUT_DIR_NAME"
  echo "hi" > "$THE_TMP/$CHM_TRAIN_OUT_DIR_NAME/$CHM_TRAIN_TRAINEDMODEL_OUT_DIR_NAME/param.mat"
  echo "blah" > "$THE_TMP/$CHM_TRAIN_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  echo "(task 1079206.1) runCHMTrain.sh End Time: 1411046427 Duration: 52523 Exit Code: 0" >> "$THE_TMP/$CHM_TRAIN_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  run checkSingleTask "$RUN_CHM_TRAIN_SH" "$THE_TMP" "1"
  [ "$status" -eq 0 ]
}


# chumJobData
@test "chumJobData() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  . $RUN_CHM_TRAIN_VIA_PANFISH source


  # Test where there is an error parsing config for 1st parameter
   echo "
   function getParameterForTaskFromConfig {
   echo \$*
   return 1
   }" > "$THE_TMP/chum.sh"
   . "$THE_TMP/chum.sh"
  run chumJobData "1" "1" $THE_TMP

  [ "$status" -eq 1 ] 
  [ "${lines[0]}" == "1 1 $THE_TMP/$RUN_CHM_TRAIN_CONFIG" ]


  # Test where there is an error parsing config for 2nd parameter
   echo "
   function getParameterForTaskFromConfig {
   if [ \$2 -eq 2 ] ; then
     echo \$*
     return 1
   fi
     return 0
   }" > "$THE_TMP/chum.sh"
   . "$THE_TMP/chum.sh"
  run chumJobData "1" "1" $THE_TMP

  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "1 2 $THE_TMP/$RUN_CHM_TRAIN_CONFIG" ]

  
  # Test where upload image directory chum fails
  echo "
   CHM_TRAIN_CHUMMEDLIST=chum.q
   CHUM_CHM_TRAIN_IMAGE_OPTS=imageopts
   function chumData {
     echo \$*
     return 1
   }
   function getParameterForTaskFromConfig {
     if [ \$2 -eq 1 ] ; then
       TASK_CONFIG_PARAM=images
     fi
     if [ \$2 -eq 2 ] ; then
       TASK_CONFIG_PARAM=labels
     fi
     return 0
   }" > "$THE_TMP/chum.sh"
   . "$THE_TMP/chum.sh"
  run chumJobData "1" "1" $THE_TMP

  [ "$status" -eq 3 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "chum.q images chm.train.chum.out imageopts" ]


  # Test where upload label directory chum fails
  echo "
   CHM_TRAIN_CHUMMEDLIST=chum.q
   CHUM_CHM_TRAIN_IMAGE_OPTS=imageopts
   CHUM_CHM_TRAIN_LABEL_OPTS=labelopts
   function chumData {
     if [ \$2 == images ] ; then
       return 0
     fi 
     echo \$*
     return 1
   }
   function getParameterForTaskFromConfig {
     if [ \$2 -eq 1 ] ; then
       TASK_CONFIG_PARAM=images
     fi
     if [ \$2 -eq 2 ] ; then
       TASK_CONFIG_PARAM=labels
     fi
     return 0
   }" > "$THE_TMP/chum.sh"
   . "$THE_TMP/chum.sh"
  run chumJobData "1" "1" $THE_TMP

  [ "$status" -eq 4 ]
  [ "${lines[0]}" == "chum.q labels chm.train.chum.out labelopts" ]

  # Test where upload job directory chum fails
    echo "
   CHM_TRAIN_CHUMMEDLIST=chum.q
   CHUM_CHM_TRAIN_IMAGE_OPTS=imageopts
   CHUM_CHM_TRAIN_LABEL_OPTS=labelopts
   CHUM_CHM_TRAIN_OPTS=jobopts
   function chumData {
     if [ \$2 == images ] ; then
       return 0
     fi
     if [ \$2 == labels ] ; then
       return 0
     fi
     echo \$*
     return 1
   }
   function getParameterForTaskFromConfig {
     if [ \$2 -eq 1 ] ; then
       TASK_CONFIG_PARAM=images
     fi
     if [ \$2 -eq 2 ] ; then
       TASK_CONFIG_PARAM=labels
     fi
     return 0
   }" > "$THE_TMP/chum.sh"
   . "$THE_TMP/chum.sh"
  run chumJobData "1" "1" ha

  [ "$status" -eq 5 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "chum.q ha chm.train.chum.out jobopts" ]


  # Test success
      echo "
   CHM_TRAIN_CHUMMEDLIST=chum.q
   function chumData {
     return 0
   }
   function getParameterForTaskFromConfig {
     if [ \$2 -eq 1 ] ; then
       TASK_CONFIG_PARAM=images
     fi
     if [ \$2 -eq 2 ] ; then
       TASK_CONFIG_PARAM=labels
     fi
     return 0
   }" > "$THE_TMP/chum.sh"
   . "$THE_TMP/chum.sh"
  run chumJobData "1" "1" ha

  [ "$status" -eq 0 ]


}

# -h flag
@test "-h flag" {
  run $RUN_CHM_TRAIN_VIA_PANFISH -h
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "Run CHM Train via Panfish." ]

}

# No .helper funcs
@test "no .helperfuncs.sh" {
  /bin/rm -f $HELPERFUNCS
  run $RUN_CHM_TRAIN_VIA_PANFISH
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "No $THE_TMP/.helperfuncs.sh found" ] 
}


# No panfishCHM.properties file
@test "no panfishCHM.properties" {
  run $RUN_CHM_TRAIN_VIA_PANFISH
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    There was a problem parsing the panfishCHM.properties file" ]
}


# No runMergeTiles.sh.config config file
@test "no runCHMTrain.sh.config" {
  
  . $HELPERFUNCS
  echo "hi" > "$THE_TMP/panfishCHM.properties"

  run $RUN_CHM_TRAIN_VIA_PANFISH 
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    Error obtaining number of jobs from $RUN_CHM_TRAIN_CONFIG file" ]
}

# Full run where runJobs fails
@test "full run where runJobs fails" {
 
    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.train.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "RUN_CHM_TRAIN_SH=runCHMTrain.sh
        CHM_TRAIN_ITERATION_FILE=foo
        CHM_TRAIN_CAST_OUT_FILE=cast
        CHM_TRAIN_CHUMMEDLIST=chum
        LAND_CHM_TRAIN_OPTS=opts
        CHM_TRAIN_FAILED_PREFIX=prefix
        CHM_TRAIN_TMP_FILE=tmp
        CHM_TRAIN_FAILED_FILE=fail
        MAX_RETRIES=mretry
        WAIT_SLEEP_TIME=sleep
        RETRY_SLEEP=retry
        CHM_TRAIN_BATCH_AND_WALLTIME_ARGS=batch
        CHM_TRAIN_OUT_DIR_NAME=out
       
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=6
    return 0
  }
  function getNextIteration {
    echo \$*
    NEXT_ITERATION=3
    return 0
  }
  function runJobs {
    echo \$*
    return 23
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_CHM_TRAIN_VIA_PANFISH -n hello
  echo "$output" 1>&2
  [ "$status" -eq 23 ]
  [ "${lines[2]}" == "foo file found setting iteration to 3" ]
  [ "${lines[3]}" == "runCHMTrain.sh 3 $THE_TMP 6 hello cast chum opts prefix tmp fail mretry sleep foo retry batch out" ]
  [ "${lines[4]}" == "Error running CHM Train" ]
  [ "${lines[5]}" == "CHM Train 0 23" ]
}


# Full run where runJobs succeeds
@test "full run where runJobs succeeds" {
  
    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.train.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "RUN_CHM_TRAIN_SH=runCHMTrain.sh
        CHM_TRAIN_ITERATION_FILE=foo
        CHM_TRAIN_CAST_OUT_FILE=cast
        CHM_TRAIN_CHUMMEDLIST=chum
        LAND_CHM_TRAIN_OPTS=opts
        CHM_TRAIN_FAILED_PREFIX=prefix
        CHM_TRAIN_TMP_FILE=tmp
        CHM_TRAIN_FAILED_FILE=fail
        MAX_RETRIES=mretry
        WAIT_SLEEP_TIME=sleep
        RETRY_SLEEP=retry
        CHM_TRAIN_BATCH_AND_WALLTIME_ARGS=batch
        CHM_TRAIN_OUT_DIR_NAME=out
       
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=6
    return 0
  }
  function getNextIteration {
    echo \$*
    return 1
  }
  function runJobs {
    echo \$*
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_CHM_TRAIN_VIA_PANFISH
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [ "${lines[2]}" == "runCHMTrain.sh 1 $THE_TMP 6 chmtrain_job cast chum opts prefix tmp fail mretry sleep foo retry batch out" ]
  [ "${lines[3]}" == "CHM Train successfully run." ]
  [ "${lines[4]}" == "CHM Train 0 0" ]
}

