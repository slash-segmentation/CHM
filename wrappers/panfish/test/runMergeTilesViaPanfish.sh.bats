#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runMergeTilesViaPanfish" 1>&2
  /bin/mkdir -p "$THE_TMP/panfish" 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/runMergeTilesViaPanfish.sh "$THE_TMP/." 1>&2
    
  export RUN_MERGE_TILES="$THE_TMP/runMergeTilesViaPanfish.sh"
  chmod a+x $RUN_MERGE_TILES
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

  . $RUN_MERGE_TILES source

  # Test where jobdir does not exist
  run checkSingleTask "$RUN_MERGE_TILES_SH" "$THE_TMP/doesnotexist" "1"
  [ "$status" -eq 1 ]

  mkdir -p "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME"
  # Test where stdout file does not exist
  run checkSingleTask "$RUN_MERGE_TILES_SH" "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  # test where stdout file size is 0
  touch "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  run checkSingleTask "$RUN_MERGE_TILES_SH" "$THE_TMP" "1"
  [ "$status" -eq 2 ]


  # test where there is a problem parsing config
  echo "hi" > "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  run checkSingleTask "$RUN_MERGE_TILES_SH" "$THE_TMP" "1"
  [ "$status" -eq 3 ]


  # test where there is no output file
  echo "1${CONFIG_DELIM}yo" > "$THE_TMP/$RUN_MERGE_TILES_CONFIG"
  echo "1${CONFIG_DELIM}foo.png" >> "$THE_TMP/$RUN_MERGE_TILES_CONFIG"
  run checkSingleTask "$RUN_MERGE_TILES_SH" "$THE_TMP" "1"
  echo "$status is" 1>&2
  [ "$status" -eq 4 ]

  # test success
  echo "hi" > "$THE_TMP/foo.png"
  run checkSingleTask "$RUN_MERGE_TILES_SH" "$THE_TMP" "1"
  [ "$status" -eq 0 ]
}


# chumJobData
@test "chumJobData() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  . $RUN_MERGE_TILES source

  # Test where chumData fails
 echo "MERGE_TILES_CHUMMEDLIST=chum.q
   CHUM_MERGE_TILES_OPTS=opts
   function chumData {
   echo \$*
   return 1
}" > "$THE_TMP/chum.sh"
  . "$THE_TMP/chum.sh"

  run chumJobData "runMergeTiles.sh" "1" "$THE_TMP"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "chum.q $THE_TMP $MERGE_TILES_CHUM_OUT opts" ]

  # Test where chumData succeeds
   echo "MERGE_TILES_CHUMMEDLIST=chum.q
   CHUM_MERGE_TILES_OPTS=opts
   function chumData {
   echo \$*
   return 0
}" > "$THE_TMP/chum.sh"
  . "$THE_TMP/chum.sh"
    run chumJobData "runMergeTiles.sh" "3" "hi"
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "chum.q hi $MERGE_TILES_CHUM_OUT opts" ]
}

# -h flag
@test "-h flag" {
  run $RUN_MERGE_TILES -h
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "Run Merge Tiles via Panfish." ]

}

# No .helper funcs
@test "no .helperfuncs.sh" {
  /bin/rm -f $HELPERFUNCS
  run $RUN_MERGE_TILES
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "No $THE_TMP/.helperfuncs.sh found" ] 
}


# No panfishCHM.properties file
@test "no panfishCHM.properties" {
  run $RUN_MERGE_TILES
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    There was a problem parsing the panfishCHM.properties file" ]
}


# No runMergeTiles.sh.config config file
@test "no runMergeTiles.sh.config" {
  
  . $HELPERFUNCS
  echo "hi" > "$THE_TMP/panfishCHM.properties"

  run $RUN_MERGE_TILES 
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    Error obtaining number of jobs from $RUN_MERGE_TILES_CONFIG file" ]
}

# -S flag unable to get status
@test "-S unable to get status" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "MERGE_TILES_CAST_OUT_FILE=hi
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    return 0
  }
  function getStatusOfJobsInCastOutFile {
    echo \$*
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -S
  [ "$status" -eq 1 ]
  echo "${lines[1]}" 1>&2
  [ "${lines[1]}" == "Getting status of running/pending jobs..." ]
  [ "${lines[2]}" == "$THE_TMP hi" ]
  [ "${lines[3]}" == "Unable to get status of jobs" ]
  [ "${lines[4]}" == "Merge Tiles 0 1" ] 
}

# -S flag jobs are not done
@test "-S flag jobs are not done" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "MERGE_TILES_CAST_OUT_FILE=hi
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    return 0
  }
  function getStatusOfJobsInCastOutFile {
    echo \$*
    JOBSTATUS=running
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -S
  [ "$status" -eq 2 ]
  echo "${lines[3]}" 1>&2
  [ "${lines[1]}" == "Getting status of running/pending jobs..." ]
  [ "${lines[2]}" == "$THE_TMP hi" ]
  [ "${lines[3]}" == "Job status returned running Job(s) still running." ]
  [ "${lines[4]}" == "Merge Tiles 0 2" ]  
}

# -S flag jobs are done
@test "-S flag jobs are done" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "MERGE_TILES_CAST_OUT_FILE=hi
        DONE_JOB_STATUS=done
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    return 0
  }
  function getStatusOfJobsInCastOutFile {
    echo \$*
    JOBSTATUS=done
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -S
  [ "$status" -eq 0 ]
  echo "${lines[3]}" 1>&2
  [ "${lines[1]}" == "Getting status of running/pending jobs..." ]
  [ "${lines[2]}" == "$THE_TMP hi" ]
  [ "${lines[3]}" == "No running/pending jobs found." ]
  [ "${lines[4]}" == "Merge Tiles 0 0" ]
}

# -C flag jobs failed
@test "-C flag jobs failed" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "MERGE_TILES_CAST_OUT_FILE=hi
        DONE_JOB_STATUS=done
        MERGE_TILES_FAILED_PREFIX=prefix
        MERGE_TILES_TMP_FILE=tmp
        MERGE_TILES_FAILED_FILE=fail
        RUN_MERGE_TILES_SH=runMergeTiles.sh
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
  function verifyResults {
    echo \$*
    NUM_FAILED_JOBS=3
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -C
  [ "$status" -eq 1 ]
  echo "${lines[2]}" 1>&2
  [ "${lines[1]}" == "Checking results..." ]

  [ "${lines[2]}" == "runMergeTiles.sh 1 $THE_TMP 1 6 no prefix tmp fail" ]
  [ "${lines[3]}" == "3 out of 6 job(s) failed." ]
  [ "${lines[4]}" == "Merge Tiles 0 1" ]
}

# -C flag success
@test "-C flag success" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "MERGE_TILES_CAST_OUT_FILE=hi
        DONE_JOB_STATUS=done
        MERGE_TILES_FAILED_PREFIX=prefix
        MERGE_TILES_TMP_FILE=tmp
        MERGE_TILES_FAILED_FILE=fail
        RUN_MERGE_TILES_SH=runMergeTiles.sh
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
  function verifyResults {
    echo \$*
    NUM_FAILED_JOBS=0
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -C
  [ "$status" -eq 0 ]
  echo "${lines[2]}" 1>&2
  [ "${lines[1]}" == "Checking results..." ]

  [ "${lines[2]}" == "runMergeTiles.sh 1 $THE_TMP 1 6 no prefix tmp fail" ]
  [ "${lines[3]}" == "All 6 job(s) completed successfully." ]
  [ "${lines[4]}" == "Merge Tiles 0 0" ]
}


# -D fail
@test "-D fail" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "MERGE_TILES_CHUMMEDLIST=chum.q
        LAND_MERGE_TILES_OPTS=opts
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
  function landData {
    echo \$*
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -D
  [ "$status" -eq 1 ]
  echo "${lines[2]}" 1>&2
  [ "${lines[1]}" == "Downloading/Landing data..." ]
  [ "${lines[2]}" == "chum.q $THE_TMP opts 0 0" ]
  [ "${lines[3]}" == "Unable to retrieve data" ]
  [ "${lines[4]}" == "Merge Tiles 0 1" ]
}


# -D mode success
@test "-D success" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "MERGE_TILES_CHUMMEDLIST=chum.q
        LAND_MERGE_TILES_OPTS=opts
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
  function landData {
    echo \$*
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -D
  [ "$status" -eq 0 ]
  echo "${lines[2]}" 1>&2
  [ "${lines[1]}" == "Downloading/Landing data..." ]
  [ "${lines[2]}" == "chum.q $THE_TMP opts 0 0" ]
  [ "${lines[3]}" == "Download successful." ]
  [ "${lines[4]}" == "Merge Tiles 0 0" ]
}


# -U mode fail
@test "-U fail" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "RUN_MERGE_TILES_SH=runMergeTiles.sh
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
  function chumJobData {
    echo \$*
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -U
  [ "$status" -eq 1 ]
  echo "${lines[2]}" 1>&2
  [ "${lines[1]}" == "Uploading/Chumming data..." ]
  [ "${lines[2]}" == "runMergeTiles.sh 1 $THE_TMP" ]
  [ "${lines[3]}" == "Unable to upload data" ]
  [ "${lines[4]}" == "Merge Tiles 0 1" ]
}


# -U mode success
@test "-U success" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "RUN_MERGE_TILES_SH=runMergeTiles.sh
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
  function chumJobData {
    echo \$*
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES -U
  [ "$status" -eq 0 ]
  echo "${lines[2]}" 1>&2
  [ "${lines[1]}" == "Uploading/Chumming data..." ]
  [ "${lines[2]}" == "runMergeTiles.sh 1 $THE_TMP" ]
  [ "${lines[3]}" == "Upload successful." ]
  [ "${lines[4]}" == "Merge Tiles 0 0" ]
}

# Full run where runJobs fails
@test "full run where runJobs fails" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "RUN_MERGE_TILES_SH=runMergeTiles.sh
        MERGE_TILES_ITERATION_FILE=foo
        MERGE_TILES_CAST_OUT_FILE=cast
        MERGE_TILES_CHUMMEDLIST=chum
        LAND_MERGE_TILES_OPTS=opts
        MERGE_TILES_FAILED_PREFIX=prefix
        MERGE_TILES_TMP_FILE=tmp
        MERGE_TILES_FAILED_FILE=fail
        MAX_RETRIES=mretry
        WAIT_SLEEP_TIME=sleep
        RETRY_SLEEP=retry
        MERGE_TILES_BATCH_AND_WALLTIME_ARGS=batch
        MERGE_TILES_OUT_DIR_NAME=out
       
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

  run $RUN_MERGE_TILES -n hello
  echo "$output" 1>&2
  [ "$status" -eq 23 ]
  [ "${lines[2]}" == "foo file found setting iteration to 3" ]
  [ "${lines[3]}" == "runMergeTiles.sh 3 $THE_TMP 6 hello cast chum opts prefix tmp fail mretry sleep foo retry batch out" ]
  [ "${lines[4]}" == "Error running Merge Tiles" ]
  [ "${lines[5]}" == "Merge Tiles 0 23" ]
}


# Full run where runJobs succeeds
@test "full run where runJobs succeeds" {

    /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "RUN_MERGE_TILES_SH=runMergeTiles.sh
        MERGE_TILES_ITERATION_FILE=foo
        MERGE_TILES_CAST_OUT_FILE=cast
        MERGE_TILES_CHUMMEDLIST=chum
        LAND_MERGE_TILES_OPTS=opts
        MERGE_TILES_FAILED_PREFIX=prefix
        MERGE_TILES_TMP_FILE=tmp
        MERGE_TILES_FAILED_FILE=fail
        MAX_RETRIES=mretry
        WAIT_SLEEP_TIME=sleep
        RETRY_SLEEP=retry
        MERGE_TILES_BATCH_AND_WALLTIME_ARGS=batch
        MERGE_TILES_OUT_DIR_NAME=out
       
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

  run $RUN_MERGE_TILES 
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [ "${lines[2]}" == "runMergeTiles.sh 1 $THE_TMP 6 mergetiles_job cast chum opts prefix tmp fail mretry sleep foo retry batch out" ]
  [ "${lines[3]}" == "Merge Tiles successfully run." ]
  [ "${lines[4]}" == "Merge Tiles 0 0" ]
}

