#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runCHMJobViaPanfish" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2 
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/runCHMViaPanfish.sh "$THE_TMP/." 1>&2

  export RUNCHM_VIA_PANFISH="$THE_TMP/runCHMViaPanfish.sh"
  chmod a+x $RUNCHM_VIA_PANFISH
  unset SGE_TASK_ID
  /bin/cp -a  "${BATS_TEST_DIRNAME}/bin/panfish" "${THE_TMP}/." 1>&2
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
# getSingleCHMTestTaskStdErrFile() tests
#
@test "getSingleCHMTestTaskStdErrFile() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # source runCHMViaPanfish.sh to unit test functions
  . $RUNCHM_VIA_PANFISH source

  # Test where job path is not a directory
  run getSingleCHMTestTaskStdErrFile "$THE_TMP/doesnotexist" "1"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP/doesnotexist is not a directory" ]

  # Test where log file does not exist
  getSingleCHMTestTaskStdErrFile "$THE_TMP" "1"
  [ $? -eq 0 ]
  [ "$CHM_STD_ERR_FILE" == "$THE_TMP/${OUT_DIR_NAME}/stderr/1.stderr" ]

  # Test with valid log file
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stderr"
  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stderr/24.stderr"
  getSingleCHMTestTaskStdErrFile "$THE_TMP" "24"
  [ "$?" == 0 ]
  [ "$CHM_STD_ERR_FILE" == "$THE_TMP/${OUT_DIR_NAME}/stderr/24.stderr" ]

}

#
# checkSingleTask() tests
#
@test "checkSingleTask() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # source runCHMViaPanfish.sh to unit test functions
  . $RUNCHM_VIA_PANFISH source

  task="runCHMViaPanfish.sh"

  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stdout"
  # Test where stdout file does not exist
  run checkSingleTask $task "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  # Test where stdout file is zero size
  touch "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"
  run checkSingleTask $task "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  echo "1${CONFIG_DELIM}xx" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}yy" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}zz" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}${OUT_DIR_NAME}/uh.png" >> "$THE_TMP/runCHM.sh.config"

  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"
  # Test where no output image is found
  run checkSingleTask $task "$THE_TMP" "1"
  [ "$status" -eq 4 ]

  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/uh.png"
  # Test where we are all good
  run checkSingleTask $task "$THE_TMP" "1"
  [ "$status" -eq 0 ]
}

# 
# getSingleCHMTestTaskStdOutFile() tests
#
@test "getSingleCHMTestTaskStdOutFile() tests" {
 
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # source runCHMViaPanfish.sh to unit test functions
  . $RUNCHM_VIA_PANFISH source

  # Test where job path is not a directory
  run getSingleCHMTestTaskStdOutFile "$THE_TMP/doesnotexist" "1"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP/doesnotexist is not a directory" ]

  # Test where log file does not exist
  getSingleCHMTestTaskStdOutFile "$THE_TMP" "1"
  [ "$?" -eq 0 ]
  [ "$CHM_STD_OUT_FILE" == "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout" ]

  # Test with valid log file
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stdout"
  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/4.stdout"
  getSingleCHMTestTaskStdOutFile "$THE_TMP" "4"
  [ "$?" == 0 ]
  [ "$CHM_STD_OUT_FILE" == "$THE_TMP/${OUT_DIR_NAME}/stdout/4.stdout" ]

}



#
# chumModelImageAndJobData() tests
#
@test "chumJobData() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  . $RUNCHM_VIA_PANFISH source

  export CHUMMEDLIST="foo.q"

  export CHUMBINARY="$THE_TMP/panfish/panfishchum"

  # Test where there is an error parsing image dir from config
  run chumJobData "runCHM.sh" "1" "$THE_TMP"
  [ "$status" -eq 1 ]
  
  # Test where there is an error parsing model dir from config
  echo "1${CONFIG_DELIM}/blah/dir/1.png" > "$THE_TMP/$RUN_CHM_CONFIG"
  run chumJobData "runCHM.sh" "1" "$THE_TMP"
  echo "status is $status and $output" 1>&2
  [ "$status" -eq 2 ]


  # Test where upload of model fails
  echo "1,,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "1${CONFIG_DELIM}${THE_TMP}" >> "$THE_TMP/$RUN_CHM_CONFIG"
  run chumJobData "runCHM.sh" "1" "$THE_TMP"
  [ "$status" -eq 3 ]
  

  # Test where upload of image data fails
  echo "0,,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumJobData "runCHM.sh" "1" "$THE_TMP"
  echo "$output and $status" 1>&2
  [ "$status" -eq 4 ]

  # Test where upload of job directory fails
  echo "0,,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,,," >> "$THE_TMP/panfish/panfishchum.tasks"

  run chumJobData "runCHM.sh" "1" "$THE_TMP"
  [ "$status" -eq 5 ]


  # Test successful
  echo "0,,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumJobData "runCHM.sh" "1" "$THE_TMP"
  [ "$status" -eq 0 ]


}

#
# -h flag
#
@test "-h flag" {
  
  run $RUNCHM_VIA_PANFISH -h

  [ "$status" -eq 1 ] 
  echo "$output" 1>&2
  [ "${lines[0]}" == "Run CHM via Panfish." ]
}

#
# invalid flag
#
@test "invalid flag" {

  run $RUNCHM_VIA_PANFISH -asdf

  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "Invalid argument" ]
  [ "${lines[1]}" == "Run CHM via Panfish." ]

}

#
# no .helperfuncs.sh 
#
@test "no .helperfuncs.sh" {
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2
  run $RUNCHM_VIA_PANFISH
  [ "$status" -eq 2 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "No $THE_TMP/.helperfuncs.sh found" ]
}

#
# no panfishCHM.properties
#
@test "no panfishCHM.properties" {

  run $RUNCHM_VIA_PANFISH
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    There was a problem parsing the panfishCHM.properties file" ]

}

#
# Unable to get number of jobs 
#
@test "Unable to get number of jobs" {

  echo "panfish.bin.dir=$THE_TMP/panfish" > "$THE_TMP/panfishCHM.properties"

  run $RUNCHM_VIA_PANFISH
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[1]}" == "ERROR:    Error obtaining number of jobs from runCHM.sh.config file" ]

}

#
# Unable to get matlab dir
#
@test "Unable to get matlab dir" {

  echo "panfish.bin.dir=$THE_TMP/panfish" > "$THE_TMP/panfishCHM.properties"
  echo "1:::a" > "$THE_TMP/runCHM.sh.config"

  run $RUNCHM_VIA_PANFISH
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[1]}" == "ERROR:    Unable to get path to matlab directory: " ]
}

#
# Unable to get parameters from first job
#
@test "Unable to get parameters from first job" {
  
  echo "panfish.bin.dir=$THE_TMP/panfish/" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=$THE_TMP" >> "$THE_TMP/panfishCHM.properties"
  run $RUNCHM_VIA_PANFISH
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[1]}" == "ERROR:    Error obtaining number of jobs from runCHM.sh.config file" ]
}


#
# full run failure
#
@test "fullrun failure" {
  
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2
  
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2


  
  # make a fake helperfuncs file to test main script
  echo "
  function getFullPath {
  GETFULLPATHRET=\$THE_TMP
  return 0 
  }
  function parseProperties {
   RUN_CHM_SH=run
   OUTPUT_DIR=out
   CHM_TEST_ITERATION_FILE=iteration
   CHM_TEST_CAST_OUT_FILE=cast
   CHUMMEDLIST=chum.q
   LAND_JOB_OPTS=landopts
   FAILED=failed
   FAILED_JOBS_TMP_FILE=failedtmp
   FAILED_JOBS_FILE=failedjobs
   MAX_RETRIES=3
   WAIT_TIME_SLEEP=0
   RETRY_SLEEP=4
   BATCH_AND_WALLTIME_ARGS=batch
   OUT_DIR_NAME=ha
   MATLAB_DIR=\$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    return 0
  }
  function getNextIteration {
    NEXT_ITERATION=1
    return 0
  }
  function getCHMTestJobParametersForTaskFromConfig {
    INPUT_IMAGE=$THE_TMP/foo/blah
    MODEL_DIR=$THE_TMP/model
    return 0
  }
  function runJobs {
    echo \$*
    return 1
  }
  function chumData {
    return 0
  }
  function landData {
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH
  [ "$status" -eq 1 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  
  [ "${lines[2]}" == "run 1 out chm_job cast chum.q landopts failed failedtmp failedjobs 3 iteration 4 batch ha" ]
  [ "${lines[3]}" == "Full run 0 1" ]
  [ "${lines[4]}" == "Error running CHMTest" ]
}

#
# full run success
#
@test "full run success" {
  
 /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2
 # make a fake helperfuncs file to test main script
  echo "
  function getFullPath {
  GETFULLPATHRET=\$THE_TMP
  return 0
  }
  function parseProperties {
   RUN_CHM_SH=run
   OUTPUT_DIR=out
   CHM_TEST_ITERATION_FILE=iteration
   CHM_TEST_CAST_OUT_FILE=cast
   CHUMMEDLIST=chum.q
   LAND_JOB_OPTS=landopts
   FAILED=failed
   FAILED_JOBS_TMP_FILE=failedtmp
   FAILED_JOBS_FILE=failedjobs
   MAX_RETRIES=3
   WAIT_TIME_SLEEP=0
   RETRY_SLEEP=4
   BATCH_AND_WALLTIME_ARGS=batch
   OUT_DIR_NAME=ha
   MATLAB_DIR=\$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    return 0
  }
  function getNextIteration {
    NEXT_ITERATION=1
    return 0
  }
  function getCHMTestJobParametersForTaskFromConfig {
    INPUT_IMAGE=$THE_TMP/foo/blah
    MODEL_DIR=$THE_TMP/model
    return 0
  }
  function runJobs {
    echo \$*
    return 0
  }
  function chumData {
    return 0
  }
  function landData {
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH
  [ "$status" -eq 0 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]

  [ "${lines[2]}" == "run 1 out chm_job cast chum.q landopts failed failedtmp failedjobs 3 iteration 4 batch ha" ]
  [ "${lines[3]}" == "CHMTest successfully run." ]
  [ "${lines[4]}" == "Full run 0 0" ]


}


# full run successful with -n flag
@test "full run successful with -n flag" {

 /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2


 # make a fake helperfuncs file to test main script
  echo "
  function getFullPath {
  GETFULLPATHRET=\$THE_TMP
  return 0
  }
  function parseProperties {
   RUN_CHM_SH=run
   OUTPUT_DIR=out
   CHM_TEST_ITERATION_FILE=iteration
   CHM_TEST_CAST_OUT_FILE=cast
   CHUMMEDLIST=chum.q
   LAND_JOB_OPTS=landopts
   FAILED=failed
   FAILED_JOBS_TMP_FILE=failedtmp
   FAILED_JOBS_FILE=failedjobs
   MAX_RETRIES=3
   WAIT_TIME_SLEEP=0
   RETRY_SLEEP=4
   BATCH_AND_WALLTIME_ARGS=batch
   OUT_DIR_NAME=ha
   MATLAB_DIR=\$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    return 0
  }
  function getNextIteration {
    NEXT_ITERATION=1
    return 0
  }
  function getCHMTestJobParametersForTaskFromConfig {
    INPUT_IMAGE=$THE_TMP/foo/blah
    MODEL_DIR=$THE_TMP/model
    return 0
  }
  function runJobs {
    echo \$*
    return 0
  }
  function chumData {
    return 0
  }
  function landData {
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH -n hello
  [ "$status" -eq 0 ]
  echo ":$output:" 1>&2
  [ "${lines[2]}" == "run 1 out hello cast chum.q landopts failed failedtmp failedjobs 3 iteration 4 batch ha" ]
  [ "${lines[4]}" == "Full run 0 0" ]
}

# -C flag to check run where run has failed jobs
@test "-C flag to check run where run has failed jobs" {
  
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "CHM_TEST_ITERATION_FILE=\"chm.test.iteration\"
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=10
    return 0
  }
  function verifyResults {
    echo \$*
    NUM_FAILED_JOBS=3
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH -C
  [ "$status" -eq 1 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[1]}" == "Checking results..." ]
  [ "${lines[2]}" == "1 $THE_TMP 1 10 no" ]
  [ "${lines[3]}" == "3 out of 10 job(s) failed." ]
  [ "${lines[4]}" == "Full run 0 1" ]
}

# -C flag to check run where run is successful
@test "-C flag to check run where run is successful" {
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "CHM_TEST_ITERATION_FILE=\"chm.test.iteration\"
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=10
    return 0
  }
  function verifyResults {
    echo \$*
    NUM_FAILED_JOBS=0
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH -C 
  [ "$status" -eq 0 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[1]}" == "Checking results..." ]
  [ "${lines[2]}" == "1 $THE_TMP 1 10 no" ]
  [ "${lines[3]}" == "All 10 job(s) completed successfully." ]
  [ "${lines[4]}" == "Full run 0 0" ]
}

# -D flag where download fails
@test "-D flag where download fails" {
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "CHM_TEST_ITERATION_FILE=\"chm.test.iteration\"
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   CHUMMEDLIST=foo
   LAND_JOB_OPTS=yikes
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=10
    return 0
  }
  function landData {
    echo \$*
    return 1
  }
  
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH -D
  [ "$status" -eq 1 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[1]}" == "Downloading/Landing data..." ]
  [ "${lines[2]}" == "foo $THE_TMP yikes 0 0" ]
  [ "${lines[3]}" == "Unable to retreive data" ]
  [ "${lines[4]}" == "Full run 0 1" ]
}

# -D flag where download is successful
@test "-D flag where download is successful" {
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "CHM_TEST_ITERATION_FILE=\"chm.test.iteration\"
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   CHUMMEDLIST=foo
   LAND_JOB_OPTS=yikes
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=10
    return 0
  }
  function landData {
    echo \$*
    return 0
  }
  
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH -D
  [ "$status" -eq 0 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[1]}" == "Downloading/Landing data..." ]
  [ "${lines[2]}" == "foo $THE_TMP yikes 0 0" ]
  [ "${lines[3]}" == "Download successful." ]
  [ "${lines[4]}" == "Full run 0 0" ]
}

# -U flag where upload fails
@test "-U flag where upload fails" {
  
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "CHM_TEST_ITERATION_FILE=\"chm.test.iteration\"
  RUN_CHM_SH=runchm
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   CHUMMEDLIST=foo
   LAND_JOB_OPTS=yikes
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=10
    return 0
  }
  function chumJobData {
    echo \$*
    return 1
  }
  
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH -U
  [ "$status" -eq 1 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[1]}" == "Uploading/Chumming data..." ]
  [ "${lines[2]}" == "runchm 1 $THE_TMP" ]
  [ "${lines[3]}" == "Unable to upload data" ]
  [ "${lines[4]}" == "Full run 0 1" ]
}


# -U flag where upload is successful
@test "-U flag where upload is successful" {
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2

  echo "1" > "$THE_TMP/chm.test.iteration"
  /bin/mv "$FAKEHELPERFUNCS" "$THE_TMP/.helperfuncs.sh" 1>&2

  # make a fake helperfuncs file to test main script
  echo "CHM_TEST_ITERATION_FILE=\"chm.test.iteration\"
  RUN_CHM_SH=runchm
  function getFullPath {
  GETFULLPATHRET=$THE_TMP
  return 0
  }
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   CHUMMEDLIST=foo
   LAND_JOB_OPTS=yikes
   return 0
  }
  function getNumberOfJobsFromConfig {
    NUMBER_JOBS=10
    return 0
  }
  function chumJobData {
    echo \$*
    return 0
  }
  
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM_VIA_PANFISH -U
  [ "$status" -eq 0 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[1]}" == "Uploading/Chumming data..." ]
  [ "${lines[2]}" == "runchm 1 $THE_TMP" ]
  [ "${lines[3]}" == "Upload successful." ]
  [ "${lines[4]}" == "Full run 0 0" ]
}

