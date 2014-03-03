#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runCHMJobViaPanfish" 1>&2
  /bin/mkdir -p "$THE_TMP/panfish" 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/runCHMViaPanfish.sh "$THE_TMP/." 1>&2
    
  export RUNCHM="$THE_TMP/runCHMViaPanfish.sh"
  chmod a+x $RUNCHM
  unset SGE_TASK_ID
  /bin/cp -a "${BATS_TEST_DIRNAME}/bin/panfish" "${THE_TMP}/panfish/."
  
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
# -h flag
#
@test "-h flag" {
  
  run $RUNCHM -h

  [ "$status" -eq 1 ] 
  echo "$output" 1>&2
  [ "${lines[0]}" == "Run CHM via Panfish." ]
}

#
# invalid flag
#
@test "invalid flag" {

  run $RUNCHM -asdf

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
  run $RUNCHM
  [ "$status" -eq 2 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "No $THE_TMP/.helperfuncs.sh found" ]
}

#
# no panfishCHM.properties
#
@test "no panfishCHM.properties" {

  run $RUNCHM
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    There was a problem parsing the panfishCHM.properties file" ]

}

#
# Unable to get number of jobs 
#
@test "Unable to get number of jobs" {

  echo "panfish.bin.dir=$THE_TMP/panfish" > "$THE_TMP/panfishCHM.properties"

  run $RUNCHM
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

  run $RUNCHM
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[1]}" == "ERROR:    Unable to get path to matlab directory: " ]
}

#
# Unable to get parameters from first job
#
@test "Unable to get parameters from first job" {

  echo "panfish.bin.dir=$THE_TMP/panfish" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=$THE_TMP" >> "$THE_TMP/panfishCHM.properties"
  echo "1:::a" > "$THE_TMP/runCHM.sh.config"

  run $RUNCHM
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[1]}" == "ERROR:    Error parsing the first job from config" ]



}


#
# full run failure no iteration file
#
@test "fullrun failure no iteration file" {

  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2
  
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
  function getNumberOfCHMTestJobsFromConfig {
    return 0
  }
  function getCHMTestJobParametersForTaskFromConfig {
    INPUT_IMAGE=$THE_TMP/foo/blah
    MODEL_DIR=$THE_TMP/model
    return 0
  }
  function runCHMTestJobs {
    echo \$*
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM
  [ "$status" -eq 1 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[1]}" == "1 $THE_TMP $THE_TMP/foo $THE_TMP/model chm_job" ]
  [ "${lines[2]}" == "Full run 0 1" ]
  [ "${lines[3]}" == "Error running CHMTest" ]
}

#
# full run failure with iteration file
#
@test "full run failure with iteration file" {
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2


  echo "4" > "$THE_TMP/chm.test.iteration"  
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
  function getNumberOfCHMTestJobsFromConfig {
    return 0
  }
  function getCHMTestJobParametersForTaskFromConfig {
    INPUT_IMAGE=$THE_TMP/foo/blah
    MODEL_DIR=$THE_TMP/model
    return 0
  }
  function runCHMTestJobs {
    echo \$*
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM
  [ "$status" -eq 1 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[2]}" == "5 $THE_TMP $THE_TMP/foo $THE_TMP/model chm_job" ]
  [ "${lines[3]}" == "Full run 0 1" ]
  [ "${lines[4]}" == "Error running CHMTest" ]
}


# full run successful with -n flag
@test "full run successful with -n flag" {
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
  function getNumberOfCHMTestJobsFromConfig {
    return 0
  }
  function getCHMTestJobParametersForTaskFromConfig {
    INPUT_IMAGE=$THE_TMP/foo/blah
    MODEL_DIR=$THE_TMP/model
    return 0
  }
  function runCHMTestJobs {
    echo \$*
    return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM -n hello
  [ "$status" -eq 0 ]
  echo ":$output:" 1>&2
  [ "${lines[0]}" == "Full run" ]
  [ "${lines[2]}" == "2 $THE_TMP $THE_TMP/foo $THE_TMP/model hello" ]
  [ "${lines[3]}" == "CHMTest successfully run." ]
  [ "${lines[4]}" == "Full run 0 0" ]
}
