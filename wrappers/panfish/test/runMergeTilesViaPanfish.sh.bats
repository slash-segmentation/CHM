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
  run checkSingleTask "$THE_TMP/doesnotexist" "1"
  [ "$status" -eq 1 ]

  mkdir -p "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME"
  # Test where stdout file does not exist
  run checkSingleTask "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  # test where stdout file size is 0
  touch "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  run checkSingleTask "$THE_TMP" "1"
  [ "$status" -eq 2 ]


  # test where there is a problem parsing config
  echo "hi" > "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME/1.stdout"
  run checkSingleTask "$THE_TMP" "1"
  [ "$status" -eq 3 ]


  # test where there is no output file
  echo "1${CONFIG_DELIM}yo" > "$THE_TMP/$RUN_MERGE_TILES_CONFIG"
  echo "1${CONFIG_DELIM}foo.png" >> "$THE_TMP/$RUN_MERGE_TILES_CONFIG"
  run checkSingleTask "$THE_TMP" "1"
  echo "$status is" 1>&2
  [ "$status" -eq 4 ]

  # test success
  echo "hi" > "$THE_TMP/foo.png"
  run checkSingleTask "$THE_TMP" "1"
  [ "$status" -eq 0 ]
}

