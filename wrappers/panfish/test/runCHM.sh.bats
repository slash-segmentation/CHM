#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runCHM" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../panfishCHM.properties "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/runCHM.sh "$THE_TMP/." 1>&2
  export RUNCHM="$THE_TMP/runCHM.sh"
  chmod a+x $RUNCHM
  export HELPERFUNCS="$THE_TMP/.helperfuncs.sh"
  export SUCCESS_CHM_TEST="$BATS_TEST_DIRNAME/bin/fakesuccesschmtest"
  export FAIL_CHM_TEST="$BATS_TEST_DIRNAME/bin/fakefailchmtest"

  chmod a+x $SUCCESS_CHM_TEST
  chmod a+x $FAIL_CHM_TEST
  unset SGE_TASK_ID
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  #echo "teardown" 1>&2
}

#
# getCHMTestJobParametersForTaskFromConfig() tests
#
@test "getCHMTestJobParametersForTaskFromConfig() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # source runCHM.sh so we can unit test this function
  . $RUNCHM source

  # Test where we can't get first parameter
  run getCHMTestJobParametersForTaskFromConfig "$THE_TMP" "1"
  [ "$status" -eq 1 ]

  # Test where we can't get 2nd parameter
  echo "1${CONFIG_DELIM}a" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}b" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}c" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}d" >> "$THE_TMP/runCHM.sh.config"
  echo "2${CONFIG_DELIM}e" >> "$THE_TMP/runCHM.sh.config"
  run getCHMTestJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$status" -eq 2 ]

  # Test where we cant get 3rd parameter
  echo "2${CONFIG_DELIM}f" >> "$THE_TMP/runCHM.sh.config"
  run getCHMTestJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$status" -eq 3 ]


  # Test where we cant get 4th parameter
  echo "2${CONFIG_DELIM}g" >> "$THE_TMP/runCHM.sh.config"
  run getCHMTestJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$status" -eq 4 ]

  # Test success
  echo "2${CONFIG_DELIM}h" >> "$THE_TMP/runCHM.sh.config"
  getCHMTestJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$?" -eq 0 ]

  [ "$INPUT_IMAGE" == "e" ]
  [ "$MODEL_DIR" == "f" ]
  [ "$CHM_OPTS" == "g" ]
  [ "$OUTPUT_IMAGE" == "h" ]
}



#
# SGE_TASK_ID not set
#
@test "SGE_TASK_ID not set" {
  unset SGE_TASK_ID
  run $RUNCHM
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "This script runs CHM on a slice/image of data." ]
}

@test "no .helperfuncs.sh" {
  /bin/rm -f "$THE_TMP/.helperfuncs.sh"

  run $RUNCHM
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "$THE_TMP/.helperfuncs.sh not found" ]
}

#
# No runCHM.sh.config file 
#
@test "no runCHM.sh.config file" {
  export SGE_TASK_ID=1
  run $RUNCHM
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "ERROR:  (task 1)  No $THE_TMP/runCHM.sh.config found" ]

}

#
# Simple valid run with successful fake CHM_test.sh call PANFISH_BASEDIR unset
#
@test "Simple valid run with successful fake CHM_test.sh call" {
  export SGE_TASK_ID=1

  # create fake panfishCHM.properties file
  echo "chm.bin.dir=${SUCCESS_CHM_TEST}" > "$THE_TMP/panfishCHM.properties"

  # create runCHM.sh.config file
  echo "1:::/foo/input.png" > "$THE_TMP/runCHM.sh.config"
  echo "1:::/foo/modeldir" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::chmopts" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::out/hist1.png/1.png" >> "$THE_TMP/runCHM.sh.config"
  
  # make output directory
  mkdir -p "$THE_TMP/out/hist1.png/chm" 1>&2

  export SGE_TASK_ID=1

  run $RUNCHM
  echo "$output" 1>&2 
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "(task 1) runCHM.sh Start Time:"* ]]
  [[ "${lines[1]}" == "(task 1)  Creating directory /tmp/"* ]]
  [[ "${lines[2]}" == "//foo/input.png /tmp/chm"* ]]
  [[ "${lines[2]}" == *" -m //foo/modeldir chmopts -s" ]]
  [[ "${lines[26]}" == "(task 1) runCHM.sh End Time: "* ]]
  [[ "${lines[26]}" == *" Exit Code: 0" ]]
}

#
# Simple valid run with successful fake CHM_test.sh call PANFISH_BASEDIR  and PANFISH_SCRATCH set
#
@test "Simple valid run with successful fake CHM_test.sh call PANFISH_BASEDIR and PANFISH_SCRATCH set" {
  export SGE_TASK_ID=1
  export PANFISH_BASEDIR="$THE_TMP"
  export PANFISH_SCRATCH="/tmp/pan"
  unset SKIP_COPY
  mkdir -p "$THE_TMP/cc/" 1>&2
  /bin/cp -a ${SUCCESS_CHM_TEST}/* "$THE_TMP/cc/." 1>&2

  # create fake panfishCHM.properties file
  echo "chm.bin.dir=/cc" > "$THE_TMP/panfishCHM.properties"



  # create runCHM.sh.config file
  echo "1:::/foo/input.png" > "$THE_TMP/runCHM.sh.config"
  echo "1:::/foo/modeldir" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::chmopts" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::out/hist1.png/1.png" >> "$THE_TMP/runCHM.sh.config"

  # make output directory
  mkdir -p "$THE_TMP/out/hist1.png/" 1>&2

  export SGE_TASK_ID=1

  run $RUNCHM
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "(task 1) runCHM.sh Start Time:"* ]]
  [[ "${lines[1]}" == "(task 1)  Creating directory /tmp/pan/chm"* ]]
  [[ "${lines[2]}" == "$THE_TMP//foo/input.png /tmp/pan/chm"* ]]
  [[ "${lines[2]}" == *" -m $THE_TMP//foo/modeldir chmopts -s" ]]
  [[ "${lines[26]}" == "(task 1) runCHM.sh End Time: "* ]]
  [[ "${lines[26]}" == *" Exit Code: 0" ]]

  [ -s "$THE_TMP/out/hist1.png/1.png" ]
}


@test "Simple valid run with successful fake CHM_test.sh call using compiled CHM" {
  export SGE_TASK_ID=1
  export PANFISH_BASEDIR="$THE_TMP"
  export PANFISH_SCRATCH="/tmp/pan"
  unset SKIP_COPY
  mkdir -p "$THE_TMP/cc/" 1>&2
  /bin/cp -a ${SUCCESS_CHM_TEST}/* "$THE_TMP/cc/." 1>&2

  # create fake panfishCHM.properties file
  echo "chm.bin.dir=/cc" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=/hello" >> "$THE_TMP/panfishCHM.properties" 
  echo "hi" > "$THE_TMP/cc/CHM_test"

  # create runCHM.sh.config file
  echo "1:::/foo/input.png" > "$THE_TMP/runCHM.sh.config"
  echo "1:::/foo/modeldir" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::chmopts" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::out/hist1.png/1.png" >> "$THE_TMP/runCHM.sh.config"

  # make output directory
  mkdir -p "$THE_TMP/out/hist1.png/" 1>&2

  export SGE_TASK_ID=1

  run $RUNCHM
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "(task 1) runCHM.sh Start Time:"* ]]
  [[ "${lines[1]}" == "(task 1)  Creating directory /tmp/pan/chm"* ]]
  [[ "${lines[2]}" == "$THE_TMP//foo/input.png /tmp/pan/chm"* ]]
  [[ "${lines[2]}" == *" -m $THE_TMP//foo/modeldir chmopts -s -M $THE_TMP//hello" ]]
  [[ "${lines[26]}" == "(task 1) runCHM.sh End Time: "* ]]
  [[ "${lines[26]}" == *" Exit Code: 0" ]]

  [ -s "$THE_TMP/out/hist1.png/1.png" ]
}



# Simple valid run with failing fake CHM_test.sh call
@test "Simple valid run with failing fake CHM_test.sh call" {
  export SGE_TASK_ID=3
  unset SKIP_COPY
  # create fake panfishCHM.properties file
  echo "chm.bin.dir=${FAIL_CHM_TEST}" > "$THE_TMP/panfishCHM.properties"

  # create runCHM.sh.config file
  echo "3:::/foo/input.png" > "$THE_TMP/runCHM.sh.config"
  echo "3:::/foo/modeldir" >> "$THE_TMP/runCHM.sh.config"
  echo "3:::chmopts" >> "$THE_TMP/runCHM.sh.config"
  echo "3:::out/hist1.png/1.png" >> "$THE_TMP/runCHM.sh.config"

  # make output directory
  mkdir -p "$THE_TMP/out/hist1.png/chm" 1>&2


  run $RUNCHM
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [[ "${lines[1]}" == "(task 3)  Creating directory /tmp/"* ]]
  [[ "${lines[2]}" == "//foo/input.png /tmp/chm"* ]]
  [[ "${lines[2]}" == *" -m //foo/modeldir chmopts -s" ]]
  [[ "${lines[27]}" == "(task 3) runCHM.sh End Time: "* ]]
  [[ "${lines[27]}" == *" Exit Code: 1" ]]

}

