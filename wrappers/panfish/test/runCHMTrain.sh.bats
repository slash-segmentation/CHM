#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runCHMTrain" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../panfishCHM.properties "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/runCHMTrain.sh "$THE_TMP/." 1>&2
  export RUNCHMTRAIN="$THE_TMP/runCHMTrain.sh"
  chmod a+x $RUNCHMTRAIN
  export HELPERFUNCS="$THE_TMP/.helperfuncs.sh"
  export SUCCESS_CHM_TRAIN="$BATS_TEST_DIRNAME/bin/fakesuccesschmtrain"
  export FAIL_CHM_TRAIN="$BATS_TEST_DIRNAME/bin/fakefailchmtrain"

  chmod a+x $SUCCESS_CHM_TRAIN
  chmod a+x $FAIL_CHM_TRAIN
  unset SGE_TASK_ID
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  echo "teardown" 1>&2
}

#
# getCHMTrainJobParametersForTaskFromConfig() tests
#
@test "getCHMTestJobParametersForTaskFromConfig() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # source runCHM.sh so we can unit test this function
  . $RUNCHMTRAIN source

  # Test where we can't get first parameter
  run getCHMTrainJobParametersForTaskFromConfig "$THE_TMP" "1"
  [ "$status" -eq 1 ]

  # Test where we can't get 2nd parameter
  echo "1${CONFIG_DELIM}a" > "$THE_TMP/runCHMTrain.sh.config"
  echo "1${CONFIG_DELIM}b" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1${CONFIG_DELIM}c" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1${CONFIG_DELIM}d" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "2${CONFIG_DELIM}e" >> "$THE_TMP/runCHMTrain.sh.config"
  run getCHMTrainJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$status" -eq 2 ]

  # Test where we cant get 3rd parameter
  echo "2${CONFIG_DELIM}f" >> "$THE_TMP/runCHMTrain.sh.config"
  run getCHMTrainJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$status" -eq 3 ]


  # Test where we cant get 4th parameter
  echo "2${CONFIG_DELIM}g" >> "$THE_TMP/runCHMTrain.sh.config"
  run getCHMTrainJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$status" -eq 4 ]

  # Test success
  echo "2${CONFIG_DELIM}h" >> "$THE_TMP/runCHMTrain.sh.config"
  getCHMTrainJobParametersForTaskFromConfig "$THE_TMP" "2"
  [ "$?" -eq 0 ]

  [ "$INPUT_IMAGES" == "e" ]
  [ "$INPUT_LABELS" == "f" ]
  [ "$TRAIN_OPTS" == "g" ]
  [ "$OUTPUT_DIR" == "h" ]
}



#
# SGE_TASK_ID not set
#
@test "SGE_TASK_ID not set" {
 
  unset SGE_TASK_ID
  run $RUNCHMTRAIN
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "This script runs CHM train on a set of images and labels/maps." ]
}

@test "no .helperfuncs.sh" {
 
  /bin/rm -f "$THE_TMP/.helperfuncs.sh"

  run $RUNCHMTRAIN
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "$THE_TMP/.helperfuncs.sh not found" ]
}

#
# No runCHMTrain.sh.config file 
#
@test "no runCHMTrain.sh.config file" {
  
  export SGE_TASK_ID=1
  run $RUNCHMTRAIN
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "ERROR:  (task 1)  No $THE_TMP/runCHMTrain.sh.config found" ]

}

#
# Simple valid run with successful fake CHM_train.sh call PANFISH_BASEDIR unset
#
@test "Simple valid run with successful fake CHM_train.sh call" {
  
  # create fake panfishCHM.properties file
  echo "chm.bin.dir=${SUCCESS_CHM_TRAIN}" > "$THE_TMP/panfishCHM.properties"

  # create runCHMTrain.sh.config file
  echo "1:::/foo/input/" > "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::/foo/labels/" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::trainopts" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::outdir" >> "$THE_TMP/runCHMTrain.sh.config"

  export SGE_TASK_ID=1

  run $RUNCHMTRAIN
  echo "$output" 1>&2 
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "(task 1) runCHMTrain.sh Start Time:"* ]]
  [ "${lines[1]}" == "//foo/input/ //foo/labels/ -m $THE_TMP/outdir trainopts -s" ]
  [[ "${lines[25]}" == "(task 1) runCHMTrain.sh End Time: "* ]]
  [[ "${lines[25]}" == *" Exit Code: 0" ]]
}

#
# Simple valid run with successful fake CHM_test.sh call PANFISH_BASEDIR  and PANFISH_SCRATCH set
#
@test "Simple valid run with successful fake CHM_test.sh call PANFISH_BASEDIR and PANFISH_SCRATCH set" {
  
  export SGE_TASK_ID=1
  export PANFISH_BASEDIR="$THE_TMP"
  export PANFISH_SCRATCH="/tmp/pan"
 
  mkdir -p "$THE_TMP/cc/" 1>&2
  /bin/cp -a ${SUCCESS_CHM_TRAIN}/* "$THE_TMP/cc/." 1>&2

  # create fake panfishCHM.properties file
  echo "chm.bin.dir=/cc" > "$THE_TMP/panfishCHM.properties"

  # create runCHMTrain.sh.config file
  echo "1:::/foo/input/" > "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::/foo/labels/" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::trainopts" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::outdir" >> "$THE_TMP/runCHMTrain.sh.config"

  run $RUNCHMTRAIN
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "(task 1) runCHMTrain.sh Start Time:"* ]]
  [ "${lines[1]}" == "$THE_TMP//foo/input/ $THE_TMP//foo/labels/ -m $THE_TMP/outdir trainopts -s" ]
  [[ "${lines[25]}" == "(task 1) runCHMTrain.sh End Time: "* ]]
  [[ "${lines[25]}" == *" Exit Code: 0" ]]

}

# Simple valid run with successful fake CHM_train.sh call using compiled CHM
@test "Simple valid run with successful fake CHM_train.sh call using compiled CHM" {

  # create fake panfishCHM.properties file
  echo "chm.bin.dir=$THE_TMP/cc" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=/hi" >> "$THE_TMP/panfishCHM.properties"

  mkdir -p "$THE_TMP/cc/" 1>&2
  /bin/cp -a ${SUCCESS_CHM_TRAIN}/* "$THE_TMP/cc/." 1>&2
  echo "hi" > "$THE_TMP/cc/CHM_train"

  # create runCHMTrain.sh.config file
  echo "1:::/foo/input/" > "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::/foo/labels/" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::trainopts" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::outdir" >> "$THE_TMP/runCHMTrain.sh.config"
 
  export SGE_TASK_ID=1

  run $RUNCHMTRAIN
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "(task 1) runCHMTrain.sh Start Time:"* ]]
  [ "${lines[1]}" == "//foo/input/ //foo/labels/ -m $THE_TMP/outdir trainopts -M //hi" ]
  [[ "${lines[25]}" == "(task 1) runCHMTrain.sh End Time: "* ]]
  [[ "${lines[25]}" == *" Exit Code: 0" ]]
}


# Simple valid run with failing fake CHM_train.sh call
@test "Simple valid run with failing fake CHM_train.sh call using compiled CHM" {

  # create fake panfishCHM.properties file
  echo "chm.bin.dir=$THE_TMP/cc" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=/hi" >> "$THE_TMP/panfishCHM.properties"

  mkdir -p "$THE_TMP/cc/" 1>&2
  /bin/cp -a ${FAIL_CHM_TRAIN}/* "$THE_TMP/cc/." 1>&2
  echo "hi" > "$THE_TMP/cc/CHM_train"

  # create runCHMTrain.sh.config file
  echo "1:::/foo/input/" > "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::/foo/labels/" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::trainopts" >> "$THE_TMP/runCHMTrain.sh.config"
  echo "1:::outdir" >> "$THE_TMP/runCHMTrain.sh.config"
  
  export SGE_TASK_ID=1

  run $RUNCHMTRAIN
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == "(task 1) runCHMTrain.sh Start Time:"* ]]
  [ "${lines[1]}" == "//foo/input/ //foo/labels/ -m $THE_TMP/outdir trainopts -M //hi" ]
  [[ "${lines[26]}" == "(task 1) runCHMTrain.sh End Time: "* ]]
  [[ "${lines[26]}" == *" Exit Code: 1" ]]
}


