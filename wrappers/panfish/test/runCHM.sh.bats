#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runCHM" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../panfishCHM.properties "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/runCHM.sh "$THE_TMP/." 1>&2
  export RUNCHM="$THE_TMP/runCHM.sh"
  chmod a+x $RUNCHM

  export SUCCESS_CHM_TEST="$BATS_TEST_DIRNAME/bin/fakesuccesschmtest"
  export FAIL_CHM_TEST="$BATS_TEST_DIRNAME/bin/fakefailchmtest"

  chmod a+x $SUCCESS_CHM_TEST
  chmod a+x $FAIL_CHM_TEST
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  #echo "teardown" 1>&2
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
  mkdir -p "$THE_TMP/out/hist1.png/chm" 1>&2

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

