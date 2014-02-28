#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runCHMJobViaPanfish" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../panfishCHM.properties "$THE_TMP/." 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../*.sh "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../../../algorithm "$THE_TMP/CHM" 1>&2
  
  export CREATECHM="$THE_TMP/createCHMJob.sh"
  export RUNCHM="$THE_TMP/run/runCHMJobViaPanfish.sh"
  chmod a+x $CREATECHM

  /bin/cp -a "${BATS_TEST_DIRNAME}/bin/panfish" "${THE_TMP}/."
  
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  echo "2" 1>&2
}

#
# No args
#
@test "No args" {
  skip 
  # create a fake model folder
  mkdir -p "$THE_TMP/model" 
  echo "fake" > "$THE_TMP/model/param.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage2.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level1_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level2_stage1.mat"

  # create fake images
  mkdir -p "$THE_TMP/images"
  echo "fake" > "$THE_TMP/images/1.png"
  echo "fake" > "$THE_TMP/images/2.png"

  run $CREATECHM createpretrained "$THE_TMP/run" -m "$THE_TMP/model" -i "$THE_TMP/images"
  [ "$status" -eq 0 ]
  
  run $RUNCHM

  [ "$status" -eq 1 ] 

  [ "${lines[0]}" == "Run CHM via Panfish." ]
}

#
# invalid mode
#
@test "invalid mode" {
  skip
  # create a fake model folder
  mkdir -p "$THE_TMP/model"
  echo "fake" > "$THE_TMP/model/param.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage2.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level1_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level2_stage1.mat"

  # create fake images
  mkdir -p "$THE_TMP/images"
  echo "fake" > "$THE_TMP/images/1.png"
  echo "fake" > "$THE_TMP/images/2.png"

  run $CREATECHM createpretrained "$THE_TMP/run" -m "$THE_TMP/model" -i "$THE_TMP/images"
  [ "$status" -eq 0 ]

  run $RUNCHM foo
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    Mode foo not supported.  Invoke with -h for list of valid options." ]
}

#
# fullrun
#
@test "fullrun" {
  skip
  # create a fake model folder
  mkdir -p "$THE_TMP/model"
  echo "fake" > "$THE_TMP/model/param.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage2.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level1_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level2_stage1.mat"

  # create fake images
  mkdir -p "$THE_TMP/images"
  echo "fake" > "$THE_TMP/images/1.png"
  echo "fake" > "$THE_TMP/images/2.png"

  run $CREATECHM createpretrained "$THE_TMP/run" -m "$THE_TMP/model" -i "$THE_TMP/images"
  [ "$status" -eq 0 ]

  # make a fake configuration file
  echo "panfish.bin.dir=$THE_TMP/panfish" > $THE_TMP/run/panfishCHM.properties
  echo "matlab.dir=/tmp" >> $THE_TMP/run/panfishCHM.properties
  echo "batch.and.walltime.args=batchy" >> $THE_TMP/run/panfishCHM.properties
  echo "cluster.list=clist" >> $THE_TMP/run/panfishCHM.properties

  # make task files for the panfish commands so the run script will think the job was successful.

  run $RUNCHM fullrun
  [ "$status" -eq 0 ]
  [ "${lines[1]}" == "ERROR:    Mode foo not supported.  Invoke with -h for list of valid options." ]
}

