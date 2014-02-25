#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/createCHMJob" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../panfishCHM.properties "$THE_TMP/." 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../*.sh "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../../../algorithm "$THE_TMP/CHM" 1>&2
  export TESTIMAGE_DIR="${BATS_TEST_DIRNAME}/createchmjob"
  export TESTIMAGE="${TESTIMAGE_DIR}/600by400.png"
  export CREATECHM="$THE_TMP/createCHMJob.sh"
  chmod a+x $CREATECHM
}

teardown(){
  #/bin/rm -rf "$THE_TMP"
  echo "2" 1>&2
}


#
# no args
#
@test "no args" {
  run $CREATECHM
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "Create CHM job." ]
}

#
# one arg
#
@test "one arg" {
  run $CREATECHM createpretrained

  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "Create CHM job." ]
}

#
# createpretrained and valid job directory  but no model
#
@test "createpretrained and valid job directory but no model" {
  run $CREATECHM createpretrained "$THE_TMP"

  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    Model must be specified via -m flag" ]
}

#
# createpretrained and valid job directory  but no images
#
@test "createpretrained and valid job directory but no images" {
  run $CREATECHM createpretrained "$THE_TMP" -m "$THE_TMP"

  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "ERROR:    Image Folder must be specified via -i flag" ]
}

#
# createpretrained with -m -b and -i specified properly
# This test has a lot of checks cause its checking everything was
# created properly
#
@test "createpretrained with -m -b and -i specified properly" {
  # create a fake model folder
  mkdir -p "$THE_TMP/model" 
  echo "fake" > "$THE_TMP/model/param.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage2.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level1_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level2_stage1.mat"


  run $CREATECHM createpretrained "$THE_TMP/run" -m "$THE_TMP/model" -i "$TESTIMAGE_DIR" -b 200x100 
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == *createpretrained* ]]
  [ "${lines[1]}" == "  Creating directory $THE_TMP/run" ]

  # Verify runCHMviaPanfish.sh was copied over
  [ -x "$THE_TMP/run/runCHMJobViaPanfish.sh" ]

  # Verify panfishCHM.properties was copied over
  [ -s "$THE_TMP/run/panfishCHM.properties" ]

  # Verify out directory was created
  [ -d "$THE_TMP/run/out/log/chm" ]
  [ -d "$THE_TMP/run/out/log/joberr" ]
  [ -d "$THE_TMP/run/out/log/jobout" ]

  # Verify runCHM.sh was copied over and is executable
  [ -s "$THE_TMP/run/runCHM.sh" ]
  [ -x "$THE_TMP/run/runCHM.sh" ]

  # Verify helperfuncs.sh was copied over
  [ -s "$THE_TMP/run/helperfuncs.sh" ]

  # Verify runCHM.sh.config was properly created
  [ -s "$THE_TMP/run/runCHM.sh.config" ]

  # verify we have 36 lines
  [ `wc -l "$THE_TMP/run/runCHM.sh.config" | sed "s/ .*//"` == "36" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/run/runCHM.sh.config"`
  [ "$aLine" == "1:::$TESTIMAGE" ]
  
  aLine=`head -n 2 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::   -b 200x100  -t 3,4" ]

  aLine=`head -n 3 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::out/600by400.png/1.png" ]

  aLine=`head -n 4 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$TESTIMAGE" ]

  aLine=`head -n 5 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::   -b 200x100  -t 3,3" ]

  aLine=`head -n 6 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::out/600by400.png/2.png" ]

  aLine=`head -n 34 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "12:::$TESTIMAGE" ]

  aLine=`head -n 35 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "12:::   -b 200x100  -t 1,1" ]

  aLine=`head -n 36 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "12:::out/600by400.png/12.png" ]
 
  [ -d "$THE_TMP/run/CHM/out" ]

  [ -x "$THE_TMP/run/CHM/CHM_test.sh" ] 

  [ -s "$THE_TMP/run/CHM/out/param.mat" ]
  [ -s "$THE_TMP/run/CHM/out/MODEL_level0_stage1.mat" ]
  [ -s "$THE_TMP/run/CHM/out/MODEL_level0_stage2.mat" ]  
  [ -s "$THE_TMP/run/CHM/out/MODEL_level1_stage1.mat" ]
  [ -s "$THE_TMP/run/CHM/out/MODEL_level2_stage1.mat" ]
}

#
# createpretrained with -m -b -i -T specified properly
# This test has a lot of checks cause its checking everything was
# created properly
#
@test "createpretrained with -m -b -i and -T specified properly" {
  # create a fake model folder
  mkdir -p "$THE_TMP/model"
  echo "fake" > "$THE_TMP/model/param.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage2.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level1_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level2_stage1.mat"


  run $CREATECHM createpretrained "$THE_TMP/run" -m "$THE_TMP/model" -i "$TESTIMAGE_DIR" -b 200x100 -T 12
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == *createpretrained* ]]
  [ "${lines[1]}" == "  Creating directory $THE_TMP/run" ]

  # Verify runCHMviaPanfish.sh was copied over
  [ -x "$THE_TMP/run/runCHMJobViaPanfish.sh" ]

  # Verify panfishCHM.properties was copied over
  [ -s "$THE_TMP/run/panfishCHM.properties" ]

  # Verify out directory was created
  [ -d "$THE_TMP/run/out/log/chm" ]
  [ -d "$THE_TMP/run/out/log/joberr" ]
  [ -d "$THE_TMP/run/out/log/jobout" ]

  # Verify runCHM.sh was copied over and is executable
  [ -s "$THE_TMP/run/runCHM.sh" ]
  [ -x "$THE_TMP/run/runCHM.sh" ]

  # Verify helperfuncs.sh was copied over
  [ -s "$THE_TMP/run/helperfuncs.sh" ]

  # Verify runCHM.sh.config was properly created
  [ -s "$THE_TMP/run/runCHM.sh.config" ]

  # verify we have 3 lines
  [ `wc -l "$THE_TMP/run/runCHM.sh.config" | sed "s/ .*//"` == "3" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/run/runCHM.sh.config"`
  [ "$aLine" == "1:::$TESTIMAGE" ]

  aLine=`head -n 2 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::   -b 200x100  -t 3,4 -t 3,3 -t 3,2 -t 3,1 -t 2,4 -t 2,3 -t 2,2 -t 2,1 -t 1,4 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 3 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::out/600by400.png/1.png" ]

}

# createpretrained with -m -b -i -o and -T specified properly
# created properly
#
@test "createpretrained with -m -b -i -o and -T specified properly" {
  # create a fake model folder
  mkdir -p "$THE_TMP/model"
  echo "fake" > "$THE_TMP/model/param.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage2.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level1_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level2_stage1.mat"


  run $CREATECHM createpretrained "$THE_TMP/run" -m "$THE_TMP/model" -i "$TESTIMAGE_DIR" -b 200x100 -T 11 -o 4x5
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == *createpretrained* ]]
  [ "${lines[1]}" == "  Creating directory $THE_TMP/run" ]

  # Verify runCHMviaPanfish.sh was copied over
  [ -x "$THE_TMP/run/runCHMJobViaPanfish.sh" ]

  # Verify panfishCHM.properties was copied over
  [ -s "$THE_TMP/run/panfishCHM.properties" ]

  # Verify out directory was created
  [ -d "$THE_TMP/run/out/log/chm" ]
  [ -d "$THE_TMP/run/out/log/joberr" ]
  [ -d "$THE_TMP/run/out/log/jobout" ]

  # Verify runCHM.sh was copied over and is executable
  [ -s "$THE_TMP/run/runCHM.sh" ]
  [ -x "$THE_TMP/run/runCHM.sh" ]

  # Verify helperfuncs.sh was copied over
  [ -s "$THE_TMP/run/helperfuncs.sh" ]

  # Verify runCHM.sh.config was properly created
  [ -s "$THE_TMP/run/runCHM.sh.config" ]

  # verify we have 3 lines
  [ `wc -l "$THE_TMP/run/runCHM.sh.config" | sed "s/ .*//"` == "6" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/run/runCHM.sh.config"`
  [ "$aLine" == "1:::$TESTIMAGE" ]

  aLine=`head -n 2 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1::: -o 4x5  -b 200x100  -t 3,4 -t 3,3 -t 3,2 -t 3,1 -t 2,4 -t 2,3 -t 2,2 -t 2,1 -t 1,4 -t 1,3 -t 1,2" ]

  aLine=`head -n 3 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::out/600by400.png/1.png" ]

  aLine=`head -n 4 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$TESTIMAGE" ]

  aLine=`head -n 5 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2::: -o 4x5  -b 200x100  -t 1,1" ]

  aLine=`head -n 6 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::out/600by400.png/2.png" ]


}
