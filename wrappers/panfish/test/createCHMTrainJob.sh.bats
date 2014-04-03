#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/createCHMTrainJob" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2

  /bin/cp ${BATS_TEST_DIRNAME}/../*.sh "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts "$THE_TMP/." 1>&2
  export CREATECHMTRAIN="$THE_TMP/createCHMTrainJob.sh"
  chmod a+x $CREATECHMTRAIN
  export HELPERFUNCS="$THE_TMP/scripts/.helperfuncs.sh"

  # should match OUT_DIR_NAME variable in .helperfuncs
  export BATS_TEST_OUT_DIR_NAME="runCHMTrainOut"
  export BATS_STDERR="stderr"
  export BATS_STDOUT="stdout"
  export BATS_TRAINEDMODEL="trainedmodel"
  export BATS_CONFIGDELIM=":::"
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  echo "teardown" 1>&2
}

#
# createCHMTrainOutputDirectories() tests
#
@test "createCHMTrainOutputDirectories() tests" {

  # source createCHMTrainJob.sh so we can call the function
  . $CREATECHMTRAIN source

  # source .helperfuncs.sh 
  . $HELPERFUNCS

  run createCHMTrainOutputDirectories $THE_TMP 
  [ "$status" -eq 0 ]
  ls -lR "$THE_TMP" 1>&2
  [ -d "$THE_TMP/$BATS_TEST_OUT_DIR_NAME/$BATS_TRAINEDMODEL" ]
  [ -d "$THE_TMP/$BATS_TEST_OUT_DIR_NAME/$BATS_STDERR" ]
  [ -d "$THE_TMP/$BATS_TEST_OUT_DIR_NAME/$BATS_STDOUT" ]
}

#
# createCHMTrainConfig() tests
#
@test "createCHMTrainConfig() tests" {

  # source createCHMTrainJob.sh so we can call the function
  . $CREATECHMTRAIN source

  # source .helperfuncs.sh 
  . $HELPERFUNCS

  # Config exists and there is an error removing it
  local outConfig="$THE_TMP/my.config"
  echo "hi" > $outConfig
  RM_CMD="/bin/false"
  run createCHMTrainConfig "inputimages" "inputlabels" "trainopts" "$outConfig"
  [ "$status" -eq 1 ]

  # Config exists with no error removing
  RM_CMD="/bin/rm"

  run createCHMTrainConfig "inputimages" "inputlabels" "trainopts" "$outConfig"
  [ "$status" -eq 0 ]
  [ -s "$outConfig" ]

  local aLine=`head -n 1 $outConfig`
  [ "$aLine" == "1${BATS_CONFIGDELIM}inputimages" ]
  aLine=`head -n 2 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}inputlabels" ]
  aLine=`head -n 3 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}trainopts" ]
  aLine=`head -n 4 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}${BATS_TEST_OUT_DIR_NAME}/$BATS_TRAINEDMODEL" ]
 
  # No config
  /bin/rm -f "$outConfig" 1>&2
  run createCHMTrainConfig "inputimages" "inputlabels" "trainopts" "$outConfig"
  [ "$status" -eq 0 ]
  [ -s "$outConfig" ]
 
}

#
# runCreateBasicTrainMode() tests
#
@test "runCreateBasicTrainMode() tests" {
  
  # source createCHMTrainJob.sh so we can call the function
  . $CREATECHMTRAIN source

  # source .helperfuncs.sh 
  . $HELPERFUNCS

  # Successful create test
  echo "hi" > "$THE_TMP/panfishCHM.properties"
  run runCreateBasicTrainMode "$THE_TMP" "$THE_TMP/scripts" "$THE_TMP/run" "$THE_TMP/images" "$THE_TMP/labels" "-S 3 -L 3"
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [ -x "$THE_TMP/run/runCHMTrain.sh" ]
  [ -x "$THE_TMP/run/runCHMTrainViaPanfish.sh" ]
  [ -s "$THE_TMP/run/.helperfuncs.sh" ]
  
  local outConfig="$THE_TMP/run/runCHMTrain.sh.config" 
  [ -s "$outConfig" ]

  local aLine=`head -n 1 $outConfig`
  [ "$aLine" == "1${BATS_CONFIGDELIM}$THE_TMP/images" ]
  aLine=`head -n 2 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}$THE_TMP/labels" ]
  aLine=`head -n 3 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}-S 3 -L 3" ]
  aLine=`head -n 4 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}${BATS_TEST_OUT_DIR_NAME}/$BATS_TRAINEDMODEL" ]
}

#
# createCHMTrainJob.sh no args and one arg
#
@test "createCHMTrainJob.sh no args" {

  run $CREATECHMTRAIN
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "Create CHM Train job." ]

  run $CREATECHMTRAIN basictrain
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "Create CHM Train job." ]
}

#
# createCHMTrainJob.sh missing -l
#
@test "createCHMTrainJob.sh missing -l" {

  local curdir=`pwd`
  cd $THE_TMP 1>&2
  /bin/mkdir -p "$THE_TMP/images" 1>&2
  echo "hi" > "$THE_TMP/panfishCHM.properties"

  run $CREATECHMTRAIN basictrain "run" -i "$THE_TMP/images" 
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[2]}" == "ERROR:    Label images directory must be specified via -l flag" ]

  cd $curdir 1>&2
}

#
# createCHMTrainJob.sh missing -i
#
@test "createCHMTrainJob.sh missing -i" {

  local curdir=`pwd`
  cd $THE_TMP 1>&2
  /bin/mkdir -p "$THE_TMP/images" 1>&2
  /bin/mkdir -p "$THE_TMP/labels" 1>&2
  echo "hi" > "$THE_TMP/panfishCHM.properties"

  run $CREATECHMTRAIN basictrain "run" -l "$THE_TMP/labels"         
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[2]}" == "ERROR:    Images directory must be specified via -i flag" ]

  cd $curdir 1>&2
}


#
# createCHMTrainJob.sh invalid mode
#
@test "createCHMTrainJob.sh invalid mode" {

  local curdir=`pwd`
  cd $THE_TMP 1>&2
  /bin/mkdir -p "$THE_TMP/images" 1>&2
  /bin/mkdir -p "$THE_TMP/labels" 1>&2
  echo "hi" > "$THE_TMP/panfishCHM.properties"

  run $CREATECHMTRAIN badmode  "run" -l "$THE_TMP/labels" -i "$THE_TMP/images"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[2]}" == "ERROR:    badmode not supported.  Invoke /tmp/createCHMTrainJob/createCHMTrainJob.sh for list of valid options." ]

  cd $curdir 1>&2
}



#
# createCHMTrainJob.sh CHM bin dir missing
#
@test "createCHMTrainJob.sh CHM bin dir missing" {

  local curdir=`pwd`
  cd $THE_TMP 1>&2
  /bin/mkdir -p "$THE_TMP/images" 1>&2
  /bin/mkdir -p "$THE_TMP/labels" 1>&2
  echo "hi" > "$THE_TMP/panfishCHM.properties"

  run $CREATECHMTRAIN basictrain  "run" -l "$THE_TMP/labels" -i "$THE_TMP/images"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[2]}" == "ERROR:    CHM bin dir  does not exist" ]

  cd $curdir 1>&2
}

#
# crateCHMTrainJob.sh valid run
#
@test "createCHMTrainJob.sh valid run" {

  local curdir=`pwd`
  cd $THE_TMP 1>&2
  /bin/mkdir -p "$THE_TMP/images" 1>&2
  /bin/mkdir -p "$THE_TMP/labels" 1>&2
  echo "chm.bin.dir=$THE_TMP" > "$THE_TMP/panfishCHM.properties"

  run $CREATECHMTRAIN basictrain  "run" -l "$THE_TMP/labels" -i "$THE_TMP/images"
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  cd $curdir 1>&2
}

#
# createCHMTrainJob.sh valid run -S and -L set
#
@test "createCHMTrainJob.sh valid run -S and -L set" {

  local curdir=`pwd`
  cd $THE_TMP 1>&2
  /bin/mkdir -p "$THE_TMP/images" 1>&2
  /bin/mkdir -p "$THE_TMP/labels" 1>&2
  echo "chm.bin.dir=$THE_TMP" > "$THE_TMP/panfishCHM.properties"

  run $CREATECHMTRAIN basictrain  "run" -l "$THE_TMP/labels" -i "$THE_TMP/images" -S 6 -L 9
  echo "$output" 1>&2
  [ "$status" -eq 0 ]


  local outConfig="$THE_TMP/run/runCHMTrain.sh.config"
  [ -s "$outConfig" ]

  local aLine=`head -n 1 $outConfig`
  [ "$aLine" == "1${BATS_CONFIGDELIM}$THE_TMP/images" ]
  aLine=`head -n 2 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}$THE_TMP/labels" ]
  aLine=`head -n 3 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}-S 6 -L 9" ]
  aLine=`head -n 4 $outConfig | tail -n 1`
  [ "$aLine" == "1${BATS_CONFIGDELIM}${BATS_TEST_OUT_DIR_NAME}/$BATS_TRAINEDMODEL" ]

  cd $curdir 1>&2
}

