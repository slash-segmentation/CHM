#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/createCHMJob" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../panfishCHM.properties "$THE_TMP/." 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../*.sh "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts "$THE_TMP/." 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../../../algorithm "$THE_TMP/CHM" 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  export TESTIMAGE_DIR="${BATS_TEST_DIRNAME}/createchmjob"
  export TESTIMAGE="${TESTIMAGE_DIR}/600by400.png"
  export CREATECHM="$THE_TMP/createCHMJob.sh"
  chmod a+x $CREATECHM
  export HELPERFUNCS="$THE_TMP/.helperfuncs.sh"
  unset SGE_TASK_ID
  # should match OUT_DIR_NAME variable in .helperfuncs
  export BATS_TEST_OUT_DIR_NAME="runChmOut"

  # should match MERGE_TILES_OUT_DIR_NAME= variable in .helperfuncs
  export BATS_TEST_MERGE_TILES_OUT_DIR_NAME="runMergeTilesOut"
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  echo "teardown" 1>&2
}

#
#
#
@test "createMergeTilesConfig() tests" {
  skip "need to fix"
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test no directory
  run createMergeTilesConfig "$THE_TMP/doesnotexist"
  [ "$status" -eq 1 ]

  # test rm command fails
  RM_CMD="/bin/false"
  touch "$THE_TMP/$RUN_MERGE_TILES_CONFIG" 1>&2
  run createMergeTilesConfig "$THE_TMP"
  [ "$status" -eq 2 ]

  RM_CMD="/bin/rm"
  # test no image directories
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}" 1>&2

  run createMergeTilesConfig "$THE_TMP"
  [ "$status" -eq 0 ]
  [ ! -e "$THE_TMP/$RUN_MERGE_TILES_CONFIG" ]

  # test 1 image directory
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/hello.png.${IMAGE_TILE_DIR_SUFFIX}" 1>&2
  run createMergeTilesConfig "$THE_TMP"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/$RUN_MERGE_TILES_CONFIG" ]

  aLine=`head -n 1 $THE_TMP/$RUN_MERGE_TILES_CONFIG`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/${OUT_DIR_NAME}/hello.png.${IMAGE_TILE_DIR_SUFFIX}" ]
  aLine=`head -n 2 $THE_TMP/$RUN_MERGE_TILES_CONFIG | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${MERGE_TILES_OUT_DIR_NAME}/$MERGED_IMAGES_OUT_DIR_NAME/hello.png" ]


  # test 3 image directories
  /bin/rm -rf "$THE_TMP/${OUT_DIR_NAME}" 1>&2
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/hello1.png.${IMAGE_TILE_DIR_SUFFIX}" 1>&2
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/hello2.png.${IMAGE_TILE_DIR_SUFFIX}" 1>&2
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/hello3.png.${IMAGE_TILE_DIR_SUFFIX}" 1>&2
  run createMergeTilesConfig "$THE_TMP"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/$RUN_MERGE_TILES_CONFIG" ]

  aLine=`head -n 1 $THE_TMP/$RUN_MERGE_TILES_CONFIG`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/${OUT_DIR_NAME}/hello1.png.${IMAGE_TILE_DIR_SUFFIX}" ]

  aLine=`head -n 2 $THE_TMP/$RUN_MERGE_TILES_CONFIG | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${MERGE_TILES_OUT_DIR_NAME}/$MERGED_IMAGES_OUT_DIR_NAME/hello1.png" ]

  aLine=`head -n 3 $THE_TMP/$RUN_MERGE_TILES_CONFIG | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}$THE_TMP/${OUT_DIR_NAME}/hello2.png.${IMAGE_TILE_DIR_SUFFIX}" ]
  aLine=`head -n 4 $THE_TMP/$RUN_MERGE_TILES_CONFIG | tail -n 1`
   echo "$aLine" 1>&2
  [ "$aLine" == "2${CONFIG_DELIM}${MERGE_TILES_OUT_DIR_NAME}/$MERGED_IMAGES_OUT_DIR_NAME/hello2.png" ]

  aLine=`head -n 5 $THE_TMP/$RUN_MERGE_TILES_CONFIG | tail -n 1`
  [ "$aLine" == "3${CONFIG_DELIM}$THE_TMP/${OUT_DIR_NAME}/hello3.png.${IMAGE_TILE_DIR_SUFFIX}" ]

  aLine=`head -n 6 $THE_TMP/$RUN_MERGE_TILES_CONFIG | tail -n 1`
  [ "$aLine" == "3${CONFIG_DELIM}${MERGE_TILES_OUT_DIR_NAME}/$MERGED_IMAGES_OUT_DIR_NAME/hello3.png" ]
}



#
#
#
@test "createMergeTilesOutputDirectories() tests" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source .helperfuncs.sh 
  . $HELPERFUNCS


  # test where job directory does not exist
  run createMergeTilesOutputDirectories "$THE_TMP/doesnotexist"
  [ "$status" -eq 1 ]

  # test successful call
  run createMergeTilesOutputDirectories "$THE_TMP"
  [ "$status" -eq 0 ]

  [ -d "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$MERGED_IMAGES_OUT_DIR_NAME" ]

  [ -d "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_OUT_DIR_NAME" ]

  [ -d "$THE_TMP/$MERGE_TILES_OUT_DIR_NAME/$STD_ERR_DIR_NAME" ]

}

#
# createImageOutputDirectories() tests
#
@test "createImageOutputDirectories() tests" {

  # source createCHMJob.sh so we can call the function
  . $CREATECHM source

  # source .helperfuncs.sh 
  . $HELPERFUNCS


  # Test where output directory is not a directory
  run createImageOutputDirectories "$THE_TMP/doesnotexist" "$THE_TMP" "png"
  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "WARNING:    Output directory $THE_TMP/doesnotexist is not a directory" ]

  # Test where output directory is not a directory
  run createImageOutputDirectories "$THE_TMP" "$THE_TMP/doesnotexist" "png"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Image directory $THE_TMP/doesnotexist is not a directory" ]

  # Test where we have no images in directory
  run createImageOutputDirectories "$THE_TMP" "$THE_TMP" "png"
  [ "$status" -eq 0 ]

  # Test where we have 1 image in directory
  echo "test" > "$THE_TMP/foo.png"
  mkdir -p "$THE_TMP/out" 1>&2

  run createImageOutputDirectories "$THE_TMP/out" "$THE_TMP" "png"
  [ "$status" -eq 0 ]
  [ -d "$THE_TMP/out/foo.png.${IMAGE_TILE_DIR_SUFFIX}" ]


  # Test where we have 2 images in directory
  # Test where we have 1 image in directory
  echo "test" > "$THE_TMP/foo2.png"
  mkdir -p "$THE_TMP/out2" 1>&2
  run createImageOutputDirectories "$THE_TMP/out2" "$THE_TMP" "png"
  [ "$status" -eq 0 ]
  [ -d "$THE_TMP/out2/foo.png.${IMAGE_TILE_DIR_SUFFIX}" ]
  [ -d "$THE_TMP/out2/foo2.png.${IMAGE_TILE_DIR_SUFFIX}" ]
}

#
#
#
@test "calculateTilesFromImageDimensions() tests" {

  # source createCHMJob.sh so we can call the function
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # negative width
  run calculateTilesFromImageDimensions "-1" "10" "600" "400"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Width must be larger then 0" ]

  # zero width
  run calculateTilesFromImageDimensions "0" "10" "600" "400"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Width must be larger then 0" ]


  # negative height
  run calculateTilesFromImageDimensions "1" "-1" "600" "400"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Height must be larger then 0" ]


  # zero height 
  run calculateTilesFromImageDimensions "1" "0" "600" "400"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Height must be larger then 0" ]

  # negative block width
  run calculateTilesFromImageDimensions "1" "1" "-1" "400"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    BlockWidth must be larger then 0" ]

  # zero block width
  run calculateTilesFromImageDimensions "1" "1" "0" "400"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    BlockWidth must be larger then 0" ]

  # negative block height
  run calculateTilesFromImageDimensions "1" "1" "600" "-1"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    BlockHeight must be larger then 0" ]

  # zero block height 
  run calculateTilesFromImageDimensions "1" "1" "600" "0"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    BlockHeight must be larger then 0" ]

  # perfectly divisible blocks
  calculateTilesFromImageDimensions "600" "400" "200" "100"
  [ $? -eq 0 ]
  [ "$TILES_W" -eq 3 ]
  [ "$TILES_H" -eq 4 ]
  # one pixel off from perfectly divisible
  calculateTilesFromImageDimensions "600" "400" "199" "99"
  [ $? -eq 0 ]
  [ "$TILES_W" -eq 4 ]
  [ "$TILES_H" -eq 5 ]

  # one pixel off from perfectly divisible the other way
  calculateTilesFromImageDimensions "600" "400" "201" "101"
  [ $? -eq 0 ]
  [ "$TILES_W" -eq 3 ]
  [ "$TILES_H" -eq 4 ]


  # big image
  calculateTilesFromImageDimensions "15500" "12000" "875" "675"
  [ $? -eq 0 ]
  [ "$TILES_W" -eq 18 ]
  [ "$TILES_H" -eq 18 ]
}

#
# createCHMTestConfig() no images
#
@test "createCHMTestConfig() no images" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS


  run createCHMTestConfig "$THE_TMP" "$THE_TMP/foo" 1 1 1 " " "png"

  [ "$status" -eq 0 ]
  [ ! -e "$THE_TMP/foo" ]
}

#
# createCHMTestConfig() 1 image 1 tile
#
@test "createCHMTestConfig() 1 image 1 tile" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function  
  . $HELPERFUNCS


  echo "png" > "$THE_TMP"/1.png

  run createCHMTestConfig "$THE_TMP" "$THE_TMP/foo" 1 1 1 " " "modeldir" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 4 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "4" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}   -t 1,1" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/1.png" ]

}

#
# createCHMTestConfig() 1 image 1 tile and opts
#
@test "createCHMTestConfig() 1 image 1 tile and ops" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createCHMTestConfig "$THE_TMP" "$THE_TMP/foo" 1 1 1 "-o 1x1" "modeldir" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 4 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "4" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}-o 1x1  -t 1,1" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/1.png" ]

}

#
# createCHMTestConfig() 1 image 18 tiles and opts
#
@test "createCHMTestConfig() 1 image 18 tiles and ops" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createCHMTestConfig "$THE_TMP" "$THE_TMP/foo" 3 2 1 "-o 1x1" "modeldir" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  cat $THE_TMP/foo 1>&2

  # verify we have 24 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "24" ]
  # Check the lines
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}-o 1x1  -t 3,2" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/1.png" ]

  aLine=`head -n 5 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 6 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 7 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}-o 1x1  -t 3,1" ]

  aLine=`head -n 8 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/2.png" ]

  aLine=`head -n 9 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "3${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 10 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "3${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 11 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "3${CONFIG_DELIM}-o 1x1  -t 2,2" ]
  aLine=`head -n 12 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "3${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/3.png" ]

  aLine=`head -n 13 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "4${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 14 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "4${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 15 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "4${CONFIG_DELIM}-o 1x1  -t 2,1" ]

  aLine=`head -n 16 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "4${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/4.png" ]

  aLine=`head -n 17 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "5${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 18 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "5${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 19 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "5${CONFIG_DELIM}-o 1x1  -t 1,2" ]

  aLine=`head -n 20 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "5${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/5.png" ]

  aLine=`head -n 21 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "6${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 22 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "6${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 23 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "6${CONFIG_DELIM}-o 1x1  -t 1,1" ]

  aLine=`head -n 24 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "6${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/6.png" ]

}

#
# createCHMTestConfig() 1 image 6 tiles and opts and only 1 job
#
@test "createCHMTestConfig() 1 image 6 tiles and opts and only 1 job" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createCHMTestConfig "$THE_TMP" "$THE_TMP/foo" 2 3 6 "-o 1x1" "modeldir" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 4 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "4" ]
  cat $THE_TMP/foo 1>&2
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/1.png" ]

}

#
# createCHMTestConfig() 1 image 6 tiles and opts and 2 jobs
#
@test "createCHMTestConfig() 1 image 6 tiles and opts and 2 jobs" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createCHMTestConfig "$THE_TMP" "$THE_TMP/foo" 2 3 5 "-o 1x1" "modeldir" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 8 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "8" ]
  cat $THE_TMP/foo 1>&2
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/1.png" ]

  aLine=`head -n 5 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}$THE_TMP/1.png" ]

  aLine=`head -n 6 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 7 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}-o 1x1  -t 1,1" ]

  aLine=`head -n 8 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}${OUT_DIR_NAME}/1.png.${IMAGE_TILE_DIR_SUFFIX}/2.png" ]

}

#
# createCHMTestConfig() 2 cols 3 rows 7 tiles per job with 2 images and opts
#
@test "createCHMTestConfig() 2 cols 3 rows 7 tiles per job with 2 images and opts" {

  # source createCHMJob.sh with source flag to load functions
  . $CREATECHM source

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP/hist1.png"
  echo "png" > "$THE_TMP/hist2.png"

  run createCHMTestConfig "$THE_TMP" "$THE_TMP/foo" 2 3 7 "-o 1x1" "modeldir" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 8 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "8" ]
  cat $THE_TMP/foo 1>&2
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1${CONFIG_DELIM}$THE_TMP/hist1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1${CONFIG_DELIM}${OUT_DIR_NAME}/hist1.png.${IMAGE_TILE_DIR_SUFFIX}/1.png" ]

  aLine=`head -n 5 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}$THE_TMP/hist2.png" ]

  aLine=`head -n 6 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}modeldir" ]

  aLine=`head -n 7 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 8 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2${CONFIG_DELIM}${OUT_DIR_NAME}/hist2.png.${IMAGE_TILE_DIR_SUFFIX}/2.png" ]
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
  [ "${lines[2]}" == "  Creating directory $THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/stderr" ]

  # Verify runCHMviaPanfish.sh was copied over
  [ -x "$THE_TMP/run/runCHMViaPanfish.sh" ]

  # Verify runMergeTiles.sh was copied over
  [ -x "$THE_TMP/run/runMergeTiles.sh" ]

  # Verify runMergeTilesOut/mergedimages  directory was created
  [ -d "$THE_TMP/run/$BATS_TEST_MERGE_TILES_OUT_DIR_NAME/mergedimages" ]

  # Verify runMergeTilesOut/stdout  directory was created
  [ -d "$THE_TMP/run/$BATS_TEST_MERGE_TILES_OUT_DIR_NAME/stdout" ]

  # Verify runMergeTilesOut/stderr  directory was created
  [ -d "$THE_TMP/run/$BATS_TEST_MERGE_TILES_OUT_DIR_NAME/stderr" ]

  # Verify a merge tiles config is created 
  [ -s "$THE_TMP/run/runMergeTiles.sh.config" ] 

  # Verify panfishCHM.properties was copied over
  [ -s "$THE_TMP/run/panfishCHM.properties" ]

  # Verify out directory was created
  [ -d "$THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles" ]

  # Verify runCHM.sh was copied over and is executable
  [ -s "$THE_TMP/run/runCHM.sh" ]
  [ -x "$THE_TMP/run/runCHM.sh" ]

  # Verify helperfuncs.sh was copied over
  [ -s "$THE_TMP/run/.helperfuncs.sh" ]

  # Verify runCHM.sh.config was properly created
  [ -s "$THE_TMP/run/runCHM.sh.config" ]

  # verify we have 48 lines
  [ `wc -l "$THE_TMP/run/runCHM.sh.config" | sed "s/ .*//"` == "48" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/run/runCHM.sh.config"`
  [ "$aLine" == "1:::$TESTIMAGE" ]
  
  aLine=`head -n 2 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  echo "The line: $aLine" 1>&2
  [ "$aLine" == "1:::$THE_TMP/model" ]

  aLine=`head -n 3 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::   -b 200x100  -t 3,4" ]

  aLine=`head -n 4 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/1.png" ]

  aLine=`head -n 5 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$TESTIMAGE" ]

  aLine=`head -n 6 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$THE_TMP/model" ]

  aLine=`head -n 7 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::   -b 200x100  -t 3,3" ]

  aLine=`head -n 8 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/2.png" ]

  aLine=`head -n 45 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "12:::$TESTIMAGE" ]

  aLine=`head -n 46 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "12:::$THE_TMP/model" ]

  aLine=`head -n 47 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "12:::   -b 200x100  -t 1,1" ]

  aLine=`head -n 48 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "12:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/12.png" ]
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
  [ "${lines[2]}" == "  Creating directory $THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/stderr" ]

  # Verify runCHMviaPanfish.sh was copied over
  [ -x "$THE_TMP/run/runCHMViaPanfish.sh" ]

  # Verify panfishCHM.properties was copied over
  [ -s "$THE_TMP/run/panfishCHM.properties" ]

  # Verify out directory was created
  [ -d "$THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles" ]

  # Verify runCHM.sh was copied over and is executable
  [ -s "$THE_TMP/run/runCHM.sh" ]
  [ -x "$THE_TMP/run/runCHM.sh" ]

  # Verify helperfuncs.sh was copied over
  [ -s "$THE_TMP/run/.helperfuncs.sh" ]

  # Verify runCHM.sh.config was properly created
  [ -s "$THE_TMP/run/runCHM.sh.config" ]

  # verify we have 4 lines
  [ `wc -l "$THE_TMP/run/runCHM.sh.config" | sed "s/ .*//"` == "4" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/run/runCHM.sh.config"`
  [ "$aLine" == "1:::$TESTIMAGE" ]

  aLine=`head -n 2 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::$THE_TMP/model" ]

  aLine=`head -n 3 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::   -b 200x100  -t 3,4 -t 3,3 -t 3,2 -t 3,1 -t 2,4 -t 2,3 -t 2,2 -t 2,1 -t 1,4 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 4 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/1.png" ]

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


  run $CREATECHM createpretrained "$THE_TMP/run" -m "$THE_TMP/model" -i "$TESTIMAGE_DIR" -b 200x100 -T 19 -o 4x5
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == *createpretrained* ]]
  [ "${lines[2]}" == "  Creating directory $THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/stderr" ]

  # Verify runCHMviaPanfish.sh was copied over
  [ -x "$THE_TMP/run/runCHMViaPanfish.sh" ]

  # Verify panfishCHM.properties was copied over
  [ -s "$THE_TMP/run/panfishCHM.properties" ]

  # Verify out directory was created
  [ -d "$THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles" ]

  # Verify runCHM.sh was copied over and is executable
  [ -s "$THE_TMP/run/runCHM.sh" ]
  [ -x "$THE_TMP/run/runCHM.sh" ]

  # Verify helperfuncs.sh was copied over
  [ -s "$THE_TMP/run/.helperfuncs.sh" ]

  # Verify runCHM.sh.config was properly created
  [ -s "$THE_TMP/run/runCHM.sh.config" ]

  # verify we have 8 lines
  [ `wc -l "$THE_TMP/run/runCHM.sh.config" | sed "s/ .*//"` == "8" ]
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/run/runCHM.sh.config"`
  [ "$aLine" == "1:::$TESTIMAGE" ]

  aLine=`head -n 2 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::$THE_TMP/model" ]

  aLine=`head -n 3 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::  -o 4x5  -b 200x100  -t 4,5 -t 4,4 -t 4,3 -t 4,2 -t 4,1 -t 3,5 -t 3,4 -t 3,3 -t 3,2 -t 3,1 -t 2,5 -t 2,4 -t 2,3 -t 2,2 -t 2,1 -t 1,5 -t 1,4 -t 1,3 -t 1,2" ]

  aLine=`head -n 4 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/1.png" ]

  aLine=`head -n 5 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$TESTIMAGE" ]

  aLine=`head -n 6 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$THE_TMP/model" ]

  aLine=`head -n 7 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::  -o 4x5  -b 200x100  -t 1,1" ]

  aLine=`head -n 8 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/2.png" ]

}


# createpretrained with relative path set for model
# created properly
#
@test "createpretrained with relative path set for model" {
  # create a fake model folder
  mkdir -p "$THE_TMP/model"
  echo "fake" > "$THE_TMP/model/param.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level0_stage2.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level1_stage1.mat"
  echo "fake" > "$THE_TMP/model/MODEL_level2_stage1.mat"

  curdir=`pwd`
  cd $THE_TMP
  run $CREATECHM createpretrained "$THE_TMP/run" -m "model" -i "$TESTIMAGE_DIR" -b 200x100 -T 19 -o 4x5
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == *createpretrained* ]]
  [ "${lines[2]}" == "  Creating directory $THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/stderr" ]

  cd $curdir

  # Verify runCHMviaPanfish.sh was copied over
  [ -x "$THE_TMP/run/runCHMViaPanfish.sh" ]

  # Verify panfishCHM.properties was copied over
  [ -s "$THE_TMP/run/panfishCHM.properties" ]

  # Verify out directory was created
  [ -d "$THE_TMP/run/${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles" ]

  # Verify runCHM.sh was copied over and is executable
  [ -s "$THE_TMP/run/runCHM.sh" ]
  [ -x "$THE_TMP/run/runCHM.sh" ]

  # Verify helperfuncs.sh was copied over
  [ -s "$THE_TMP/run/.helperfuncs.sh" ]

  # Verify runCHM.sh.config was properly created
  [ -s "$THE_TMP/run/runCHM.sh.config" ]

  # verify we have 8 lines
  [ `wc -l "$THE_TMP/run/runCHM.sh.config" | sed "s/ .*//"` == "8" ]
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/run/runCHM.sh.config"`
  [ "$aLine" == "1:::$TESTIMAGE" ]

  aLine=`head -n 2 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::$THE_TMP/model" ]

  aLine=`head -n 3 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::  -o 4x5  -b 200x100  -t 4,5 -t 4,4 -t 4,3 -t 4,2 -t 4,1 -t 3,5 -t 3,4 -t 3,3 -t 3,2 -t 3,1 -t 2,5 -t 2,4 -t 2,3 -t 2,2 -t 2,1 -t 1,5 -t 1,4 -t 1,3 -t 1,2" ]

  aLine=`head -n 4 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "1:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/1.png" ]

  aLine=`head -n 5 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$TESTIMAGE" ]

  aLine=`head -n 6 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::$THE_TMP/model" ]

  aLine=`head -n 7 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::  -o 4x5  -b 200x100  -t 1,1" ]

  aLine=`head -n 8 "$THE_TMP/run/runCHM.sh.config" | tail -n 1`
  [ "$aLine" == "2:::${BATS_TEST_OUT_DIR_NAME}/600by400.png.tiles/2.png" ]

}
