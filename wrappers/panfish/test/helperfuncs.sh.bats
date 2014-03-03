#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/helperfuncs" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh "$THE_TMP/." 1>&2
  export TESTIMAGE="${BATS_TEST_DIRNAME}/createchmjob/600by400.png" 
  export TESTIMAGE_DIR="${BATS_TEST_DIRNAME}/createchmjob"
  export HELPERFUNCS="$THE_TMP/.helperfuncs.sh"
  export PANFISH_TEST_BIN="${BATS_TEST_DIRNAME}/bin/panfish"
  unset SGE_TASK_ID
}

teardown(){
  /bin/rm -rf "$THE_TMP"
  echo "end of tear down" 1>&2
}


#
# parseProperties() tests
#
@test "parseProperties() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test where no properties file is found
  run parseProperties "$THE_TMP" "$THE_TMP/s"

  [ "$status" -eq 1 ]
  echo "$output" 1>&2
  [ "${lines[0]}" == "  Config $THE_TMP/panfishCHM.properties not found" ]


  echo "panfish.bin.dir=/xx" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=/matlab" >> "$THE_TMP/panfishCHM.properties"
  echo "batch.and.walltime.args=batch" >> "$THE_TMP/panfishCHM.properties"
  echo "cluster.list=foo.q" >> "$THE_TMP/panfishCHM.properties"
  echo "chm.bin.dir=/bin/chm" >> "$THE_TMP/panfishCHM.properties"
  echo "max.retries=3" >> "$THE_TMP/panfishCHM.properties"
  echo "retry.sleep=10" >> "$THE_TMP/panfishCHM.properties"
  echo "job.wait.sleep=20" >> "$THE_TMP/panfishCHM.properties"
  echo "land.job.options=ljo" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.job.options=cjo" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.image.options=-b --deletebefore -x *.*" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.model.options=--exclude *.foo" >> "$THE_TMP/panfishCHM.properties"
  # test with valid complete config
  parseProperties "$THE_TMP" "$THE_TMP/s"
  
  [ "$PANFISH_BIN_DIR" == "/xx" ]
  [ "$MATLAB_DIR" == "/matlab" ]
  [ "$BATCH_AND_WALLTIME_ARGS" == "batch" ]
  [ "$CHUMMEDLIST" == "foo.q" ]
  [ "$CHM_BIN_DIR" == "/bin/chm" ]
  [ "$MAX_RETRIES" == "3" ]
  [ "$RETRY_SLEEP" == "10" ]
  [ "$WAIT_SLEEP_TIME" == "20" ]

  [ "$LAND_JOB_OPTS" == "ljo" ]
  [ "$CHUM_JOB_OPTS" == "cjo" ]
  [ "$CHUM_IMAGE_OPTS" == "-b --deletebefore -x *.*" ]
  [ "$CHUM_MODEL_OPTS" == "--exclude *.foo" ]

  [ "$CASTBINARY" == "/xxpanfishcast" ]
  [ "$CHUMBINARY" == "/xxpanfishchum" ]
  [ "$LANDBINARY" == "/xxpanfishland" ]
  [ "$PANFISHSTATBINARY" == "/xxpanfishstat" ]
}

# 
# getSingleCHMTestTaskStdOutFile() tests
#
@test "getSingleCHMTestTaskStdOutFile() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where job path is not a directory
  run getSingleCHMTestTaskStdOutFile "$THE_TMP/doesnotexist" "1"
#  echo "$output" 1>&2
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
# getSingleCHMTestTaskStdErrFile() tests
#
@test "getSingleCHMTestTaskStdErrFile() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where job path is not a directory
  run getSingleCHMTestTaskStdErrFile "$THE_TMP/doesnotexist" "1"
#  echo "$output" 1>&2
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
@test "checkSingleCHMTestTask() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stdout"
  # Test where stdout file does not exist
  run checkSingleCHMTestTask "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  # Test where stdout file is zero size
  touch "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"
  run checkSingleCHMTestTask "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  echo "1${CONFIG_DELIM}xx" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}yy" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}zz" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}${OUT_DIR_NAME}/uh.png" >> "$THE_TMP/runCHM.sh.config"

  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"
  # Test where no output image is found
  run checkSingleCHMTestTask "$THE_TMP" "1"
  echo "$output" 1>&2
  [ "$status" -eq 3 ]

  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/uh.png"  
  # Test where we are all good
  run checkSingleCHMTestTask "$THE_TMP" "1"
  [ "$status" -eq 0 ]
}

#
# parseProperties() tests
#
@test "parseProperties() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test where no properties file is found
  run parseProperties "$THE_TMP" "$THE_TMP/s"

  [ "$status" -eq 1 ]
  echo "${lines[0]}" 1>&2
  [ "${lines[0]}" == "  Config $THE_TMP/panfishCHM.properties not found" ]


  echo "panfish.bin.dir=/xx" > "$THE_TMP/panfishCHM.properties"
  echo "matlab.dir=/matlab" >> "$THE_TMP/panfishCHM.properties"
  echo "batch.and.walltime.args=batch" >> "$THE_TMP/panfishCHM.properties"
  echo "cluster.list=foo.q" >> "$THE_TMP/panfishCHM.properties"
  echo "chm.bin.dir=/bin/chm" >> "$THE_TMP/panfishCHM.properties"
  echo "max.retries=3" >> "$THE_TMP/panfishCHM.properties"
  echo "retry.sleep=10" >> "$THE_TMP/panfishCHM.properties"
  echo "job.wait.sleep=20" >> "$THE_TMP/panfishCHM.properties"
  echo "land.job.options=ljo" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.job.options=cjo" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.image.options=-b --deletebefore -x *.*" >> "$THE_TMP/panfishCHM.properties"
  echo "chum.model.options=--exclude *.foo" >> "$THE_TMP/panfishCHM.properties"
  # test with valid complete config
  parseProperties "$THE_TMP" "$THE_TMP/s"
  
  [ "$PANFISH_BIN_DIR" == "/xx" ]
  [ "$MATLAB_DIR" == "/matlab" ]
  [ "$BATCH_AND_WALLTIME_ARGS" == "batch" ]
  [ "$CHUMMEDLIST" == "foo.q" ]
  [ "$CHM_BIN_DIR" == "/bin/chm" ]
  [ "$MAX_RETRIES" == "3" ]
  [ "$RETRY_SLEEP" == "10" ]
  [ "$WAIT_SLEEP_TIME" == "20" ]

  [ "$LAND_JOB_OPTS" == "ljo" ]
  [ "$CHUM_JOB_OPTS" == "cjo" ]
  [ "$CHUM_IMAGE_OPTS" == "-b --deletebefore -x *.*" ]
  [ "$CHUM_MODEL_OPTS" == "--exclude *.foo" ]

  [ "$CASTBINARY" == "/xxpanfishcast" ]
  [ "$CHUMBINARY" == "/xxpanfishchum" ]
  [ "$LANDBINARY" == "/xxpanfishland" ]
  [ "$PANFISHSTATBINARY" == "/xxpanfishstat" ]
}

# 
# getSingleCHMTestTaskStdOutFile() tests
#
@test "getSingleCHMTestTaskStdOutFile() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where job path is not a directory
  run getSingleCHMTestTaskStdOutFile "$THE_TMP/doesnotexist" "1"
#  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP/doesnotexist is not a directory" ]

  # Test where log file does not exist
  getSingleCHMTestTaskStdOutFile "$THE_TMP" "1"
  [ "$?" -eq 0 ] 
  [ "$CHM_STD_OUT_FILE" == "$THE_TMP/out/stdout/1.stdout" ]

  # Test with valid log file
  mkdir -p "$THE_TMP/out/stdout"
  echo "blah" > "$THE_TMP/out/stdout/4.stdout"
  getSingleCHMTestTaskStdOutFile "$THE_TMP" "4"
  [ "$?" == 0 ] 
  [ "$CHM_STD_OUT_FILE" == "$THE_TMP/out/stdout/4.stdout" ]

}

# 
# getSingleCHMTestTaskStdErrFile() tests
#
@test "getSingleCHMTestTaskStdErrFile() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where job path is not a directory
  run getSingleCHMTestTaskStdErrFile "$THE_TMP/doesnotexist" "1"
#  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP/doesnotexist is not a directory" ]

  # Test where log file does not exist
  getSingleCHMTestTaskStdErrFile "$THE_TMP" "1"
  [ $? -eq 0 ]
  [ "$CHM_STD_ERR_FILE" == "$THE_TMP/out/stderr/1.stderr" ]

  # Test with valid log file
  mkdir -p "$THE_TMP/out/stderr"
  echo "blah" > "$THE_TMP/out/stderr/24.stderr"
  getSingleCHMTestTaskStdErrFile "$THE_TMP" "24"
  [ "$?" == 0 ]     
  [ "$CHM_STD_ERR_FILE" == "$THE_TMP/out/stderr/24.stderr" ]

}

#
# checkSingleTask() tests
#
@test "checkSingleCHMTestTask() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  mkdir -p "$THE_TMP/out/stdout"
  # Test where stdout file does not exist
  run checkSingleCHMTestTask "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  # Test where stdout file is zero size
  touch "$THE_TMP/out/stdout/1.stdout"
  run checkSingleCHMTestTask "$THE_TMP" "1"
  [ "$status" -eq 2 ]

  echo "1:::xx" > "$THE_TMP/runCHM.sh.config"
  echo "1:::yy" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::zz" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::out/uh.png" >> "$THE_TMP/runCHM.sh.config"

  echo "blah" > "$THE_TMP/out/stdout/1.stdout"
  # Test where no output image is found
  run checkSingleCHMTestTask "$THE_TMP" "1"
  echo "$output" 1>&2
  [ "$status" -eq 3 ]

  echo "hi" > "$THE_TMP/out/uh.png"  
  # Test where we are all good
  run checkSingleCHMTestTask "$THE_TMP" "1"
  [ "$status" -eq 0 ]
}

#
# parseWidthHeightParameter() with various values
#
@test "parseWidthHeightParameter() with various values" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  parseWidthHeightParameter "200x100"
 
  [ $? -eq 0 ] 
  [ "$PARSED_WIDTH" -eq 200 ]
  [ "$PARSED_HEIGHT" -eq 100 ]

  parseWidthHeightParameter "20"

  [ $? -eq 0 ]
  [ "$PARSED_WIDTH" -eq 20 ]
  [ "$PARSED_HEIGHT" -eq 20 ]

}

# 
# getImageDimensions() tests
#
@test "getImageDimensions() tests" {
  run which identify

  if [ "$status" -ne 0 ] ; then
    skip "No Image Magick identify command found"
  fi 

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  getImageDimensions "$TESTIMAGE"
  [ $? -eq 0 ]
  [ "$PARSED_WIDTH" -eq 600 ]
  [ "$PARSED_HEIGHT" -eq 400 ]

  # try on file that does not exist
  run getImageDimensions "$THE_TMP"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP is not a file" ]

  # try on non image file
  run getImageDimensions "$HELPERFUNCS"

  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == "WARNING:    Unable to parse block parameter:"* ]]

}

@test "getImageDimensionsFromDirOfImages() tests" {
  run which identify

  if [ "$status" -ne 0 ] ; then
    skip "No Image Magick identify command found"
  fi

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  run getImageDimensionsFromDirOfImages "$TESTIMAGE" "png"
  
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $TESTIMAGE is not a directory" ]

  # try on file that does not exist
  run getImageDimensionsFromDirOfImages "$THE_TMP/doesnotexist" "png"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    $THE_TMP/doesnotexist is not a directory" ]

  mkdir -p "$THE_TMP/emptydir" 1>&2 

  # try on directory with no images
  run getImageDimensionsFromDirOfImages "$THE_TMP/emptydir" "png"

  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    No images found in $THE_TMP/emptydir with suffix png" ]


  getImageDimensionsFromDirOfImages "$TESTIMAGE_DIR" "png"

  [ $? -eq 0 ] 
  [ "$PARSED_WIDTH" -eq 600 ]
  [ "$PARSED_HEIGHT" -eq 400 ]
}

#
# calculateTilesFromImageDimensions() tests
#
@test "calculateTilesFromImageDimensions() tests" {
  
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
# createImageOutputDirectories() tests
#
@test "createImageOutputDirectories() tests" {
    
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where output directory is not a directory
  run createImageOutputDirectories "$THE_TMP/doesnotexist" "$THE_TMP" "png"
  [ "$status" -eq 1 ] 
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
# verifyCHMTestResults() tests
#
@test "verifyCHMTestResults() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test with 1 job successful no file
  echo "1${CONFIG_DELIM}xx" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}yy" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}zz" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}${OUT_DIR_NAME}/blah/1.png" >> "$THE_TMP/runCHM.sh.config"


  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stdout"
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/blah"
  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/blah/1.png"
   # Test with 1 job successful no file
  verifyCHMTestResults "1" "$THE_TMP" "1" "1" "no"
  [ $? -eq 0 ]
  [ "$NUM_FAILED_JOBS" -eq 0 ]

  # Test with 1 job successful with file
  verifyCHMTestResults "1" "$THE_TMP" "1" "1" "yes"
  [ $? -eq 0 ]
  [ "$NUM_FAILED_JOBS" -eq 0 ]
  [ ! -e "$THE_TMP/failed.jobs" ] 


  # Test with 1 job failure no file
  /bin/rm -f "$THE_TMP/${OUT_DIR_NAME}/blah/1.png"
  run verifyCHMTestResults "1" "$THE_TMP" "1" "1" "no"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    No output image found for task 1 $THE_TMP/${OUT_DIR_NAME}/blah/1.png" ]
  [ ! -e "$THE_TMP/failed.jobs" ]

  echo "hello" > "$THE_TMP/failed.jobs"
  # Test with 1 job failure with file and failed file exists 
  run verifyCHMTestResults "1" "$THE_TMP" "1" "1" "yes"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    No output image found for task 1 $THE_TMP/${OUT_DIR_NAME}/blah/1.png" ]
  [ -s "$THE_TMP/failed.jobs" ]
  [ -s "$THE_TMP/failed.0.jobs" ]

  aLine=`head -n 1 "$THE_TMP/failed.jobs"` 
  [ "$aLine" == "1" ]  


  /bin/rm -f "$THE_TMP/failed.jobs"
  /bin/rm -f "$THE_TMP/failed.0.jobs"
  echo "2${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "2${CONFIG_DELIM}yy" >> "$THE_TMP/runCHM.sh.config"
  echo "2${CONFIG_DELIM}zz" >> "$THE_TMP/runCHM.sh.config"
  echo "2${CONFIG_DELIM}${OUT_DIR_NAME}/blah/2.png" >> "$THE_TMP/runCHM.sh.config"
  echo "3${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "3${CONFIG_DELIM}yy" >> "$THE_TMP/runCHM.sh.config"
  echo "3${CONFIG_DELIM}zz" >> "$THE_TMP/runCHM.sh.config"
  echo "3${CONFIG_DELIM}${OUT_DIR_NAME}/blah/3.png" >> "$THE_TMP/runCHM.sh.config"

    echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/blah/1.png"

  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/2.stdout"
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/blah/2.png"
  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/3.stdout"
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/blah/3.png"
  # Test with 3 jobs all successful no file
  verifyCHMTestResults "1" "$THE_TMP" "1" "3" "no"
  [ $? -eq 0 ]
  [ "$NUM_FAILED_JOBS" -eq 0 ]
  [ ! -e "$THE_TMP/failed.jobs" ]


  # Test with 3 jobs all successful with file
  run verifyCHMTestResults "1" "$THE_TMP" "1" "3" "yes"
  [ "$status" -eq 0 ]
  [ "$NUM_FAILED_JOBS" -eq 0 ]
  [ ! -e "$THE_TMP/failed.jobs" ]

  # Test with 5 jobs 2 failed and 3 successful no file
  /bin/rm -f "$THE_TMP/${OUT_DIR_NAME}/blah/1.png"

  echo "4${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "4${CONFIG_DELIM}yy" >> "$THE_TMP/runCHM.sh.config"
  echo "4${CONFIG_DELIM}zz" >> "$THE_TMP/runCHM.sh.config"
  echo "4${CONFIG_DELIM}${OUT_DIR_NAME}/blah/4.png" >> "$THE_TMP/runCHM.sh.config"
  echo "5${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "5${CONFIG_DELIM}yy" >> "$THE_TMP/runCHM.sh.config"
  echo "5${CONFIG_DELIM}zz" >> "$THE_TMP/runCHM.sh.config"
  echo "5${CONFIG_DELIM}${OUT_DIR_NAME}/blah/5.png" >> "$THE_TMP/runCHM.sh.config"

  echo "blah" > "$THE_TMP/${OUT_DIR_NAME}/stdout/4.stdout"
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/blah/4.png"
  run verifyCHMTestResults "1" "$THE_TMP" "1" "5" "no"
  [ "$status" -eq 1 ]
  [ ! -e "$THE_TMP/failed.jobs" ]



  # Test with 5 jobs 2 failed and 3 successful with file
  run verifyCHMTestResults "2" "$THE_TMP" "1" "5" "yes"
  [ "$status" -eq 1 ]
  [ -s "$THE_TMP/failed.jobs" ]
  aLine=`head -n 1 $THE_TMP/failed.jobs`
  [ "$aLine" == "1" ]
  aLine=`head -n 2 $THE_TMP/failed.jobs | tail -n 1`
  [ "$aLine" == "5" ]

}


#
# getSizeOfPath() tests
#
@test "getSizeOfPath() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test non existant file
  run getSizeOfPath "$THE_TMP/doesnotexist"
  [ "$status" -eq 1 ]

  # Test on file
  echo "hi" > "$THE_TMP/hi"
  getSizeOfPath "$THE_TMP/hi"
 
  [ $? -eq 0 ]
  [ "$NUM_BYTES" -ge 3 ]

  getSizeOfPath "$THE_TMP"

  [ $? -eq 0 ]
  [ "$NUM_BYTES" -ge 20000 ]

}

#
# getFullPath() tests
#
@test "getFullPath() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  
  # Test full path on path that is already full
  getFullPath "$THE_TMP"
  
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP" ]

  curdir=`pwd`
  cd $THE_TMP
  
  # try .
  getFullPath "."
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP" ]

  # try relative path
  mkdir -p "$THE_TMP/foo"
  getFullPath "foo"
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP/foo" ]

  # try another relative path
  getFullPath "foo/.."
  [ "$?" -eq 0 ]
  [ "$GETFULLPATHRET" == "$THE_TMP" ]

  cd $curdir

}

#
# moveOldClusterFolders() tests
#
@test "moveOldClusterFolders() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where CHUMMEDLIST is empty
  export CHUMMEDLIST=""
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]

  # Test with CHUMMEDLIST not empty (one entry) but no folders exist
  export CHUMMEDLIST="foo_shadow.q"
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]

  # Test with CHUMMEDLIST not empty (two entries) but no folders exist
  export CHUMMEDLIST="foo_shadow.q,blah_shadow.q"
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]

  
  # Test with CHUMMEDLIST not empty (one entry) and folders exist
  export CHUMMEDLIST="foo_shadow.q"
  mkdir -p "$THE_TMP/foo_shadow.q" 1>&2
  run moveOldClusterFolders "1" "$THE_TMP"
  [ "$status" -eq 0 ]
  [ ! -d "$THE_TMP/foo_shadow.q" ]
  [ -d "$THE_TMP/foo_shadow.q.1.old" ] 


  # Test with CHUMMEDLIST not empty (two entries) and folders exist
  export CHUMMEDLIST="foo_shadow.q,blah_shadow.q"
  mkdir -p "$THE_TMP/foo_shadow.q" 1>&2
  mkdir -p "$THE_TMP/blah_shadow.q" 1>&2
  run moveOldClusterFolders "3" "$THE_TMP"
  [ "$status" -eq 0 ]
  echo "$output" 1>&2
  ls -la "$THE_TMP" 1>&2
  [ ! -d "$THE_TMP/foo_shadow.q" ]
  [ -d "$THE_TMP/foo_shadow.q.3.old" ] 
  [ ! -d "$THE_TMP/blah_shadow.q" ]
  [ -d "$THE_TMP/blah_shadow.q.3.old" ]
 
  # Test where destination foo_shadow.q.3.old already exists
  mkdir -p "$THE_TMP/foo_shadow.q" 1>&2
  run moveOldClusterFolders "3" "$THE_TMP"
  [ "$status" -eq 0 ]
  echo "$output" 1>&2
  ls -la "$THE_TMP" 1>&2
  [ ! -d "$THE_TMP/foo_shadow.q" ]
  [ -d "$THE_TMP/foo_shadow.q.3.old" ]
  [ ! -d "$THE_TMP/blah_shadow.q" ]
  [ -d "$THE_TMP/blah_shadow.q.3.old" ]
}

#
# checkForKillFile() tests
#
@test "checkForKillFile() tests" {
    
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # test where there is no kill file
  run checkForKillFile "$THE_TMP"
  [ "$status" -eq 0 ]

  export CHUMBINARY="echo"
  export CHUMMEDLIST="foo.q"
  touch "$THE_TMP/KILL.JOB.REQUEST"
  # test where there is a kill file
  run checkForKillFile "$THE_TMP"

  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "  $THE_TMP/KILL.JOB.REQUEST detected.  Chumming kill file to remote cluster(s) and exiting..." ]
  [ "${lines[1]}" == "  Running echo --path $THE_TMP/KILL.JOB.REQUEST --cluster foo.q > $THE_TMP/killed.chum.out" ]

  [ -s "$THE_TMP/killed.chum.out" ]
  aLine=`head -n 1 "$THE_TMP/killed.chum.out"`
  [ "$aLine" == "--path /tmp/helperfuncs/KILL.JOB.REQUEST --cluster foo.q" ]

  # test where there is a kill file and we request to just have code return instead of exiting
  touch "$THE_TMP/KILL.JOB.REQUEST"
  run checkForKillFile "$THE_TMP" "dontexit"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "  $THE_TMP/KILL.JOB.REQUEST detected.  Chumming kill file to remote cluster(s)" ]
  [ "${lines[1]}" == "  Running echo --path $THE_TMP/KILL.JOB.REQUEST --cluster foo.q > $THE_TMP/killed.chum.out" ]
  
}

# 
# chumData() tests
#
@test "chumData() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2

  
  # Test where cluster list is empty
  run chumData "" "/foo" "$THE_TMP/yikes" ""
  [ "$status" -eq 1 ] 
  [ "${lines[0]}" == "WARNING:    No clusters in cluster list" ]


  export CHUMBINARY="echo"
  # Test where chum succeeds
  run chumData "foo.q" "/foo" "$THE_TMP/yikes" ""
  [ "$status" -eq 0 ]
  [ -s "$THE_TMP/yikes" ]
  aLine=`head -n 1 $THE_TMP/yikes`
  [ "$aLine" == "--listchummed --path /foo --cluster foo.q" ]

  # Test where chum succeeds and has other args
  run chumData "foo.q" "/foo" "$THE_TMP/yikes" "--exclude foo --exclude *.gee"
  [ "$status" -eq 0 ]
  [ -s "$THE_TMP/yikes" ]
  aLine=`head -n 1 $THE_TMP/yikes`
  [ "$aLine" == "--listchummed --path /foo --cluster foo.q --exclude foo --exclude *.gee" ]


  # Test where chum fails
  export CHUMBINARY="/bin/false"
  run chumData "foo.q" "/foo" "$THE_TMP/yikes" ""
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Chum of /foo failed" ]

  # test where chum succeeds
  export CHUMBINARY="$THE_TMP/panfish/panfishchum"
  echo "0,chummed.clusters=wow,," > "$THE_TMP/panfish/panfishchum.tasks"
  chumData "foo.q" "/foo" "$THE_TMP/yikes" ""
  [ $? == 0 ]
  [ "$CHUMMEDLIST" == "wow" ]
  
}

#
# landData() tests
#
@test "landData() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  
  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2

  # Test where cluster list is empty
  run landData "" "/foo" ""
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    No clusters in cluster list" ]

  # Test where land succeeds
  export LANDBINARY="echo"
  run landData "foo" "/foo" "" "1" "0"
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "--path /foo --cluster foo" ]

  # Test where land fails
  export LANDBINARY="/bin/false"
  run landData "foo" "/foo" "" "1" "0"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Download attempt # 1 of 2 of /foo failed.  Sleeping 0 seconds" ]
  [ "${lines[1]}" == "WARNING:    Download attempt # 2 of 2 of /foo failed.  Sleeping 0 seconds" ]

  # Test where first try fails and second succeeds
  export LANDBINARY="$THE_TMP/panfish/panfishland"
  echo "1,error,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,,," >> "$THE_TMP/panfish/panfishland.tasks"
  run landData "foo" "/foo" "" "1" "0"
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "error" ]
  [ "${lines[1]}" == "WARNING:    Download attempt # 1 of 2 of /foo failed.  Sleeping 0 seconds" ]

}

#
# getStatusOfJobsInCastOutFile() tests
# 
@test "getStatusOfJobsInCastOutFile() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2

  # Test with no cast.out file
  run getStatusOfJobsInCastOutFile "$THE_TMP" "cast.out"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    No $THE_TMP/cast.out file found" ]
 
  # Test where panfishstat binary outputs error 
  export PANFISHSTATBINARY="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run getStatusOfJobsInCastOutFile "$THE_TMP" "cast.out"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Error calling /bin/false --statusofjobs $THE_TMP/cast.out" ]

  # Test with successful panfishstat run
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  echo "0,status=hello,," > "$THE_TMP/panfish/panfishstat.tasks"  
  getStatusOfJobsInCastOutFile "$THE_TMP" "cast.out"
  [ "$?" -eq 0 ]
  [ "$JOBSTATUS" == "hello" ]
}

#
# waitForJobs() tests
#
@test "waitForJobs() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export WAIT_SLEEP_TIME=0
  export CHUMMEDLIST="foo.q"
  echo "hello" > "$THE_TMP/cast.out"

  # Test where job finishes 1st time
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  run waitForJobs "3" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 3 Start Time: "* ]]
  [ "${lines[1]}" == "  Iteration 3 job status is NA.  Sleeping 0 seconds" ]
  [[ "${lines[2]}" == " WaitForJobs in cast.out Iteration 3 End Time: "* ]]
  

  # Test where getStatus errors 1st time and works second going to run state and finishes 3rd time
  echo "1,status=uh,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
   
  run waitForJobs "2" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 2 Start Time: "* ]]
  [ "${lines[1]}" == "  Iteration 2 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[2]}" == "WARNING:    Error calling $PANFISHSTATBINARY --statusofjobs $THE_TMP/cast.out" ]
  [ "${lines[3]}" == "  Iteration 2 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[4]}" == "  Iteration 2 job status is running.  Sleeping 0 seconds" ]
  [[ "${lines[5]}" == " WaitForJobs in cast.out Iteration 2 End Time: "* ]]

  
  # Test where a kill file is found
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  export CHUMBINARY="/bin/true"
  echo "die" > "$THE_TMP/KILL.JOB.REQUEST"
  run waitForJobs "3" "$THE_TMP" "cast.out"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 3 Start Time: "* ]]
  [ "${lines[1]}" == "  $THE_TMP/KILL.JOB.REQUEST detected.  Chumming kill file to remote cluster(s) and exiting..." ]
  [ "${lines[2]}" == "  Running /bin/true --path $THE_TMP/KILL.JOB.REQUEST --cluster foo.q > $THE_TMP/killed.chum.out" ]

  # Test where DOWNLOAD.DATA.REQUEST file shows up and job then finishes after
  echo "yo" > "$THE_TMP/chm.cast.out"
  export LANDBINARY="/bin/echo"
  export LAND_JOB_OPTS="--exclude hi.* --exclude bye.*" 
  echo "0,status=running,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,,touch $THE_TMP/DOWNLOAD.DATA.REQUEST" >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"  
  /bin/rm -f $THE_TMP/KILL.JOB.REQUEST

  run waitForJobs "3" "$THE_TMP" "chm.cast.out"
  [ "$status" -eq 0 ]
  echo "$output" 1>&2
  [[ "${lines[0]}" == " WaitForJobs in chm.cast.out Iteration 3 Start Time: "* ]]
  [ "${lines[1]}" == "  Iteration 3 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[2]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [ "${lines[3]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [ "${lines[4]}" == "  DOWNLOAD.DATA.REQUEST file found.  Performing download" ]
  [ "${lines[5]}" == "--path $THE_TMP --cluster foo.q --exclude hi.* --exclude bye.*" ]
  [ "${lines[6]}" == "  Removing DOWNLOAD.DATA.REQUEST file" ]
  [ "${lines[7]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [[ "${lines[8]}" == " WaitForJobs in chm.cast.out Iteration 3 End Time: "* ]] 
  [ ! -e "$THE_TMP/DOWNLOAD.DATA.REQUEST" ]

}

#
# moveCastOutFile() tests
#
@test "moveCastOutFile() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where there isn't a cast.out file
  run moveCastOutFile "1" "$THE_TMP" "hi"
  [ "$status" -eq 0 ]

  # Test where there is a cast.out file
  echo "hi" > "$THE_TMP/cast.out"
  run moveCastOutFile "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/cast.out.1.out" ]
  [ ! -e "$THE_TMP/cast.out" ]

  # Test where there is a cast.1.out file
  echo "bye" > "$THE_TMP/cast.out"
  run moveCastOutFile "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/cast.out.1.out" ]
  [ ! -e "$THE_TMP/cast.out" ]
  aLine=`head -n 1 "$THE_TMP/cast.out.1.out"`
  [ "$aLine" == "bye" ]

  # Test where mv command fails
  export MV_CMD="/bin/false"
  echo "yo" > "$THE_TMP/cast.out" "cast.out"
  run moveCastOutFile "1" "$THE_TMP"
  [ "$status" -eq 1 ]
}

#
# moveOldDataForNewIteration() tests
#
@test "moveOldDataForNewIteration() tests" {
    
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
 
  # test where CHUMMEDLIST is empty and no cast.out file
  unset CHUMMEDLIST
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
 
  # test where there is an error moving the old cluster folders
  export MV_CMD="/bin/false"
  export CHUMMEDLIST="foo.q"
  mkdir -p "$THE_TMP/foo.q" 1>&2
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cluster folders for previous iteration 0" ]

  # test where there is an error moving cluster folders and moving cast.out file
  export MV_CMD="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 3 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cluster folders for previous iteration 0" ]
  [ "${lines[1]}" == "WARNING:    Unable to move cast.out file for previous iteration 0" ]

  unset CHUMMEDLIST
  export MV_CMD="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cast.out file for previous iteration 0" ]

  

}

#
# castCHMTestJob() tests
#
@test "castCHMTestJob() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where cast fails
  export CASTBINARY="/bin/false"
  export CHUMMEDLIST="hi.q"
  /bin/rm -f "$THE_TMP/cast.out" 1>&2
  thecurdir=`pwd`
  run castCHMTestJob "1" "$THE_TMP" "1" "2" "foo"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Error calling /bin/false -t 1-2 -q hi.q -N foo  --writeoutputlocal -o $THE_TMP/${OUT_DIR_NAME}/stdout/\$TASK_ID.stdout -e $THE_TMP/${OUT_DIR_NAME}/stderr/\$TASK_ID.stderr $THE_TMP/runCHM.sh > $THE_TMP/chm.test.cast.out" ]
  newcurdir=`pwd`
  [ "$thecurdir" == "$newcurdir" ]

  # Test where all is good and there is a task file
  export CASTBINARY="/bin/echo"
  export CHUMMEDLIST="hi.q"
  export BATCH_AND_WALLTIME_ARGS="--walltime 2:00:00 --batchfactor hi.q::0.5"
  /bin/rm -f "$THE_TMP/cast.out" 1>&2
  thecurdir=`pwd`
  echo "blah" > "$THE_TMP/tasky"
  run castCHMTestJob "1" "$THE_TMP" "$THE_TMP/tasky" "2" "foo"
  [ "$status" -eq 0 ]
  aLine=`head -n 1 "$THE_TMP/chm.test.cast.out"`
  echo "$aLine" 1>&2
  [ "$aLine" == "--taskfile $THE_TMP/tasky -q hi.q -N foo --walltime 2:00:00 --batchfactor hi.q::0.5 --writeoutputlocal -o $THE_TMP/${OUT_DIR_NAME}/stdout/\$TASK_ID.stdout -e $THE_TMP/${OUT_DIR_NAME}/stderr/\$TASK_ID.stderr $THE_TMP/runCHM.sh" ]
  newcurdir=`pwd`
  [ "$thecurdir" == "$newcurdir" ]
}

#
# chumModelImageAndJobData() tests
#
@test "chumModelImageAndJobData() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export CHUMMEDLIST="foo.q"

  export CHUMBINARY="$THE_TMP/panfish/panfishchum"


  # Test where chum of model data fails
  echo "1,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Chum of $THE_TMP/model failed" ]
  [ "${lines[1]}" == "WARNING:    Unable to upload input model directory" ]


  # Test where upload input image data fails
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Chum of $THE_TMP/image failed" ]
  [ "${lines[1]}" == "WARNING:    Unable to upload input image directory" ]

  # Test where upload Job directory fails
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 3 ]
  [ "${lines[0]}" == "WARNING:    Chum of $THE_TMP failed" ]
  [ "${lines[1]}" == "WARNING:    Unable to upload job directory" ]


  # Test successful run
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 0 ]
}

# 
# waitForDownloadAndVerifyCHMTestJobs() tests
#
@test "waitForDownloadAndVerifyCHMTestJobs() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export WAIT_SLEEP_TIME=0
  export RETRY_SLEEP=0
  export CHUMMEDLIST="foo.q"
  export LANDBINARY="$THE_TMP/panfish/panfishland"

  # no cast.out file and land fails and verify fails
  echo "1,error,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "1,error,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "1,error,," >> "$THE_TMP/panfish/panfishland.tasks"
 
  run waitForDownloadAndVerifyCHMTestJobs "1" "$THE_TMP" "3" "cast.out"
  [ "$status" -eq 1 ]
  [ "${lines[4]}" == "  While checking if any jobs exist, it appears no cast.out exists." ]
  [ "${lines[11]}" == "WARNING:    Unable to download data.  Will continue on with checking results just in case." ] 
 

  # all good but verify fails
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  echo "hi" > "$THE_TMP/cast.out"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,error,," > "$THE_TMP/panfish/panfishland.tasks"
 
  run waitForDownloadAndVerifyCHMTestJobs "1" "$THE_TMP" "3" "cast.out"
 
  [ "$status" -eq 1 ]
  [ "${lines[7]}" == "  Creating failed.jobs file" ]
  [ -s "$THE_TMP/failed.jobs" ]

  # all good
  echo "hi" > "$THE_TMP/cast.out"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,error,," > "$THE_TMP/panfish/panfishland.tasks"
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stdout" 1>&2
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stderr" 1>&2
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/1.png" 1>&2
  echo "yo" > "$THE_TMP/${OUT_DIR_NAME}/1.png/1.png"
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"

     # make runchm.sh.config file
  echo "1${CONFIG_DELIM}xx" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png/1.png" >> "$THE_TMP/runCHM.sh.config"

  run waitForDownloadAndVerifyCHMTestJobs "1" "$THE_TMP" "1" "cast.out"

  [ "$status" -eq 0 ]

}

#
# runCHMTestJobs() tests
# 
@test "runCHMTestJobs() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export CHUMMEDLIST="foo.q"

  export CHUMBINARY="$THE_TMP/panfish/panfishchum"
  export CASTBINARY="$THE_TMP/panfish/panfishcast"
  export LANDBINARY="$THE_TMP/panfish/panfishland"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  export RETRY_SLEEP=0
  export WAIT_SLEEP_TIME=0
  export MAX_RETRIES=5

  # Test jobs already completed successfully
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stdout" 1>&2
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/stderr" 1>&2
  mkdir -p "$THE_TMP/${OUT_DIR_NAME}/1.png" 1>&2
  echo "yo" > "$THE_TMP/${OUT_DIR_NAME}/1.png/1.png"
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout"

     # make runchm.sh.config file
  echo "1${CONFIG_DELIM}xx" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png/1.png" >> "$THE_TMP/runCHM.sh.config"
 
  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [[ "${lines[1]}" == " WaitForJobs in chm.test.cast.out Iteration 0 Start Time: "* ]]
  [ "${lines[2]}" == "  Iteration 0 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[3]}" == "WARNING:    No $THE_TMP/chm.test.cast.out file found" ]
  [[ "${lines[4]}" == " WaitForJobs in chm.test.cast.out Iteration 0 End Time: "* ]]
  [ "${lines[5]}" == "  While checking if any jobs exist, it appears no chm.test.cast.out exists." ]
  [ "${lines[6]}" == "landcall" ]
  [[ "${lines[7]}" == " RunCHMTestJobs End Time: "* ]]

 
  # Test failure to chum data
  /bin/rm -f "$THE_TMP/${OUT_DIR_NAME}/1.png/1.png" 1>&2
  echo "0,download,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "1,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [ "${lines[11]}" == "ERROR:    Unable to upload job data" ]
  [ -s "$THE_TMP/failed.jobs" ]
  aLine=`head -n 1 "$THE_TMP/failed.jobs"`
  [ "$aLine" == "1" ]
   
  # Test failure to submit jobs
  echo "0,download,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,chumerror,," > "$THE_TMP/panfish/panfishcast.tasks"

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 1 ]
  [ "${lines[10]}" == "ERROR:    Unable to submit jobs" ]

  # Test success 1st time
  export CASTBINARY="/bin/echo"
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,,echo hi > $THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout" >> "$THE_TMP/panfish/panfishland.tasks"
  /bin/rm -f "$THE_TMP/${OUT_DIR_NAME}/stdout/1.stdout" 1>&2 
  echo "hi" > "$THE_TMP/${OUT_DIR_NAME}/1.png/1.png"
     # make runchm.sh.config file
  echo "1${CONFIG_DELIM}xx" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}${OUT_DIR_NAME}/1.png/1.png" >> "$THE_TMP/runCHM.sh.config"

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [[ "${lines[11]}" == " RunCHMTestJobs End Time: "* ]]
  [ -s "$THE_TMP/chm.test.cast.out" ]
  aLine=`head -n 1 "$THE_TMP/chm.test.cast.out"`
  [ "$aLine" == "--taskfile $THE_TMP/failed.jobs -q foo.q -N jobname --writeoutputlocal -o $THE_TMP/${OUT_DIR_NAME}/stdout/\$TASK_ID.stdout -e $THE_TMP/${OUT_DIR_NAME}/stderr/\$TASK_ID.stderr $THE_TMP/runCHM.sh" ]

  # Test 1st iteration fail then success
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,,echo hi > $THE_TMP/${OUT_DIR_NAME}/1.png/1.png" >> "$THE_TMP/panfish/panfishland.tasks"

  /bin/rm -f "$THE_TMP/${OUT_DIR_NAME}/1.png/1.png" 1>&2

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [[ "${lines[16]}" == " WaitForJobs in chm.test.cast.out Iteration 1 End Time: "* ]]

  # Test where failures exceed MAX_RETRIES
  export MAX_RETRIES=1
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"

  /bin/rm -f "$THE_TMP/${OUT_DIR_NAME}/1.png/1.png" 1>&2

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [ "${lines[21]}" == "WARNING:    Error running jobs...." ]
  [[ "${lines[22]}" == " RunCHMTestJobs End Time: "* ]]
}

#
# getParameterForTaskFromConfig() tests
#
@test "getParameterForTaskFromConfig() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  
  # Test where initial egrep fails
  run getParameterForTaskFromConfig "1" "1" "$THE_TMP/doesnotexist"
  [ "$status" -eq 1 ] 

  # Test where line to parse is larger then lines found
  echo "1${CONFIG_DELIM}a" > "$THE_TMP/config"
  echo "1${CONFIG_DELIM}b" >> "$THE_TMP/config"
  echo "1${CONFIG_DELIM}c" >> "$THE_TMP/config"
  echo "2${CONFIG_DELIM}d" >> "$THE_TMP/config"
  run getParameterForTaskFromConfig "1" "4" "$THE_TMP/config"
  echo "$output" 1>&2
  [ "$status" -eq 2 ]



  # Test valid extraction from 1st line
  getParameterForTaskFromConfig "1" "1" "$THE_TMP/config"
  [ "$?" -eq 0 ] 
  [ "$TASK_CONFIG_PARAM" == "a" ]
  

  # Test valid extraction from 2nd line
  getParameterForTaskFromConfig "1" "2" "$THE_TMP/config"
  [ "$?" -eq 0 ]
  [ "$TASK_CONFIG_PARAM" == "b" ]

  # Test valid extraction from 3rd line
  getParameterForTaskFromConfig "1" "1" "$THE_TMP/config"
  [ "$?" -eq 0 ]
  [ "$TASK_CONFIG_PARAM" == "a" ]


  # Test valid extraction from 4th line
  echo "2${CONFIG_DELIM}e" >> "$THE_TMP/config"
  echo "2${CONFIG_DELIM}f" >> "$THE_TMP/config"
  echo "2${CONFIG_DELIM} g -g ?" >> "$THE_TMP/config"
  getParameterForTaskFromConfig "2" "4" "$THE_TMP/config"
  [ "$?" -eq 0 ]
  [ "$TASK_CONFIG_PARAM" == " g -g ?" ]
}

#
# getCHMTestJobParametersForTaskFromConfig() tests
#
@test "getCHMTestJobParametersForTaskFromConfig() tests" {
  
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where we can't get first parameter
  run getCHMTestJobParametersForTaskFromConfig "1" "$THE_TMP"
  [ "$status" -eq 1 ]
  
  # Test where we can't get 2nd parameter
  echo "1${CONFIG_DELIM}a" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}b" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}c" >> "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}d" >> "$THE_TMP/runCHM.sh.config"
  echo "2${CONFIG_DELIM}e" >> "$THE_TMP/runCHM.sh.config"
  run getCHMTestJobParametersForTaskFromConfig "2" "$THE_TMP"
  [ "$status" -eq 2 ]

  # Test where we cant get 3rd parameter
  echo "2${CONFIG_DELIM}f" >> "$THE_TMP/runCHM.sh.config"
  run getCHMTestJobParametersForTaskFromConfig "2" "$THE_TMP"
  [ "$status" -eq 3 ]


  # Test where we cant get 4th parameter
  echo "2${CONFIG_DELIM}g" >> "$THE_TMP/runCHM.sh.config"
  run getCHMTestJobParametersForTaskFromConfig "2" "$THE_TMP"
  [ "$status" -eq 4 ]

  # Test success
  echo "2${CONFIG_DELIM}h" >> "$THE_TMP/runCHM.sh.config"
  getCHMTestJobParametersForTaskFromConfig "2" "$THE_TMP"
  [ "$?" -eq 0 ]

  [ "$INPUT_IMAGE" == "e" ]
  [ "$MODEL_DIR" == "f" ]
  [ "$CHM_OPTS" == "g" ]
  [ "$OUTPUT_IMAGE" == "h" ]
}

#
# getNumberOfCHMTestJobsFromConfig() tests
#
@test "getNumberOfCHMTestJobsFromConfig() tests" {
 
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where there isnt a config
  run getNumberOfCHMTestJobsFromConfig "$THE_TMP"
  [ "$status" -eq 1 ]

  # Test on empty file
  touch "$THE_TMP/runCHM.sh.config"
  run getNumberOfCHMTestJobsFromConfig "$THE_TMP"
  [ "$status" -eq 2 ]

  # Test on file without proper job #${CONFIG_DELIM} prefix on last line
  echo "yikes" > "$THE_TMP/runCHM.sh.config"
  getNumberOfCHMTestJobsFromConfig "$THE_TMP"
  [ "$?" -eq 0 ]
  [ "$NUMBER_JOBS" == "yikes" ]

  # Test on file with 1 job
  echo "1${CONFIG_DELIM}a" > "$THE_TMP/runCHM.sh.config"
  echo "1${CONFIG_DELIM}b" >> "$THE_TMP/runCHM.sh.config"
  getNumberOfCHMTestJobsFromConfig "$THE_TMP"
  [ "$?" -eq 0 ]
  [ "$NUMBER_JOBS" -eq 1 ]


  # Test on file with 4 jobs
  echo "4${CONFIG_DELIM}asdf" >> "$THE_TMP/runCHM.sh.config"
  getNumberOfCHMTestJobsFromConfig "$THE_TMP"
  [ "$?" -eq 0 ]
  [ "$NUMBER_JOBS" -eq 4 ]


 

}

  # Test where getStatus errors 1st time and works second going to run state and finishes 3rd time
  echo "1,status=uh,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
   
  run waitForJobs "2" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 2 Start Time: "* ]]
  [ "${lines[1]}" == "  Iteration 2 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[2]}" == "WARNING:    Error calling $PANFISHSTATBINARY --statusofjobs $THE_TMP/cast.out" ]
  [ "${lines[3]}" == "  Iteration 2 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[4]}" == "  Iteration 2 job status is running.  Sleeping 0 seconds" ]
  [[ "${lines[5]}" == " WaitForJobs in cast.out Iteration 2 End Time: "* ]]

  
  # Test where a kill file is found
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  export CHUMBINARY="/bin/true"
  echo "die" > "$THE_TMP/KILL.JOB.REQUEST"
  run waitForJobs "3" "$THE_TMP" "cast.out"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " WaitForJobs in cast.out Iteration 3 Start Time: "* ]]
  [ "${lines[1]}" == "  $THE_TMP/KILL.JOB.REQUEST detected.  Chumming kill file to remote cluster(s) and exiting..." ]
  [ "${lines[2]}" == "  Running /bin/true --path $THE_TMP/KILL.JOB.REQUEST --cluster foo.q > $THE_TMP/killed.chum.out" ]

  # Test where DOWNLOAD.DATA.REQUEST file shows up and job then finishes after
  echo "yo" > "$THE_TMP/chm.cast.out"
  export LANDBINARY="/bin/echo"
  export LAND_JOB_OPTS="--exclude hi.* --exclude bye.*" 
  echo "0,status=running,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,,touch $THE_TMP/DOWNLOAD.DATA.REQUEST" >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=running,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"  
  /bin/rm -f $THE_TMP/KILL.JOB.REQUEST

  run waitForJobs "3" "$THE_TMP" "chm.cast.out"
  [ "$status" -eq 0 ]
  echo "$output" 1>&2
  [[ "${lines[0]}" == " WaitForJobs in chm.cast.out Iteration 3 Start Time: "* ]]
  [ "${lines[1]}" == "  Iteration 3 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[2]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [ "${lines[3]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [ "${lines[4]}" == "  DOWNLOAD.DATA.REQUEST file found.  Performing download" ]
  [ "${lines[5]}" == "--path $THE_TMP --cluster foo.q --exclude hi.* --exclude bye.*" ]
  [ "${lines[6]}" == "  Removing DOWNLOAD.DATA.REQUEST file" ]
  [ "${lines[7]}" == "  Iteration 3 job status is running.  Sleeping 0 seconds" ]
  [[ "${lines[8]}" == " WaitForJobs in chm.cast.out Iteration 3 End Time: "* ]] 
  [ ! -e "$THE_TMP/DOWNLOAD.DATA.REQUEST" ]

}

#
# moveCastOutFile() tests
#
@test "moveCastOutFile() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where there isn't a cast.out file
  run moveCastOutFile "1" "$THE_TMP" "hi"
  [ "$status" -eq 0 ]

  # Test where there is a cast.out file
  echo "hi" > "$THE_TMP/cast.out"
  run moveCastOutFile "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/cast.out.1.out" ]
  [ ! -e "$THE_TMP/cast.out" ]

  # Test where there is a cast.1.out file
  echo "bye" > "$THE_TMP/cast.out"
  run moveCastOutFile "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
  [ -e "$THE_TMP/cast.out.1.out" ]
  [ ! -e "$THE_TMP/cast.out" ]
  aLine=`head -n 1 "$THE_TMP/cast.out.1.out"`
  [ "$aLine" == "bye" ]

  # Test where mv command fails
  export MV_CMD="/bin/false"
  echo "yo" > "$THE_TMP/cast.out" "cast.out"
  run moveCastOutFile "1" "$THE_TMP"
  [ "$status" -eq 1 ]
}

#
# moveOldDataForNewIteration() tests
#
@test "moveOldDataForNewIteration() tests" {
    
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
 
  # test where CHUMMEDLIST is empty and no cast.out file
  unset CHUMMEDLIST
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 0 ]
 
  # test where there is an error moving the old cluster folders
  export MV_CMD="/bin/false"
  export CHUMMEDLIST="foo.q"
  mkdir -p "$THE_TMP/foo.q" 1>&2
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cluster folders for previous iteration 0" ]

  # test where there is an error moving cluster folders and moving cast.out file
  export MV_CMD="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 3 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cluster folders for previous iteration 0" ]
  [ "${lines[1]}" == "WARNING:    Unable to move cast.out file for previous iteration 0" ]

  unset CHUMMEDLIST
  export MV_CMD="/bin/false"
  echo "hi" > "$THE_TMP/cast.out"
  run moveOldDataForNewIteration "1" "$THE_TMP" "cast.out"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Unable to move cast.out file for previous iteration 0" ]

  

}

#
# castCHMTestJob() tests
#
@test "castCHMTestJob() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # Test where cast fails
  export CASTBINARY="/bin/false"
  export CHUMMEDLIST="hi.q"
  /bin/rm -f "$THE_TMP/cast.out" 1>&2
  thecurdir=`pwd`
  run castCHMTestJob "1" "$THE_TMP" "1" "2" "foo"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Error calling /bin/false -t 1-2 -q hi.q -N foo  --writeoutputlocal -o $THE_TMP/out/stdout/\$TASK_ID.stdout -e $THE_TMP/out/stderr/\$TASK_ID.stderr $THE_TMP/runCHM.sh > $THE_TMP/chm.test.cast.out" ]
  newcurdir=`pwd`
  [ "$thecurdir" == "$newcurdir" ]

  # Test where all is good and there is a task file
  export CASTBINARY="/bin/echo"
  export CHUMMEDLIST="hi.q"
  export BATCH_AND_WALLTIME_ARGS="--walltime 2:00:00 --batchfactor hi.q::0.5"
  /bin/rm -f "$THE_TMP/cast.out" 1>&2
  thecurdir=`pwd`
  echo "blah" > "$THE_TMP/tasky"
  run castCHMTestJob "1" "$THE_TMP" "$THE_TMP/tasky" "2" "foo"
  [ "$status" -eq 0 ]
  aLine=`head -n 1 "$THE_TMP/chm.test.cast.out"`
  echo "$aLine" 1>&2
  [ "$aLine" == "--taskfile $THE_TMP/tasky -q hi.q -N foo --walltime 2:00:00 --batchfactor hi.q::0.5 --writeoutputlocal -o $THE_TMP/out/stdout/\$TASK_ID.stdout -e $THE_TMP/out/stderr/\$TASK_ID.stderr $THE_TMP/runCHM.sh" ]
  newcurdir=`pwd`
  [ "$thecurdir" == "$newcurdir" ]
}

#
# chumModelImageAndJobData() tests
#
@test "chumModelImageAndJobData() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export CHUMMEDLIST="foo.q"

  export CHUMBINARY="$THE_TMP/panfish/panfishchum"


  # Test where chum of model data fails
  echo "1,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "WARNING:    Chum of $THE_TMP/model failed" ]
  [ "${lines[1]}" == "WARNING:    Unable to upload input model directory" ]


  # Test where upload input image data fails
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "WARNING:    Chum of $THE_TMP/image failed" ]
  [ "${lines[1]}" == "WARNING:    Unable to upload input image directory" ]

  # Test where upload Job directory fails
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 3 ]
  [ "${lines[0]}" == "WARNING:    Chum of $THE_TMP failed" ]
  [ "${lines[1]}" == "WARNING:    Unable to upload job directory" ]


  # Test successful run
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  run chumModelImageAndJobData "$iteration" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model"
  [ "$status" -eq 0 ]
}

# 
# waitForDownloadAndVerifyCHMTestJobs() tests
#
@test "waitForDownloadAndVerifyCHMTestJobs() tests" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export WAIT_SLEEP_TIME=0
  export RETRY_SLEEP=0
  export CHUMMEDLIST="foo.q"
  export LANDBINARY="$THE_TMP/panfish/panfishland"

  # no cast.out file and land fails and verify fails
  echo "1,error,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "1,error,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "1,error,," >> "$THE_TMP/panfish/panfishland.tasks"
 
  run waitForDownloadAndVerifyCHMTestJobs "1" "$THE_TMP" "3" "cast.out"
  [ "$status" -eq 1 ]
  [ "${lines[4]}" == "  While checking if any jobs exist, it appears no cast.out exists." ]
  [ "${lines[11]}" == "WARNING:    Unable to download data.  Will continue on with checking results just in case." ] 
 

  # all good but verify fails
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  echo "hi" > "$THE_TMP/cast.out"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,error,," > "$THE_TMP/panfish/panfishland.tasks"
 
  run waitForDownloadAndVerifyCHMTestJobs "1" "$THE_TMP" "3" "cast.out"
 
  [ "$status" -eq 1 ]
  [ "${lines[7]}" == "  Creating failed.jobs file" ]
  [ -s "$THE_TMP/failed.jobs" ]

  # all good
  echo "hi" > "$THE_TMP/cast.out"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,error,," > "$THE_TMP/panfish/panfishland.tasks"
  mkdir -p "$THE_TMP/out/stdout" 1>&2
  mkdir -p "$THE_TMP/out/stderr" 1>&2
  mkdir -p "$THE_TMP/out/1.png" 1>&2
  echo "yo" > "$THE_TMP/out/1.png/1.png"
  echo "hi" > "$THE_TMP/out/stdout/1.stdout"

     # make runchm.sh.config file
  echo "1:::xx" > "$THE_TMP/runCHM.sh.config"
  echo "1:::xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::out/1.png/1.png" >> "$THE_TMP/runCHM.sh.config"

  run waitForDownloadAndVerifyCHMTestJobs "1" "$THE_TMP" "1" "cast.out"

  [ "$status" -eq 0 ]

}

#
# runCHMTestJobs() tests
# 
@test "runCHMTestJobs() tests" {
  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS
  /bin/cp -a "$PANFISH_TEST_BIN" "$THE_TMP/panfish" 1>&2
  export CHUMMEDLIST="foo.q"

  export CHUMBINARY="$THE_TMP/panfish/panfishchum"
  export CASTBINARY="$THE_TMP/panfish/panfishcast"
  export LANDBINARY="$THE_TMP/panfish/panfishland"
  export PANFISHSTATBINARY="$THE_TMP/panfish/panfishstat"
  export RETRY_SLEEP=0
  export WAIT_SLEEP_TIME=0
  export MAX_RETRIES=5

  # Test jobs already completed successfully
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  mkdir -p "$THE_TMP/out/stdout" 1>&2
  mkdir -p "$THE_TMP/out/stderr" 1>&2
  mkdir -p "$THE_TMP/out/1.png" 1>&2
  echo "yo" > "$THE_TMP/out/1.png/1.png"
  echo "hi" > "$THE_TMP/out/stdout/1.stdout"

     # make runchm.sh.config file
  echo "1:::xx" > "$THE_TMP/runCHM.sh.config"
  echo "1:::xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::out/1.png/1.png" >> "$THE_TMP/runCHM.sh.config"
 
  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [[ "${lines[1]}" == " WaitForJobs in chm.test.cast.out Iteration 0 Start Time: "* ]]
  [ "${lines[2]}" == "  Iteration 0 job status is NA.  Sleeping 0 seconds" ]
  [ "${lines[3]}" == "WARNING:    No $THE_TMP/chm.test.cast.out file found" ]
  [[ "${lines[4]}" == " WaitForJobs in chm.test.cast.out Iteration 0 End Time: "* ]]
  [ "${lines[5]}" == "  While checking if any jobs exist, it appears no chm.test.cast.out exists." ]
  [ "${lines[6]}" == "landcall" ]
  [[ "${lines[7]}" == " RunCHMTestJobs End Time: "* ]]

 
  # Test failure to chum data
  /bin/rm -f "$THE_TMP/out/1.png/1.png" 1>&2
  echo "0,download,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "1,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [ "${lines[11]}" == "ERROR:    Unable to upload job data" ]
  [ -s "$THE_TMP/failed.jobs" ]
  aLine=`head -n 1 "$THE_TMP/failed.jobs"`
  [ "$aLine" == "1" ]
   
  # Test failure to submit jobs
  echo "0,download,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "1,chumerror,," > "$THE_TMP/panfish/panfishcast.tasks"

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 1 ]
  [ "${lines[10]}" == "ERROR:    Unable to submit jobs" ]

  # Test success 1st time
  export CASTBINARY="/bin/echo"
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,,echo hi > $THE_TMP/out/stdout/1.stdout" >> "$THE_TMP/panfish/panfishland.tasks"
  /bin/rm -f "$THE_TMP/out/stdout/1.stdout" 1>&2 
  echo "hi" > "$THE_TMP/out/1.png/1.png"
     # make runchm.sh.config file
  echo "1:::xx" > "$THE_TMP/runCHM.sh.config"
  echo "1:::xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::xx" >> "$THE_TMP/runCHM.sh.config"
  echo "1:::out/1.png/1.png" >> "$THE_TMP/runCHM.sh.config"

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [[ "${lines[11]}" == " RunCHMTestJobs End Time: "* ]]
  [ -s "$THE_TMP/chm.test.cast.out" ]
  aLine=`head -n 1 "$THE_TMP/chm.test.cast.out"`
  [ "$aLine" == "--taskfile $THE_TMP/failed.jobs -q foo.q -N jobname --writeoutputlocal -o $THE_TMP/out/stdout/\$TASK_ID.stdout -e $THE_TMP/out/stderr/\$TASK_ID.stderr $THE_TMP/runCHM.sh" ]

  # Test 1st iteration fail then success
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,,echo hi > $THE_TMP/out/1.png/1.png" >> "$THE_TMP/panfish/panfishland.tasks"

  /bin/rm -f "$THE_TMP/out/1.png/1.png" 1>&2

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [[ "${lines[16]}" == " WaitForJobs in chm.test.cast.out Iteration 1 End Time: "* ]]

  # Test where failures exceed MAX_RETRIES
  export MAX_RETRIES=1
  echo "0,chummed.clusters=foo.q,," > "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,chummed.clusters=foo.q,," >> "$THE_TMP/panfish/panfishchum.tasks"
  echo "0,status=done,," > "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,status=done,," >> "$THE_TMP/panfish/panfishstat.tasks"
  echo "0,landcall,," > "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"
  echo "0,landcall,," >> "$THE_TMP/panfish/panfishland.tasks"

  /bin/rm -f "$THE_TMP/out/1.png/1.png" 1>&2

  run runCHMTestJobs "0" "$THE_TMP" "$THE_TMP/image" "$THE_TMP/model" "1" "jobname"
  [ "$status" -eq 1 ]
  [[ "${lines[0]}" == " RunCHMTestJobs Start Time: "* ]]
  [ "${lines[21]}" == "WARNING:    Error running jobs...." ]
  [[ "${lines[22]}" == " RunCHMTestJobs End Time: "* ]]


}
