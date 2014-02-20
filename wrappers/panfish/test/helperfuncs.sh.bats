#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/helperfuncs" 1>&2
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp -a ${BATS_TEST_DIRNAME}/../scripts/helperfuncs.sh "$THE_TMP/." 1>&2
  export TESTIMAGE="${BATS_TEST_DIRNAME}/createchmjob/600by400.png" 
  export TESTIMAGE_DIR="${BATS_TEST_DIRNAME}/createchmjob"
  export HELPERFUNCS="$THE_TMP/helperfuncs.sh"

}

teardown(){
  /bin/rm -rf "$THE_TMP"
  echo "2" 1>&2
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
  echo "${lines[0]}" > /home/churas/wellshit 
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
# createConfig() no images
#
@test "createConfig() no images" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS


  run createConfig "$THE_TMP" "$THE_TMP/foo" 1 1 1 " " "png"

  [ "$status" -eq 0 ]
  [ ! -e "$THE_TMP/foo" ] 
}

#
# createConfig() 1 image 1 tile
#
@test "createConfig() 1 image 1 tile" {

  # source helperfuncs.sh to we can call the function  
  . $HELPERFUNCS


  echo "png" > "$THE_TMP"/1.png

  run createConfig "$THE_TMP" "$THE_TMP/foo" 1 1 1 " " "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 3 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "3" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1:::$THE_TMP/1.png" ]
  
  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::   -t 1,1" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::out/1.png/1.png" ]

}

#
# createConfig() 1 image 1 tile and opts
#
@test "createConfig() 1 image 1 tile and ops" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createConfig "$THE_TMP" "$THE_TMP/foo" 1 1 1 "-o 1x1" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 3 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "3" ]

  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1:::$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::-o 1x1  -t 1,1" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::out/1.png/1.png" ]

}

#
# createConfig() 1 image 18 tiles and opts
#
@test "createConfig() 1 image 18 tiles and ops" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createConfig "$THE_TMP" "$THE_TMP/foo" 3 2 1 "-o 1x1" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]
  
  cat $THE_TMP/foo 1>&2

  # verify we have 18 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "18" ]
  # Check the lines
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1:::$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::-o 1x1  -t 3,2" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::out/1.png/1.png" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::$THE_TMP/1.png" ]

  aLine=`head -n 5 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::-o 1x1  -t 3,1" ]

  aLine=`head -n 6 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::out/1.png/2.png" ]

  aLine=`head -n 7 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "3:::$THE_TMP/1.png" ]

  aLine=`head -n 8 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "3:::-o 1x1  -t 2,2" ]

  aLine=`head -n 9 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "3:::out/1.png/3.png" ]

    aLine=`head -n 10 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "4:::$THE_TMP/1.png" ]
  
  aLine=`head -n 11 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "4:::-o 1x1  -t 2,1" ]

  aLine=`head -n 12 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "4:::out/1.png/4.png" ]

  aLine=`head -n 13 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "5:::$THE_TMP/1.png" ]
  
  aLine=`head -n 14 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "5:::-o 1x1  -t 1,2" ]

  aLine=`head -n 15 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "5:::out/1.png/5.png" ]

  aLine=`head -n 16 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "6:::$THE_TMP/1.png" ]

  aLine=`head -n 17 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "6:::-o 1x1  -t 1,1" ]

  aLine=`head -n 18 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "6:::out/1.png/6.png" ]

}

#
# createConfig() 1 image 6 tiles and opts and only 1 job
#
@test "createConfig() 1 image 6 tiles and opts and only 1 job" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createConfig "$THE_TMP" "$THE_TMP/foo" 2 3 6 "-o 1x1" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 3 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "3" ]
  cat $THE_TMP/foo 1>&2
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1:::$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::out/1.png/1.png" ]

}

#
# createConfig() 1 image 6 tiles and opts and 2 jobs
#
@test "createConfig() 1 image 6 tiles and opts and 2 jobs" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP"/1.png

  run createConfig "$THE_TMP" "$THE_TMP/foo" 2 3 5 "-o 1x1" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 6 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "6" ]
  cat $THE_TMP/foo 1>&2
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1:::$THE_TMP/1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::out/1.png/1.png" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::$THE_TMP/1.png" ]

  aLine=`head -n 5 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::-o 1x1  -t 1,1" ]

  aLine=`head -n 6 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::out/1.png/2.png" ]

}

#
# createConfig() 2 cols 3 rows 7 tiles per job with 2 images and opts
#
@test "createConfig() 2 cols 3 rows 7 tiles per job with 2 images and opts" {

  # source helperfuncs.sh to we can call the function
  . $HELPERFUNCS

  # make a fake image
  echo "png" > "$THE_TMP/hist1.png"
  echo "png" > "$THE_TMP/hist2.png"

  run createConfig "$THE_TMP" "$THE_TMP/foo" 2 3 7 "-o 1x1" "png"

  [ "$status" -eq 0 ]
  [  -e "$THE_TMP/foo" ]

  # verify we have 6 lines
  [ `wc -l "$THE_TMP/foo" | sed "s/ .*//"` == "6" ]
  cat $THE_TMP/foo 1>&2
  # Check each line
  local aLine=`head -n 1 "$THE_TMP/foo"`
  [ "$aLine" == "1:::$THE_TMP/hist1.png" ]

  aLine=`head -n 2 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 3 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "1:::out/hist1.png/1.png" ]

  aLine=`head -n 4 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::$THE_TMP/hist2.png" ]

  aLine=`head -n 5 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::-o 1x1  -t 2,3 -t 2,2 -t 2,1 -t 1,3 -t 1,2 -t 1,1" ]

  aLine=`head -n 6 "$THE_TMP/foo" | tail -n 1`
  [ "$aLine" == "2:::out/hist2.png/2.png" ]
}


