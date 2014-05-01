#!/usr/bin/env bats

setup() {
  CHM_TEST="${BATS_TEST_DIRNAME}/../../CHM_test.sh"
  export THE_TMP="${BATS_TMPDIR}/chm_test."`uuidgen`
  /bin/mkdir -p $THE_TMP

  export TWOHONDO_DIR="${BATS_TEST_DIRNAME}/200x190train"
  export TWOHONDO_TRAIN_DIR="$TWOHONDO_DIR/n2s2"
  export TWOHONDO_IMAGE="$TWOHONDO_DIR/10.png"
  export TWOHONDO_IMAGE_TIFF="$TWOHONDO_DIR/10.tiff"
}

teardown() {
  /bin/rm -f "$THE_TMP/10.png" 1>&2
  /bin/rm -f "$THE_TMP/10.tiff" 1>&2
  /bin/rmdir "$THE_TMP" 1>&2
   
}

#
# Test single tile outside of image -s -b 200x200
#
@test "Test single tile outside of image -s -b 200x200 -t -1,-1" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  # Check for Image Magick compare program
  run which compare

  if [ "$status" -eq 1 ] ; then
    skip "compare (Image Magick program) not in path"
  fi

  run $CHM_TEST $TWOHONDO_IMAGE $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x200 -t -1,-1
  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/10.png" ]

  run compare -metric ae "$THE_TMP/10.png" $TWOHONDO_DIR/10.allblack.png /dev/null
 
  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 0 pixels have different intensities
  [ "${lines[0]}" -eq 0 ]
}

#
# Test single tile outside of tiff image -s -b 200x200
#
@test "Test single tile outside of tiff image -s -b 200x200 -t -1,-1" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  # Check for Image Magick compare program
  run which compare

  if [ "$status" -eq 1 ] ; then
    skip "compare (Image Magick program) not in path"
  fi

  run $CHM_TEST $TWOHONDO_IMAGE_TIFF $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x200 -t -1,-1
  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/10.tiff" ]

  run compare -metric ae "$THE_TMP/10.tiff" $TWOHONDO_DIR/10.allblack.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 0 pixels have different intensities
  [ "${lines[0]}" -eq 0 ]
}



#
# Test single tile outside of tiff image -s -b 200x190 -t -1,-1
#
@test "Test single tile outside of tiff image -s -b 200x190 -t -1,-1" {
  
  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  # Check for Image Magick compare program
  run which compare

  if [ "$status" -eq 1 ] ; then
    skip "compare (Image Magick program) not in path"
  fi

  run $CHM_TEST $TWOHONDO_IMAGE_TIFF $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x190 -t -1,-1
  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/10.tiff" ]

  run compare -metric ae "$THE_TMP/10.tiff" $TWOHONDO_DIR/10.allblack.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 100 pixels have different intensities
  [ "${lines[0]}" -eq 0 ]

}

#
# Test single tile outside of image -s -b 200x190 -t -1,-1
#
@test "Test single tile outside of image -s -b 200x190 -t -1,-1" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  # Check for Image Magick compare program
  run which compare

  if [ "$status" -eq 1 ] ; then
    skip "compare (Image Magick program) not in path"
  fi

  run $CHM_TEST $TWOHONDO_IMAGE $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x190 -t -1,-1
  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/10.png" ]

  run compare -metric ae "$THE_TMP/10.png" $TWOHONDO_DIR/10.allblack.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 0 pixels have different intensities
  [ "${lines[0]}" -eq 0 ]

}

#
# Test single tile -s -b 200x190 -t 1,1 from 200x190train
#
@test "Test single tile -s -b 200x190 -t 1,1 from 200x190train (1 min runtime)" {
  
  # Verify matlab is in the users path via which command
  run which matlab
  
  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  # Check for Image Magick compare program
  run which compare

  if [ "$status" -eq 1 ] ; then
    skip "compare (Image Magick program) not in path"
  fi 

  run $CHM_TEST $TWOHONDO_IMAGE $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x190 -t 1,1 -o 0x0

  echo "Output from compare: $output" 1>&2

  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/10.png" ]

  run compare -metric ae "$THE_TMP/10.png" $TWOHONDO_DIR/10.alltiles.probmap.nooverlap.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 37039 pixels have different intensities
  [ "${lines[0]}" -eq 37039 ]


}

#
# Test single tiff tile -s -b 200x190 -t 1,1 from 200x190train
#
@test "Test single tiff tile -s -b 200x190 -t 1,1 from 200x190train (1 min runtime)" {
  
  # Verify matlab is in the users path via which command
  run which matlab
  
  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  # Check for Image Magick compare program
  run which compare

  if [ "$status" -eq 1 ] ; then
    skip "compare (Image Magick program) not in path"
  fi

  run $CHM_TEST $TWOHONDO_IMAGE_TIFF $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x190 -o 0x0 -t 1,1

  echo "Output from compare: $output" 1>&2

  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/10.tiff" ]

  run compare -metric ae "$THE_TMP/10.tiff" $TWOHONDO_DIR/10.alltiles.probmap.nooverlap.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 37039 pixels have different intensities
  [ "${lines[0]}" -eq 37039 ]

}

#
# Test multiple tiles -s -b 200x190 -t 1,2 -t 4,4 from 200x190train
#
@test "Test multiple tiles -s -b 200x190 -t 1,2 -t 4,4 from 200x190train (2 min runtime)" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  # Check for Image Magick compare program
  run which compare

  if [ "$status" -eq 1 ] ; then
    skip "compare (Image Magick program) not in path"
  fi

  run $CHM_TEST $TWOHONDO_IMAGE $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x190 -o 0x0 -t 1,2 -t 3,5

  echo "Output from compare: $output" 1>&2

  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/10.png" ]
  run compare -metric ae "$THE_TMP/10.png" $TWOHONDO_DIR/10.alltiles.probmap.nooverlap.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 34852 pixels have different intensities
  [ "${lines[0]}" -eq 34852 ]

}

