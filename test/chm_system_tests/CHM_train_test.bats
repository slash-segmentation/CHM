#!/usr/bin/env bats

setup() {
  if [ -n "$CHM_ALT_BIN_DIR" ] ; then
    CHM_TEST="$CHM_ALT_BIN_DIR/CHM_test.sh"
    CHM_TRAIN="$CHM_ALT_BIN_DIR/CHM_train.sh"
  else
    CHM_TEST="${BATS_TEST_DIRNAME}/../../algorithm/CHM_test.sh"
    CHM_TRAIN="${BATS_TEST_DIRNAME}/../../algorithm/CHM_train.sh"
  fi
  echo "Running tests on $CHM_TEST" 1>&2

  export THE_TMP="${BATS_TMPDIR}/chm_train."`uuidgen`
  /bin/mkdir -p $THE_TMP

  export TWOHONDO_DIR="${BATS_TEST_DIRNAME}/200x190train"
  export TWOHONDO_TRAIN_DIR="$TWOHONDO_DIR/n2s2"
  export TWOHONDO_IMAGES_DIR="$TWOHONDO_DIR/trainimages"
  export TWOHONDO_LABELS_DIR="$TWOHONDO_DIR/labels"
  export TWOHONDO_IMAGE="$TWOHONDO_DIR/10.png"
  export TWOHONDO_IMAGE_TIFF="$TWOHONDO_DIR/10.tiff"

  export ONEHONDO_DIR="${BATS_TEST_DIRNAME}/100x95train"
  export ONEHONDO_TINYIMAGES_DIR="$ONEHONDO_DIR/trainimages"
  export ONEHONDO_TINYLABELS_DIR="$ONEHONDO_DIR/labels"
  export ONEHONDO_TINYIMAGE="$ONEHONDO_DIR/5.png"
}

teardown() {
  /bin/rm -rf "$THE_TMP" 1>&2
}

#
# Train and Test
# Test single tile -s -b 100x95 -t 1,1 from 100x95train
#
@test "Train and Test single tile -s -b 100x95 -t 1,1 from 100x95rain (5 min runtime)" {
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

  /bin/mkdir -p "$THE_TMP/result" 1>&2

  run $CHM_TRAIN "$ONEHONDO_TINYIMAGES_DIR" "$ONEHONDO_TINYLABELS_DIR" -m "$THE_TMP/result" -S 2 -L 1 -s

  echo "$output" 1>&2
  [ "$status" -eq 0 ]


  run $CHM_TEST $ONEHONDO_TINYIMAGE $THE_TMP -m "$THE_TMP/result" -s -b 100x95 -t 1,1 -o 0x0 -h

  echo "Output from compare: $output" 1>&2

  [ "$status" -eq 0 ]
  
  # Verify result image was created
  [ -s "$THE_TMP/5.png" ]
   
  run compare -metric ae "$THE_TMP/5.png" $ONEHONDO_DIR/5.alltiles.probmap.nooverlap.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 700 pixels have different intensities
  [ "${lines[0]}" -lt 700 ]

  # no run CHM again this time with no -h
  run $CHM_TEST $ONEHONDO_TINYIMAGE $THE_TMP -m "$THE_TMP/result" -s -b 100x95 -t 1,1 -o 0x0 

  echo "Output from compare: $output" 1>&2

  [ "$status" -eq 0 ]

  # Verify result image was created
  [ -s "$THE_TMP/5.png" ]

  run compare -metric ae "$THE_TMP/5.png" $ONEHONDO_DIR/5.alltiles.probmap.nooverlap.png /dev/null

  echo "Output from compare: $output" 1>&2
  # verify the comparison ran without error
  [ "$status" -eq 0 ]

  # verify that no more then 700 pixels have different intensities
  [ "${lines[0]}" -lt 700 ]


}

