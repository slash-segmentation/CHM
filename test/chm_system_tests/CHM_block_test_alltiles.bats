#!/usr/bin/env bats

setup() {
  CHM_TEST="${BATS_TEST_DIRNAME}/../../algorithm/CHM_test.sh"
  export THE_TMP="${BATS_TMPDIR}/"`uuidgen`
  /bin/mkdir -p $THE_TMP

  export TWOHONDO_DIR="${BATS_TEST_DIRNAME}/200x190train"
  export TWOHONDO_TRAIN_DIR="$TWOHONDO_DIR/n2s2"
  export TWOHONDO_IMAGE="$TWOHONDO_DIR/10.png"
}

teardown() {
  /bin/rm -f "$THE_TMP/10.png" 1>&2
  /bin/rmdir $THE_TMP 1>&2
}


#
# Test entire 10.png from 200x190train with -b 200x190 (10 min runtime)
#
@test "Test entire 10.png from 200x190train with -b 200x190 (10 min runtime)" {

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

  run $CHM_TEST $TWOHONDO_IMAGE $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x190
  echo "$output" > /home/churas/x2 
  echo "$CHM_TEST $TWOHONDO_IMAGE $THE_TMP -m $TWOHONDO_TRAIN_DIR -s -b 200x190" >> /home/churas/x
  [ "$status" -eq 0 ] 
  [ "${lines[11]}" = "Block Processing 20 blocks." ]

  # Verify result image matches closely to original result
  [ -s "$THE_TMP/10.png" ]

  run compare -metric ae "$THE_TMP/10.png" $TWOHONDO_DIR/10.alltiles.probmap.nooverlap.png /dev/null

  # verify the comparison ran without error
  [ "$status" -eq 0 ] 

  # verify that no more then 100 pixels have different intensities
  [ "${lines[0]}" -lt 100 ]
  echo "${output}"
}
