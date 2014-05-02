#!/usr/bin/env bats

setup() {
  export THE_TMP="${BATS_TMPDIR}/chm_train_compiled."`uuidgen`
  /bin/mkdir -p $THE_TMP 1>&2
  /bin/cp "${BATS_TEST_DIRNAME}/../CHM_train.sh" "$THE_TMP/." 1>&2
  chmod a+x "$THE_TMP/CHM_train.sh"
  export CHM_TRAIN="$THE_TMP/CHM_train.sh"

  export FAKE_SUCCESS="${BATS_TEST_DIRNAME}/bin/fakesuccess/success.sh"
  export FAKE_FAIL="${BATS_TEST_DIRNAME}/bin/fakefail/fail.sh"
  export CHM_TRAIN_NAME="CHM_train"
  curdir=`pwd`
  cd $THE_TMP
  export CHM_TRAIN_PROG_NAME="CHM Image Training Phase Script.  @@VERSION@@"
}

teardown() {
  cd $curdir
  /bin/rm -rf $THE_TMP
}

#
# CHM_train.sh with no arguments
#
@test "CHM_train.sh with no arguments" {
  run $CHM_TRAIN
  echo "$output" 1>&2
  [ "$status" -eq 1 ] 
  [ "${lines[0]}" == "$CHM_TRAIN_PROG_NAME" ]
  [ "${lines[1]}" == "$CHM_TRAIN <inputs> <labels> <optional arguments>" ]
}

#
# CHM_train.sh with one argument
#
@test "CHM_train.sh with one argument" {
  run $CHM_TRAIN "$THE_TMP"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "$CHM_TRAIN_PROG_NAME" ]
}

#
# CHM_train.sh invalid argument
#
@test "CHM_train.sh invalid argument" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -badarg foo
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "Invalid argument." ]
  [ "${lines[1]}" == "$CHM_TRAIN_PROG_NAME" ]
}

#
# CHM_train.sh with 3 arguments without - prefix
# 
@test "CHM_train.sh with 3 arguments without - prefix" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" well
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "You provided more than 2 required arguments. Did you accidently use a glob expression without escaping the asterisk?" ]
  [ "${lines[1]}" == "$CHM_TRAIN_PROG_NAME" ]
}

#
# CHM_train.sh where -m points to a nonexistant directory that cannot be created
#
@test "CHM_train.sh where -m points to a nonexistant directory that cannot be created" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -m "/dev/null"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[1]}" == "Model folder could not be created." ]
  [ "${lines[2]}" == "$CHM_TRAIN_PROG_NAME" ]
}

#
# CHM_train.sh where -S 1
#
@test "CHM_train.sh where -S 1" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -S 1
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "Invalid number of training stages." ]
  [ "${lines[1]}" == "$CHM_TRAIN_PROG_NAME" ]
}

#
# CHM_train.sh where -L 0
#
@test "CHM_train.sh where -L 0" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -L 0
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "Invalid number of training levels." ]
  [ "${lines[1]}" == "$CHM_TRAIN_PROG_NAME" ]
}


#
# CHM_train.sh with fake successful matlab and 2 basic arguments
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments" {


  # put fake CHM_train in $THE_TMP
  /bin/cp $FAKE_SUCCESS "$THE_TMP/$CHM_TRAIN_NAME" 1>&2
  chmod a+x "$THE_TMP/$CHM_TRAIN_NAME" 1>&2

  # setup fake matlab
  /bin/mkdir -p "$THE_TMP/bin/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/runtime/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/sys/os/glnxa64" 1>&2

  run $CHM_TRAIN blah alsoblah -M "$THE_TMP" 
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[0]}" == "blah alsoblah ./temp/ 2 4 0" ]
}

#
# CHM_train.sh with fake successful matlab and 2 basic arguments and -s
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and -s" {

    # put fake CHM_train in $THE_TMP
  /bin/cp $FAKE_SUCCESS "$THE_TMP/$CHM_TRAIN_NAME" 1>&2
  chmod a+x "$THE_TMP/$CHM_TRAIN_NAME" 1>&2

  # setup fake matlab
  /bin/mkdir -p "$THE_TMP/bin/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/runtime/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/sys/os/glnxa64" 1>&2


  run $CHM_TRAIN blah alsoblah -s -M "$THE_TMP"
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "Warning: argument -s is ignored for compiled version, it is always single-threaded" ]
 
}

#
# CHM_train.sh with fake fail matlab and 2 basic arguments and -s
#
@test "CHM_train.sh with fake fail matlab and 2 basic arguments and -s" {

      # put fake CHM_train in $THE_TMP
  /bin/cp $FAKE_FAIL "$THE_TMP/$CHM_TRAIN_NAME" 1>&2
  chmod a+x "$THE_TMP/$CHM_TRAIN_NAME" 1>&2

  # setup fake matlab
  /bin/mkdir -p "$THE_TMP/bin/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/runtime/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/sys/os/glnxa64" 1>&2


  run $CHM_TRAIN blah alsoblah -M "$THE_TMP"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" == "blah alsoblah ./temp/ 2 4 0" ]

}


#
# CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 4 -L 3
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 4 -L 3" {

  # put fake CHM_train in $THE_TMP
  /bin/cp $FAKE_SUCCESS "$THE_TMP/$CHM_TRAIN_NAME" 1>&2
  chmod a+x "$THE_TMP/$CHM_TRAIN_NAME" 1>&2

  # setup fake matlab
  /bin/mkdir -p "$THE_TMP/bin/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/runtime/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/sys/os/glnxa64" 1>&2


  run $CHM_TRAIN blah alsoblah -s -S 4 -L 3 -M "$THE_TMP"
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[1]}" == "blah alsoblah ./temp/ 4 3 0" ]

}

#
# CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 2 -L 1 -m -r
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 2 -L 1 -m -r" {

    # put fake CHM_train in $THE_TMP
  /bin/cp $FAKE_SUCCESS "$THE_TMP/$CHM_TRAIN_NAME" 1>&2
  chmod a+x "$THE_TMP/$CHM_TRAIN_NAME" 1>&2

  # setup fake matlab
  /bin/mkdir -p "$THE_TMP/bin/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/runtime/glnxa64" 1>&2
  /bin/mkdir -p "$THE_TMP/sys/os/glnxa64" 1>&2


  run $CHM_TRAIN blah alsoblah -s -S 2 -L 1 -m "$THE_TMP" -r -M "$THE_TMP"
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[1]}" == "blah alsoblah $THE_TMP 2 1 1" ]
}

