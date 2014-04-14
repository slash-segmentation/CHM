#!/usr/bin/env bats

setup() {
  CHM_TEST="${BATS_TEST_DIRNAME}/../algorithm/CHM_test.sh"
  CHM_TRAIN="${BATS_TEST_DIRNAME}/../algorithm/CHM_train.sh"
  export THE_TMP="${BATS_TMPDIR}/chmtrain."`uuidgen`
  /bin/mkdir -p $THE_TMP
  export SUCCESS_MATLAB="$BATS_TEST_DIRNAME/bin/fakesuccessmatlab"
  export FAIL_MATLAB="$BATS_TEST_DIRNAME/bin/fakefailmatlab"
  chmod a+x $SUCCESS_MATLAB/matlab
  chmod a+x $FAIL_MATLAB/matlab

  export FAKE_NPROC="$BATS_TEST_DIRNAME/bin/fakenproc"
  chmod a+x $FAKE_NPROC
  curdir=`pwd`
  cd $THE_TMP
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
  [ "${lines[0]}" = "CHM Image Training Phase Script." ]
  [ "${lines[1]}" = "$CHM_TRAIN <inputs> <labels> <optional arguments>" ]
}

#
# CHM_train.sh with one argument
#
@test "CHM_train.sh with one argument" {
  run $CHM_TRAIN "$THE_TMP"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "CHM Image Training Phase Script." ]
  [ "${lines[1]}" = "$CHM_TRAIN <inputs> <labels> <optional arguments>" ]
}

#
# CHM_train.sh invalid argument
#
@test "CHM_train.sh invalid argument" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -badarg foo
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid argument." ]
  [ "${lines[1]}" = "CHM Image Training Phase Script." ]
  [ "${lines[2]}" = "$CHM_TRAIN <inputs> <labels> <optional arguments>" ]
}


#
# CHM_train.sh where -m points to a nonexistant directory that cannot be created
#
@test "CHM_train.sh where -m points to a nonexistant directory that cannot be created" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -m "/dev/null"
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[1]}" = "Model folder could not be created." ]
  [ "${lines[2]}" = "CHM Image Training Phase Script." ]
  [ "${lines[3]}" = "$CHM_TRAIN <inputs> <labels> <optional arguments>" ]
}

#
# CHM_train.sh where -S 1
#
@test "CHM_train.sh where -S 1" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -S 1
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid number of training stages." ]
  [ "${lines[1]}" = "CHM Image Training Phase Script." ]
  [ "${lines[2]}" = "$CHM_TRAIN <inputs> <labels> <optional arguments>" ]
}

#
# CHM_train.sh where -L 0
#
@test "CHM_train.sh where -L 0" {
  run $CHM_TRAIN "$THE_TMP" "$THE_TMP" -L 0
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid number of training levels." ]
  [ "${lines[1]}" = "CHM Image Training Phase Script." ]
  [ "${lines[2]}" = "$CHM_TRAIN <inputs> <labels> <optional arguments>" ]
}


#
# CHM_train.sh with fake successful matlab and 2 basic arguments and nproc 2
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and nproc 2" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$FAKE_NPROC:$PATH

  # set proc count to 2
  export FAKE_NPROC_PROCS=2

  run $CHM_TRAIN blah alsoblah 
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[0]}" = "-nodisplay -r run_from_shell('CHM_train(''blah'',''alsoblah'',''./temp/'',2,4,0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_train.sh with fake successful matlab and 2 basic arguments and nproc 25
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and nproc 25" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$FAKE_NPROC:$PATH

  # set proc count to 2
  export FAKE_NPROC_PROCS=25


  run $CHM_TRAIN blah alsoblah
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_train(''blah'',''alsoblah'',''./temp/'',2,4,0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_train.sh with fake successful matlab and 2 basic arguments -s and nproc 25
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments -s and nproc 25" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$FAKE_NPROC:$PATH

  # set proc count to 2
  export FAKE_NPROC_PROCS=25


  run $CHM_TRAIN blah alsoblah -s
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[0]}" = "-nodisplay -nojvm -singleCompThread -r run_from_shell('CHM_train(''blah'',''alsoblah'',''./temp/'',2,4,0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}



#
# CHM_train.sh with fake successful matlab and 2 basic arguments and -s
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and -s" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TRAIN blah alsoblah -s
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  
  [ "${lines[0]}" = "-nodisplay -nojvm -singleCompThread -r run_from_shell('CHM_train(''blah'',''alsoblah'',''./temp/'',2,4,0);');" ]
 
  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_train.sh with fake fail matlab and 2 basic arguments and -s
#
@test "CHM_train.sh with fake fail matlab and 2 basic arguments and -s" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$FAIL_MATLAB:$PATH

  run $CHM_TRAIN blah alsoblah -s
  echo "$output" 1>&2
  [ "$status" -eq 1 ]

  [ "${lines[0]}" = "-nodisplay -nojvm -singleCompThread -r run_from_shell('CHM_train(''blah'',''alsoblah'',''./temp/'',2,4,0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}


#
# CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 4 -L 3
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 4 -L 3" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TRAIN blah alsoblah -s -S 4 -L 3
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[0]}" = "-nodisplay -nojvm -singleCompThread -r run_from_shell('CHM_train(''blah'',''alsoblah'',''./temp/'',4,3,0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 2 -L 1 -m -r
#
@test "CHM_train.sh with fake successful matlab and 2 basic arguments and -s -S 2 -L 1 -m -r" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TRAIN blah alsoblah -s -S 2 -L 1 -m "$THE_TMP" -r
  echo "$output" 1>&2
  [ "$status" -eq 0 ]

  [ "${lines[0]}" = "-nodisplay -nojvm -singleCompThread -r run_from_shell('CHM_train(''blah'',''alsoblah'',''$THE_TMP'',2,1,1);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

