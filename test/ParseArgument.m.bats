#!/usr/bin/env bats

setup() { 
  curdir=`pwd`
  cd "${BATS_TEST_DIRNAME}/../algorithm"
}

teardown() {
  cd $curdir
  stty sane >/dev/null 2>&1 # restore terminal settings

}
#
# ParseArgument(true)
#
@test "ParseArgument(true)" {
  
  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi  

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(''true'')');"
  echo "$output" 1>&2
  [ "${lines[*]: -3:1}" == "     1" ]
  [ "$status" -eq 0 ] 
}

#
# ParseArgument(t)
#
@test "ParseArgument(t)" {
  
  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(''t'')');"
  echo "$output" 1>&2
  [ "${lines[*]: -3:1}" == "     1" ]
  [ "$status" -eq 0 ]
}

#
# ParseArgument(false)
#
@test "ParseArgument(false)" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(''false'')');"
  echo "$output" 1>&2
  [ "${lines[*]: -3:1}" == "     0" ]
  [ "$status" -eq 0 ]
}

#
# ParseArgument(f)
#
@test "ParseArgument(f)" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(''f'')');"
  echo "$output" 1>&2
  [ "${lines[*]: -3:1}" == "     0" ]
  [ "$status" -eq 0 ]
}


#
# ParseArgument(3)
#
@test "ParseArgument(3)" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(3)');"
  echo "$output" 1>&2
  [ "${lines[*]: -3:1}" == "     3" ]
  [ "$status" -eq 0 ]
}

#
# ParseArgument(-5.5)
#
@test "ParseArgument(-5.5)" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(-5.5)');"
  echo "$output" 1>&2
  [[ "${lines[*]: -3:1}" == "   -5.5"* ]]
  [ "$status" -eq 0 ]
}

#
# ParseArgument([2 4])
#
@test "ParseArgument([2 4])" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(''[2 4]'')');"
  echo "$output" 1>&2
  echo ":${lines[*]: -3:1}:" 1>&2
  [ "${lines[*]: -3:1}" == "     2     4" ]
  [ "$status" -eq 0 ]
}

#
# ParseArgument([2 4;5,-1])
#
@test "ParseArgument([2 4;5,-1])" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('ParseArgument(''[2 4;5,-1]'')');"
  echo "$output" 1>&2
  echo ":${lines[*]: -3:1}:" 1>&2
  [ "${lines[*]: -4:1}" == "     2     4" ]
  [ "${lines[*]: -3:1}" == "     5    -1" ]
  [ "$status" -eq 0 ]
}


