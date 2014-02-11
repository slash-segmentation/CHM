#!/usr/bin/env bats

setup() {
  CHM_TEST="${BATS_TEST_DIRNAME}/../algorithm/CHM_test.sh"
  export THE_TMP="${BATS_TMPDIR}/"`uuidgen`
  /bin/mkdir -p $THE_TMP
  export SUCCESS_MATLAB="$BATS_TEST_DIRNAME/bin/fakesuccessmatlab"
  export FAIL_MATLAB="$BATS_TEST_DIRNAME/bin/fakefailmatlab"
  chmod a+x $SUCCESS_MATLAB/matlab
  chmod a+x $FAIL_MATLAB/matlab
}

teardown() {
  /bin/rmdir $THE_TMP
}
#
# Test CHM_test with no arguments
#
@test "CHM_test.sh with no arguments" {
  run $CHM_TEST

  [ "$status" -eq 1 ] 
  [ "${lines[0]}" = "CHM Image Testing Phase Script." ]
  [ "${lines[2]}" = "$CHM_TEST <input_files> <output_folder> <optional arguments>" ]
}

#
# CHM_test.sh with one argument
#
@test "CHM_test.sh with one argument" {
  run $CHM_TEST foo

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "CHM Image Testing Phase Script." ]
  [ "${lines[2]}" = "$CHM_TEST <input_files> <output_folder> <optional arguments>" ]
}

#
# CHM_test.sh where output folder already exists as a file
#
@test "CHM_test.sh where output folder already exists as a file" {
  
  fakeFile="${THE_TMP}/1"
  run touch $fakeFile
  [ "$status" -eq 0 ]

  run $CHM_TEST blah $fakeFile

  [ "$status" -eq 1 ] 

  [ "${lines[0]}" = "Output directory already exists as a file." ]
  [ "${lines[1]}" = "CHM Image Testing Phase Script." ]
  run rm -f $fakeFile

  [ "$status" -eq 0 ]

}

#
# CHM_test.sh where model folder is not a directory
#
@test "CHM_test.sh where model folder is not a directory" {
  fakeFile="${THE_TMP}/1"
  run touch $fakeFile
  [ "$status" -eq 0 ]

  run $CHM_TEST blah ${THE_TMP} -m $fakeFile

  [ "$status" -eq 1 ]

  [ "${lines[0]}" = "Model folder is not a directory." ]
  [ "${lines[1]}" = "CHM Image Testing Phase Script." ]
  run rm -f $fakeFile

  [ "$status" -eq 0 ]

}

#
# CHM_test.sh with invalid argument
#
@test "CHM_test.sh with invalid argument" {
  run $CHM_TEST blah ${THE_TMP} -invalidarg
  [ "$status" -eq 1 ]

  [ "${lines[0]}" = "Invalid argument." ]
  [ "${lines[1]}" = "CHM Image Testing Phase Script." ]
}

#
# CHM_test.sh with fake successful matlab and 2 basic arguments
#
@test "CHM_test.sh with fake successful matlab and 2 basic arguments" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP}

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''./temp/'');');" ]
 
  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with fake fail matlab and 2 basic arguments
#
@test "CHM_test.sh with fake fail matlab and 2 basic arguments" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$FAIL_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP}

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''./temp/'');');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

# 
# CHM_test.sh with fake successful matlab and block size argument
#
@test "CHM_test.sh with fake successful matlab and block size argument" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 200x100

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test_blocks(''blah'',''$THE_TMP'',[100 200],[0 0],''./temp/'');');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

# 
# CHM_test.sh with fake fail matlab and block size argument
#

@test "CHM_test.sh with fake fail matlab and block size argument" { 
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$FAIL_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 200x100

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test_blocks(''blah'',''$THE_TMP'',[100 200],[0 0],''./temp/'');');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

# 
# CHM_test.sh with fake successful matlab and block size and overlap argument
# 
@test "CHM_test.sh with fake successful matlab and block size and overlap argument" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$FAIL_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 200x100 -o 3x1

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test_blocks(''blah'',''$THE_TMP'',[100 200],[1 3],''./temp/'');');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with overlap argument but no block size argument
#
@test "CHM_test.sh with overlap argument but no block size argument" {

  run $CHM_TEST blah ${THE_TMP} -o 3x1

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Overlap size can only be used with block size." ]
  [ "${lines[1]}" = "CHM Image Testing Phase Script." ]
}

#
# CHM_test.sh with invalid block size
#
@test "CHM_test.sh with invalid block size" {

  run $CHM_TEST blah ${THE_TMP} -b 10x-1

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid block size." ]
  [ "${lines[1]}" = "CHM Image Testing Phase Script." ]
}


#
# CHM_test.sh with invalid overlap
#
@test "CHM_test.sh with invalid block size" {

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -o -1x2

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid overlap size." ]
  [ "${lines[1]}" = "CHM Image Testing Phase Script." ]

}



