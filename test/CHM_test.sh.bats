#!/usr/bin/env bats

setup() {
  CHM_TEST="${BATS_TEST_DIRNAME}/../algorithm/CHM_test.sh"
  export THE_TMP="${BATS_TMPDIR}/"`uuidgen`
  /bin/mkdir -p $THE_TMP
  export SUCCESS_MATLAB="$BATS_TEST_DIRNAME/bin/fakesuccessmatlab"
  export FAIL_MATLAB="$BATS_TEST_DIRNAME/bin/fakefailmatlab"
  chmod a+x $SUCCESS_MATLAB/matlab
  chmod a+x $FAIL_MATLAB/matlab
  export CHM_TEST_PROG_NAME="CHM Image Testing Phase Script.  @@VERSION@@"
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
  [ "${lines[0]}" = "$CHM_TEST_PROG_NAME" ]
  [ "${lines[2]}" = "$CHM_TEST <input_files> <output_folder> <optional arguments>" ]
}

#
# CHM_test.sh with one argument
#
@test "CHM_test.sh with one argument" {
  run $CHM_TEST foo

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "$CHM_TEST_PROG_NAME" ]
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
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
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
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
  run rm -f $fakeFile

  [ "$status" -eq 0 ]

}

#
# CHM_test.sh with invalid argument
#
@test "CHM_test.sh with invalid argument" {
  run $CHM_TEST blah ${THE_TMP} -invalidarg
  [ "$status" -eq 1 ]
  
  # this is a little weird.  Looking at CHM_test.sh code I'd expect -invalidarg, but
  # we are getting ? character.
  [ "${lines[0]}" = "Invalid argument: ?." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
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
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''auto'',[50 50],''./temp/'',[],''true'',0);');" ]
 
  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test with fake successful matlab 2 basic arguments and -h
#
@test "CHM_test.sh with fake successful matlab and 2 basic arguments and -h" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -h

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''auto'',[50 50],''./temp/'',[],''false'',0);');" ]

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
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''auto'',[50 50],''./temp/'',[],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with basic arguments and -s
#
@test "CHM_test.sh with basic arguments and -s" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -s

  [ "$status" -eq 0 ]
[ "${lines[0]}" = "-nodisplay -singleCompThread -nojvm -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''auto'',[50 50],''./temp/'',[],''true'',1);');" ]

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
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[100 200],[50 50],''./temp/'',[],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

# 
# CHM_test.sh with fake successful matlab and block size argument and -s
#
@test "CHM_test.sh with fake successful matlab and block size argument and -s" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -s -b 200x100
  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -nojvm -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[100 200],[50 50],''./temp/'',[],''true'',1);');" ]

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
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[100 200],[50 50],''./temp/'',[],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

# 
# CHM_test.sh with fake successful matlab and block size and overlap argument
# 
@test "CHM_test.sh with fake successful matlab and block size and overlap argument" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 200x100 -o 3x1

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[100 200],[1 3],''./temp/'',[],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with fake successful matlab and single valid value for block size
#
@test "CHM_test.sh with fake successful matlab and single valid value for block size" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 50 
  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[50 50],[50 50],''./temp/'',[],''true'',0);');" ]
  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with fake successful matlab and single valid value for overlap size
#
@test "CHM_test.sh with fake successful matlab and single valid value for overlap size" {
  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 50 -o 20
  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[50 50],[20 20],''./temp/'',[],''true'',0);');" ]
  # reset path
  export PATH=$A_TEMP_PATH
}


#
# CHM_test.sh with overlap argument but no block size argument
#
@test "CHM_test.sh with overlap argument but no block size argument" {

  A_TEMP_PATH=$PATH
  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH
  

  run $CHM_TEST blah ${THE_TMP} -o 3x1
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''auto'',[1 3],''./temp/'',[],''true'',0);');" ]

  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with tile argument but no block size argument
#
@test "CHM_test.sh with tile argument but no block size argument" {

  A_TEMP_PATH=$PATH
  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -t 3,1
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',''auto'',[50 50],''./temp/'',[1 3],''true'',0);');" ]

  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with invalid block size
#
@test "CHM_test.sh with invalid block size 10x-1" {

  run $CHM_TEST blah ${THE_TMP} -b 10x-1
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid size: 10x-1. Expecting single number or WxH." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
}

#
# CHM_test.sh with invalid block size
#
@test "CHM_test.sh with invalid block size -1x300" {

  run $CHM_TEST blah ${THE_TMP} -b -1x300
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid size: -1x300. Expecting single number or WxH." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
}


#
# CHM_test.sh with invalid block size
#
@test "CHM_test.sh with invalid block size 10x0" {

  run $CHM_TEST blah ${THE_TMP} -b 10x0
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid size: 10x0. Neither dimension can be 0." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
}

#
# CHM_test.sh with invalid block size
#
@test "CHM_test.sh with invalid block size 0x20" {

  run $CHM_TEST blah ${THE_TMP} -b 0x20
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid size: 0x20. Neither dimension can be 0." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
}

#
# CHM_test.sh with invalid block size
#
@test "CHM_test.sh with invalid block size 0x0" {

  run $CHM_TEST blah ${THE_TMP} -b 0x0
  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid size: 0x0. Neither dimension can be 0." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]
}

#
# CHM_test.sh with invalid overlap -1x2
#
@test "CHM_test.sh with invalid overlap size -1x2" {

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -o -1x2

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid size: -1x2. Expecting single number or WxH." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]

}

#
# CHM_test.sh with invalid overlap 2x-1
#
@test "CHM_test.sh with invalid overlap size 2x-1" {

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -o 2x-1

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid size: 2x-1. Expecting single number or WxH." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]

}


#
# CHM_test.sh with valid overlap 0x2
#
@test "CHM_test.sh with valid overlap size 0x2" {

  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -o 0x2

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[1 10],[2 0],''./temp/'',[],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH

}

#
# CHM_test.sh with valid overlap 2x0
#
@test "CHM_test.sh with valid overlap size 2x0" {

  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -o 2x0

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[1 10],[0 2],''./temp/'',[],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh with valid overlap 0x0
#
@test "CHM_test.sh with valid overlap size 0x0" {

  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -o 0x0

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[1 10],[0 0],''./temp/'',[],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH
}

#
# CHM_test.sh invalid tile -t 4
#
@test "CHM_test.sh invalid tile -t 4" {

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -t 4

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid position: 4. Expecting COL,ROW." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]

}

#
# CHM_test.sh invalid tile -t 4x4
#
@test "CHM_test.sh invalid tile -t 4x4" {

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -t 4x4

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid position: 4x4. Expecting COL,ROW." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]

}

#
# CHM_test.sh good tile -t 5,6 and a bad tile -t foo
#
@test "CHM_test.sh good tile -t 5,6 and a bad tile -t foo" {

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -t 5,6 -t foo

  [ "$status" -eq 1 ]
  [ "${lines[0]}" = "Invalid position: foo. Expecting COL,ROW." ]
  [ "${lines[1]}" = "$CHM_TEST_PROG_NAME" ]

}

#
# CHM_test.sh valid tile -t 0,0
#
@test "CHM_test.sh valid tile -t 0,0" {

  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -t 0,0

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[1 10],[50 50],''./temp/'',[0 0],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH

}

#
# CHM_test.sh valid tile -t -1,-1
#
@test "CHM_test.sh valid tile -t -1,-1" {

  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -t -1,-1

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[1 10],[50 50],''./temp/'',[-1 -1],''true'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH

}



#
# CHM_test.sh multiple valid tiles with repeat -t 1,1 -t 2,3 -t 1,1
#
@test "CHM_test.sh multiple valid tiles with repeat -t 1,1 -t 2,3 -t 1,1 -h" {

  A_TEMP_PATH=$PATH

  # put fake matlab in path
  export PATH=$SUCCESS_MATLAB:$PATH

  run $CHM_TEST blah ${THE_TMP} -b 10x1 -t 1,1 -t 2,3 -t 1,1 -h

  [ "$status" -eq 0 ]
  [ "${lines[0]}" = "-nodisplay -singleCompThread -r run_from_shell('CHM_test(''blah'',''$THE_TMP'',[1 10],[50 50],''./temp/'',[1 1;3 2;1 1],''false'',0);');" ]

  # reset path
  export PATH=$A_TEMP_PATH

}

