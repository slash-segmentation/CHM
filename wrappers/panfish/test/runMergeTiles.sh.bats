#!/usr/bin/env bats

setup() {

  export THE_TMP="${BATS_TMPDIR}/runMergeTiles" 1>&2
  mkdir -p "$THE_TMP" 1>&2
  /bin/cp ${BATS_TEST_DIRNAME}/../scripts/runMergeTiles.sh "$THE_TMP/." 1>&2
    
  export RUN_MERGE_TILES="$THE_TMP/runMergeTiles.sh"
  chmod a+x $RUN_MERGE_TILES
  unset SGE_TASK_ID
  export ORIGHELPERFUNCS="${BATS_TEST_DIRNAME}/../scripts/.helperfuncs.sh"
  
  export FAKEHELPERFUNCS="$THE_TMP/.helperfuncs.sh"
  # Make fake base helperfuncs used by some tests
  echo "
  function logEcho {
    echo \$*
    return 0
  }
  function logMessage {
    echo \$*
    return 0
  }
  function logWarning {
    echo \$*
    return 0
  }
  function jobFailed {
    echo \$*
    exit 1
  }
  function logStartTime {
    echo \$*
    return 0
  }
  function logEndTime {
    echo \$*
    return 0
  }
" > "$FAKEHELPERFUNCS"

}

teardown(){
  /bin/rm -rf "$THE_TMP"
   echo "teardown" 1>&2
}

#
#
#
@test "getMergeTilesJobParametersForTaskFromConfig() tests" {
  
  # use the actual helper funcs
  /bin/cp $ORIGHELPERFUNCS "$FAKEHELPERFUNCS" 1>&2

  # source helperfuncs.sh to we can call the function
  . $FAKEHELPERFUNCS

  # source runMergeTiles.sh to unit test functions
  . $RUN_MERGE_TILES source



  # Test where no config file exists
  run getMergeTilesJobParametersForTaskFromConfig "$THE_TMP" "1"
  [ "$status" -eq 1 ]

  # Test where there was an error getting the second line
  echo "1${CONFIG_DELIM}hello" > "$THE_TMP/$RUN_MERGE_TILES_CONFIG"
  run getMergeTilesJobParametersForTaskFromConfig "$THE_TMP" "1"
  [ "$status" -eq 2 ]


  # Test success
  echo "1${CONFIG_DELIM}hello" > "$THE_TMP/$RUN_MERGE_TILES_CONFIG"
  echo "1${CONFIG_DELIM}bye" >> "$THE_TMP/$RUN_MERGE_TILES_CONFIG"
  getMergeTilesJobParametersForTaskFromConfig "$THE_TMP" "1"
  [ "$?" -eq 0 ]

  [ "$INPUT_TILE_DIR" == "hello" ]
  [ "$OUTPUT_IMAGE" == "bye" ]

}



#
# no .helperfuncs.sh 
#
@test "no .helperfuncs.sh" {
  export SGE_TASK_ID=1
  /bin/rm -f "$THE_TMP/.helperfuncs.sh" 1>&2
  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "$THE_TMP/.helperfuncs.sh not found" ]
}

#
# error parsing properties
#
@test "error parsing properties" {
echo "
RM_CMD=/bin/rm
function parseProperties {
  MATLAB_DIR=$THE_TMP
  return 1
}
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "Error parsing properties" ]


}

#
# SGE_TASK_ID not set
#
@test "SGE_TASK_ID not set" {

  unset SGE_TASK_ID
echo "
RM_CMD=/bin/rm
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "This script merges image tiles for a slice/image of data." ]
}

#
# no config file found
#
@test "no config file found" {


  export SGE_TASK_ID=1
echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RM_CMD=/bin/rm
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
" >> "$THE_TMP/.helperfuncs.sh"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "No $THE_TMP/runMergeTiles.sh.config found" ]
}

# Error getting merge tiles job parameters for task
@test "Error getting merge tiles job parameters for task" {


  export SGE_TASK_ID=1
echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RUN_MERGE_TILES_SH=runMergeTiles
RM_CMD=/bin/rm
START_TIME=0
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getMergeTilesJobParametersForTaskFromConfig {
    echo \$*
    return 1
  }
" >> "$THE_TMP/.helperfuncs.sh"
  echo "hi" > "$THE_TMP/runMergeTiles.sh.config"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 1 ]
  [ "${lines[0]}" == "runMergeTiles" ]
  [ "${lines[1]}" == "$THE_TMP 1" ]
  [ "${lines[2]}" == "runMergeTiles 0 1" ]

}


# No tiles to merge
@test "No tiles to merge" {


  unset PANFISH_BASEDIR
  unset PANFISH_SCRATCH
  export SGE_TASK_ID=1

echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RUN_MERGE_TILES_SH=runMergeTiles
START_TIME=0
RM_CMD=/bin/rm
CONVERT_CMD=/bin/echo
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getMergeTilesJobParametersForTaskFromConfig {
    echo \$*
    INPUT_TILE_DIR="$THE_TMP/tiles"
    OUTPUT_IMAGE="foo.png"
    return 0
  }
  function makeDirectory {
    mkdir \$1
    return \$?
  }
   
" >> "$THE_TMP/.helperfuncs.sh"
  echo "hi" > "$THE_TMP/runMergeTiles.sh.config"

  mkdir -p "$THE_TMP/tiles" 1>&2

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 9 ]
  [ "${lines[0]}" == "runMergeTiles" ]
  [ "${lines[1]}" == "$THE_TMP 1" ]
  [ "${lines[2]}" == "Found 0 tiles" ]
  [ "${lines[3]}" == "No tiles to merge" ]
  [ "${lines[4]}" == "runMergeTiles 0 9" ]

}


# 1 tile to merge
@test "1 tile to merge" {


  unset PANFISH_BASEDIR
  unset PANFISH_SCRATCH
  export SGE_TASK_ID=1

echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RUN_MERGE_TILES_SH=runMergeTiles
START_TIME=0
RM_CMD=/bin/rm
CP_CMD=/bin/echo
CONVERT_CMD=/bin/echo
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getMergeTilesJobParametersForTaskFromConfig {
    echo \$*
    INPUT_TILE_DIR="$THE_TMP/tiles"
    OUTPUT_IMAGE="foo.png"
    return 0
  }
  function makeDirectory {
    mkdir \$1
    return \$?
  }

" >> "$THE_TMP/.helperfuncs.sh"
  echo "hi" > "$THE_TMP/runMergeTiles.sh.config"

  mkdir -p "$THE_TMP/tiles" 1>&2
  echo "hi" > "$THE_TMP/tiles/1.png"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "runMergeTiles" ]
  [ "${lines[1]}" == "$THE_TMP 1" ]
  [ "${lines[2]}" == "Found 1 tiles" ]
  [ "${lines[3]}" == "/$THE_TMP/tiles/1.png $THE_TMP/foo.png" ]
  [ "${lines[4]}" == "runMergeTiles 0 0" ]
}


# multiple tiles to merge no file to copy
@test "multiple tiles to merge no file to copy" {


  unset PANFISH_BASEDIR
  unset PANFISH_SCRATCH
  export SGE_TASK_ID=1

echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RUN_MERGE_TILES_SH=runMergeTiles
START_TIME=0
RM_CMD=/bin/rm
CP_CMD=/bin/echo
TIME_V_CMD=/bin/echo
CONVERT_CMD=convert
UUIDGEN_CMD=\"/bin/echo 1\"
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getMergeTilesJobParametersForTaskFromConfig {
    echo \$*
    INPUT_TILE_DIR="$THE_TMP/tiles"
    OUTPUT_IMAGE="foo.png"
    return 0
  }
  function makeDirectory {
    mkdir \$1
    return \$?
  }

" >> "$THE_TMP/.helperfuncs.sh"
  echo "hi" > "$THE_TMP/runMergeTiles.sh.config"

  mkdir -p "$THE_TMP/tiles" 1>&2
  echo "hi" > "$THE_TMP/tiles/1.png"
  echo "hi" > "$THE_TMP/tiles/2.png"
  echo "hi" > "$THE_TMP/tiles/3.png"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 2 ]
  [ "${lines[0]}" == "runMergeTiles" ]
  [ "${lines[1]}" == "$THE_TMP 1" ]
  [ "${lines[2]}" == "Found 3 tiles" ]
  [[ "${lines[3]}" == "Running convert /$THE_TMP/tiles/1.png -compose plus /$THE_TMP/tiles/2.png -composite /$THE_TMP/tiles/3.png -composite /tmp/"* ]]
  [ "${lines[4]}" == "convert /$THE_TMP/tiles/1.png -compose plus /$THE_TMP/tiles/2.png -composite /$THE_TMP/tiles/3.png -composite /tmp/mergetiles.1/foo.png" ]
  [ "${lines[5]}" == "No /tmp/mergetiles.1/foo.png found to copy" ]
  [ "${lines[6]}" == "runMergeTiles 0 2" ]
}


# multiple files to merge
@test "multiple tiles to merge" {


  unset PANFISH_BASEDIR
  unset PANFISH_SCRATCH
  export SGE_TASK_ID=1

echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RUN_MERGE_TILES_SH=runMergeTiles
START_TIME=0
RM_CMD=/bin/rm
CP_CMD=/bin/echo
TIME_V_CMD=/bin/echo
CONVERT_CMD=convert
PANFISH_SCRATCH=\"\$THE_TMP/hi\"
UUIDGEN_CMD=\"/bin/echo 1\"
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getMergeTilesJobParametersForTaskFromConfig {
    echo \$*
    INPUT_TILE_DIR="$THE_TMP/tiles"
    OUTPUT_IMAGE="foo.png"
    return 0
  }
  function makeDirectory {
    mkdir \$1
    return \$?
  }

" >> "$THE_TMP/.helperfuncs.sh"
  echo "hi" > "$THE_TMP/runMergeTiles.sh.config"
  mkdir -p "$THE_TMP/hi/mergetiles.1" 1>&2
  echo "image" > "$THE_TMP/hi/mergetiles.1/foo.png"

  mkdir -p "$THE_TMP/tiles" 1>&2
  echo "hi" > "$THE_TMP/tiles/1.png"
  echo "hi" > "$THE_TMP/tiles/2.png"
  echo "hi" > "$THE_TMP/tiles/3.png"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "runMergeTiles" ]
  [ "${lines[1]}" == "$THE_TMP 1" ]
  [ "${lines[3]}" == "Found 3 tiles" ]
  [ "${lines[4]}" == "Running convert /$THE_TMP/tiles/1.png -compose plus /$THE_TMP/tiles/2.png -composite /$THE_TMP/tiles/3.png -composite $THE_TMP/hi/mergetiles.1/foo.png" ]
  [ "${lines[5]}" == "convert /$THE_TMP/tiles/1.png -compose plus /$THE_TMP/tiles/2.png -composite /$THE_TMP/tiles/3.png -composite $THE_TMP/hi/mergetiles.1/foo.png" ]
  [ "${lines[6]}" == "-f $THE_TMP/hi/mergetiles.1/foo.png $THE_TMP/foo.png" ]
  [ "${lines[7]}" == "runMergeTiles 0 0" ]
}

# Error running convert
@test "error running convert" {


  unset PANFISH_BASEDIR
  unset PANFISH_SCRATCH
  export SGE_TASK_ID=1

echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RUN_MERGE_TILES_SH=runMergeTiles
START_TIME=0
RM_CMD=/bin/rm
CP_CMD=/bin/echo
TIME_V_CMD=/bin/false
CONVERT_CMD=/bin/false
PANFISH_SCRATCH=\"\$THE_TMP/hi\"
UUIDGEN_CMD=\"/bin/echo 1\"
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getMergeTilesJobParametersForTaskFromConfig {
    echo \$*
    INPUT_TILE_DIR="$THE_TMP/tiles"
    OUTPUT_IMAGE="foo.png"
    return 0
  }
  function makeDirectory {
    mkdir \$1
    return \$?
  }

" >> "$THE_TMP/.helperfuncs.sh"
  echo "hi" > "$THE_TMP/runMergeTiles.sh.config"

  mkdir -p "$THE_TMP/tiles" 1>&2
  echo "hi" > "$THE_TMP/tiles/1.png"
  echo "hi" > "$THE_TMP/tiles/2.png"
  echo "hi" > "$THE_TMP/tiles/3.png"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 3 ]
  [ "${lines[0]}" == "runMergeTiles" ]
  [ "${lines[1]}" == "$THE_TMP 1" ]
  [ "${lines[3]}" == "Found 3 tiles" ]
  [ "${lines[4]}" == "Running /bin/false /$THE_TMP/tiles/1.png -compose plus /$THE_TMP/tiles/2.png -composite /$THE_TMP/tiles/3.png -composite $THE_TMP/hi/mergetiles.1/foo.png" ]
  [ "${lines[5]}" == "No $THE_TMP/hi/mergetiles.1/foo.png found to copy" ]
  [ "${lines[6]}" == "runMergeTiles 0 3" ]
}


# PANFISH_BASEDIR and PANFISH_SCRATCH set
@test "multiple tiles to merge PANFISH_BASEDIR and PANFISH_SCRATCH set" {


  unset PANFISH_BASEDIR
  unset PANFISH_SCRATCH
  export SGE_TASK_ID=1

echo "RUN_MERGE_TILES_CONFIG=runMergeTiles.sh.config
RUN_MERGE_TILES_SH=runMergeTiles
START_TIME=0
RM_CMD=/bin/rm
CP_CMD=/bin/echo
TIME_V_CMD=/bin/echo
CONVERT_CMD=convert
PANFISH_SCRATCH=\"\$THE_TMP/hi\"
PANFISH_BASEDIR=\"$THE_TMP/pan\"
UUIDGEN_CMD=\"/bin/echo 1\"
  function parseProperties {
   MATLAB_DIR=$THE_TMP
   return 0
  }
  function getMergeTilesJobParametersForTaskFromConfig {
    echo \$*
    INPUT_TILE_DIR="tiles"
    OUTPUT_IMAGE="foo.png"
    return 0
  }
  function makeDirectory {
    mkdir \$1
    return \$?
  }

" >> "$THE_TMP/.helperfuncs.sh"
  echo "hi" > "$THE_TMP/runMergeTiles.sh.config"
  mkdir -p "$THE_TMP/hi/mergetiles.1" 1>&2
  echo "image" > "$THE_TMP/hi/mergetiles.1/foo.png"

  mkdir -p "$THE_TMP/pan/tiles" 1>&2
  echo "hi" > "$THE_TMP/pan/tiles/1.png"
  echo "hi" > "$THE_TMP/pan/tiles/2.png"
  echo "hi" > "$THE_TMP/pan/tiles/3.png"

  run $RUN_MERGE_TILES
  echo "$output" 1>&2
  [ "$status" -eq 0 ]
  [ "${lines[0]}" == "runMergeTiles" ]
  [ "${lines[1]}" == "$THE_TMP 1" ]
  [ "${lines[3]}" == "Found 3 tiles" ]
  [ "${lines[4]}" == "Running convert $THE_TMP/pan/tiles/1.png -compose plus $THE_TMP/pan/tiles/2.png -composite $THE_TMP/pan/tiles/3.png -composite $THE_TMP/hi/mergetiles.1/foo.png" ]
  [ "${lines[5]}" == "convert $THE_TMP/pan/tiles/1.png -compose plus $THE_TMP/pan/tiles/2.png -composite $THE_TMP/pan/tiles/3.png -composite $THE_TMP/hi/mergetiles.1/foo.png" ]
  [ "${lines[6]}" == "-f $THE_TMP/hi/mergetiles.1/foo.png $THE_TMP/foo.png" ]
  [ "${lines[7]}" == "runMergeTiles 0 0" ]
}

