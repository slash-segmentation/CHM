#!/usr/bin/env bats

setup() { 
  curdir=`pwd`
  cd "${BATS_TEST_DIRNAME}/../algorithm"
  export THE_TMP="${BATS_TMPDIR}/GetInputFiles."`uuidgen`
  mkdir -p "$THE_TMP" 1>&2
}

teardown() {
  cd $curdir
  /bin/rm -rf "$THE_TMP" 1>&2
  #  stty sane >/dev/null 2>&1 # restore terminal settings

}
#
# GetInputFiles(''GetInputFiles.m'')
#
@test "GetInputFiles(''GetInputFiles.m'')" {
  
  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi  

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('GetInputFiles(''GetInputFiles.m'')');"
  echo "$output" 1>&2
  [ "${lines[*]: -3:1}" == "    'GetInputFiles.m'" ]
  [ "$status" -eq 0 ] 
}

#
# GetInputFiles(/tmp/nonexistdir)
#
@test "GetInputFiles(''/tmp/nonexistdir'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('GetInputFiles(''/tmp/nonexistdirxasdf'')');"
  echo "$output" 1>&2
  [ "${lines[*]: -5:1}" == "No such file \"/tmp/nonexistdirxasdf\"" ]
  [ "${lines[*]: -3:1}" == "   Empty cell array: 1-by-0" ]
  [ "$status" -eq 0 ]
}

#
# GetInputFiles.m(/dirofpng)  -- 1 png
#
@test "GetInputFiles(''/dirwith1png'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  touch "$THE_TMP/1.png"

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('GetInputFiles(''$THE_TMP'')');"
  echo "$output" 1>&2
  
  [[ "${lines[*]: -3:1}" == *"'$THE_TMP/1.png'" ]]
  [ "$status" -eq 0 ]
}

#
# GetInputFiles.m(/dirofpng)  -- multiple png
#
@test "GetInputFiles(''/dirwith2png'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  touch "$THE_TMP/1.png"
  touch "$THE_TMP/2.png"

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('out=GetInputFiles(''$THE_TMP'');disp(strjoin(out));');"
  echo "$output" 1>&2
  
  [[ "${lines[*]: -3:1}" == *"$THE_TMP/1.png $THE_TMP/2.png" ]]
  [ "$status" -eq 0 ]
}

#
# GetInputFiles.m(/dirofPNG)  -- multiple PNG (alternate case)
#
@test "GetInputFiles(''/dirwith2PNG'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  touch "$THE_TMP/1.PNG"
  touch "$THE_TMP/2.PNG"

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('out=GetInputFiles(''$THE_TMP'');disp(strjoin(out));');"
  echo "$output" 1>&2

  [[ "${lines[*]: -3:1}" == *"$THE_TMP/1.PNG $THE_TMP/2.PNG" ]]
  [ "$status" -eq 0 ]
}

#
# GetInputFiles.m(/diroftif)  -- 1 tif
#
@test "GetInputFiles(''/dirwith1tif'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  touch "$THE_TMP/1.tif"

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('out=GetInputFiles(''$THE_TMP'');disp(out);');"
  echo "$output" 1>&2
  
  [[ "${lines[*]: -3:1}" == *"'$THE_TMP/1.tif'" ]]
  [ "$status" -eq 0 ]
}

#
# GetInputFiles.m(/diroftif)  -- multiple tif
#
@test "GetInputFiles(''/dirwith2tif'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  touch "$THE_TMP/1.tif"
  touch "$THE_TMP/2.tif"

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('out=GetInputFiles(''$THE_TMP'');disp(strjoin(out));');"
  echo "$output" 1>&2
 
  [[ "${lines[*]: -3:1}" == *"$THE_TMP/1.tif $THE_TMP/2.tif" ]]
  [ "$status" -eq 0 ]
}


#
# GetInputFiles.m(/diroftiff)  -- 1 tiff
#
@test "GetInputFiles(''/dirwith1tiff'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  touch "$THE_TMP/1.tiff"

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('out=GetInputFiles(''$THE_TMP'');disp(out);');"
  echo "$output" 1>&2
  
  [[ "${lines[*]: -3:1}" == *"'$THE_TMP/1.tiff'" ]]
  [ "$status" -eq 0 ]
}

#
# GetInputFiles.m(/diroftiff)  -- multiple tiff
#
@test "GetInputFiles(''/dirwith2tiff'')" {

  # Verify matlab is in the users path via which command
  run which matlab

  if [ "$status" -eq 1 ] ; then
    skip "matlab not in path"
  fi

  touch "$THE_TMP/1.tiff"
  touch "$THE_TMP/2.tiff"

  run matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('out=GetInputFiles(''$THE_TMP'');disp(strjoin(out));');"
  echo "$output" 1>&2
 
  [[ "${lines[*]: -3:1}" == *"$THE_TMP/1.tiff $THE_TMP/2.tiff" ]]
  [ "$status" -eq 0 ]
}


