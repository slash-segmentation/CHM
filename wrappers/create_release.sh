#!/bin/bash

VERSION=2.1 # update manually
BUILD_NUM=`git log --pretty=format:'' | wc -l`  ## `git rev-list HEAD --count` # number of commits

SCRIPT_DIR=`dirname $0`

SCRIPT_FULLPATH_DIR="`pwd`/$SCRIPT_DIR"
echo ""
echo "Changing to $SCRIPT_FULLPATH_DIR directory"
echo ""

cd $SCRIPT_FULLPATH_DIR

if [ $? != 0 ] ; then
  echo ""
  echo "ERROR changing to $SCRIPT_FULLPATH_DIR directory"
  echo ""
  exit 1
fi

SKIP_GIT="false"

if [ $# -eq 1 ] ; then
  if [ "$1" == "-s" ] ; then
    echo "Skipping git calls... Replacing git build number: $BUILD_NUM with $BUILD_NUM.test"
    SKIP_GIT="true"
    BUILD_NUM="${BUILD_NUM}.test"
  fi
fi

VERSION=$VERSION.$BUILD_NUM

VERSION_KEYWORD="@@VERSION@@"
MATLAB_RELEASE_KEYWORD="@@MATLAB_RELEASE_VERSION@@"
MATLAB_MCR_KEYWORD="@@MATLAB_MCR_VERSION@@"

#
# Does an in place replacement of any <keyword to replace> string in
# any *.txt or *.sh in the directory path specified with <replacement value>.  
# If replacement fails function exits with code 1 otherwise 0 returned
#
# replaceKeyWord(directory path,keyword to replace,replacement value)
#
function replaceKeyWord {
  local chmDir=$1
  local tagKey=$2
  local vTag=$3
  for y in `echo "txt sh"` ; do 
    for z in `find "$chmDir" -name "*.${y}" -type f` ; do
      replaceKeyWordInFile "$z" "$tagKey" "$vTag"
      if [ $? != 0 ] ; then
        echo "Error replacing $tagKey in file $z"
        exit 1
      fi
    done
  done
  return 0
}

#
# Uses sed to do an inplace replace of string <tag to replace>
# with <value to set> in the file passed in as the first argument
# Function returns exit code of sed command.
#
# setTagInFile(file to modify,tag to replace,value to set)
#
function replaceKeyWordInFile {
  local theFile=$1
  local tagKey=$2
  local vTag=$3
  echo "Setting $tagKey to $vTag in $theFile"
  sed -i "s/$tagKey/$vTag/g" "$theFile"
  return $?
}

CHMSRCDIR_NAME="chm-source-${VERSION}"
CHMSRCDIR="$CHMSRCDIR_NAME"
if [[ -d "$CHMSRCDIR" ]]; then echo "Temporary directory '$CHMSRCDIR' exists, remove it before trying again." 1>&2; exit 1; fi;

VERSION_STRING="(version: $VERSION built `date +'%d %b %Y'`)"

echo ""
echo "Creating release v$VERSION ... (hopefully you did a git pull before this)"
echo ""

if [ "$SKIP_GIT" == "false" ] ; then
  echo "WARNING THIS SCRIPT WILL INVOKE GIT COMMIT AND GIT PUSH!!!  If you don't want this"
  echo "you have 10 seconds to hit Ctrl-c and reinvoke with -s flag"
  sleep 10
fi

echo ""
echo "* Creating 'source' version..."
mkdir "$CHMSRCDIR"
cp -r ../algorithm/* "$CHMSRCDIR"
cp ../LICENSE.txt "$CHMSRCDIR"
mv "$CHMSRCDIR/ReadMe.txt" "$CHMSRCDIR/README-algorithm.txt"
cp ./release-readmes/README-source.txt "$CHMSRCDIR/README.txt"

replaceKeyWord "$CHMSRCDIR" "$VERSION_KEYWORD" "$VERSION_STRING"

# Generate chm source tarball
CHM_SRC_TARBALL="chm-source-$VERSION.tar.gz"
GZIP=-9 tar -czf "$CHM_SRC_TARBALL" "$CHMSRCDIR"

echo "Generated $CHM_SRC_TARBALL"
rm -rf "$CHMSRCDIR"

echo ""
echo "* Creating 'compiled' version..."
matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('cd ./compiled;compile_exec;');";

if [ $? != 0 ] ; then
  echo ""
  echo "Error running matlab command: matlab -nodisplay -singleCompThread -nojvm -r \"run_from_shell('cd ./compiled;compile_exec;');\""
  echo ""
  exit 1
fi

# Extract matlab release string ie R2013a from matlab-version.txt file
MATLAB_RELEASE_STRING=`grep "MATLAB Version" ./compiled/matlab-version.txt | sed "s/^.*(//" | sed "s/)//"`

# Make a lower case version of matlab release string so we can put it in the chm
# compiled tarball filename
MATLAB_LOWERCASE_RELEASE_STRING=`echo "$MATLAB_RELEASE_STRING" | tr '[:upper:]' '[:lower:]'`

CHMCOMPDIR_NAME="chm-compiled-${MATLAB_LOWERCASE_RELEASE_STRING}-${VERSION}"
CHMCOMPDIR="./$CHMCOMPDIR_NAME"
if [[ -d "$CHMCOMPDIR" ]]; then echo "Temporary directory '$CHMCOMPDIR' exists, remove it before trying again." 1>&2; exit 1; fi;

mkdir "$CHMCOMPDIR"
cp ./compiled/CHM_test* "$CHMCOMPDIR"
cp ./compiled/CHM_train* "$CHMCOMPDIR"
cp ../LICENSE.txt "$CHMCOMPDIR"
cp ./release-readmes/README-compiled.txt "$CHMCOMPDIR/README.txt"

replaceKeyWord "$CHMCOMPDIR" "$VERSION_KEYWORD" "$VERSION_STRING"

# Extract the matlab mcr version from matlab-version.txt file.  Could always make
# this easier by making the matlab command output the exact value we need
MATLAB_MCR_STRING=`grep "Compiler Version" ./compiled/matlab-version.txt | sed "s/^.*: *//" | sed -re "s/([0-9]+\.[0-9]+)\.[0-9]*.*/\1/"`

replaceKeyWord "$CHMCOMPDIR" "$MATLAB_RELEASE_KEYWORD" "$MATLAB_RELEASE_STRING"
replaceKeyWord "$CHMCOMPDIR" "$MATLAB_MCR_KEYWORD" "$MATLAB_MCR_STRING"

# Generate compiled chm tarball
CHM_COMP_TARBALL="chm-compiled-$MATLAB_LOWERCASE_RELEASE_STRING-$VERSION.tar.gz"
GZIP=-9 tar -czf "$CHM_COMP_TARBALL" "$CHMCOMPDIR"

echo "Generated $CHM_COMP_TARBALL"
rm -rf "$CHMCOMPDIR"

# Verify system/functional tests pass on CHM in the source tarball
echo "Running functional/system tests on $CHM_SRC_TARBALL"
echo "This may take a good 20 minutes..."

A_TMP_DIR="/tmp/$VERSION.`uuidgen`"
mkdir -p "$A_TMP_DIR" 
if [ $? != 0 ] ; then
  echo "Unable to make $A_TMP_DIR"
  exit 1
fi

curdir=`pwd`

tar --directory=$A_TMP_DIR -zxf "$curdir/$CHM_SRC_TARBALL"

export CHM_ALT_BIN_DIR="$A_TMP_DIR/$CHMSRCDIR_NAME"

bats ../test/chm_system_tests/

if [ $? != 0 ] ; then
  echo ""
  echo "Error running bats tests on $CHM_SRC_TARBALL"
  echo ""
  /bin/rm -rf "$A_TMP_DIR"
  exit 1
fi

# Verify system/functional tests pass on CHM in compiled tarball
echo "Running functional/system tests on $CHM_COMP_TARBALL"
echo "This may take a good 10 minutes..."

tar --directory=$A_TMP_DIR -zxf "$curdir/$CHM_COMP_TARBALL"

export CHM_ALT_BIN_DIR="$A_TMP_DIR/$CHMCOMPDIR_NAME"

bats ./compiled/test/chm_system_tests/

if [ $? != 0 ] ; then
  echo ""
  echo "Error running bats tests on $CHM_COMP_TARBALL"
  echo ""
  /bin/rm -rf "$A_TMP_DIR"
  exit 1
fi
/bin/rm -rf "$A_TMP_DIR"

echo "All tests pass..."

if [ "$SKIP_GIT" == "false" ] ; then
  echo ""
  echo "* Pushing compiled versions to GitHub..."
  echo "(as soon as we get a better build system this step won't be necessary since the compiled versions won't be stored in the repo)"
  git add ./compiled/CHM_test ./compiled/CHM_train ./compiled/matlab-version.txt
  git commit ./compiled/CHM_test ./compiled/CHM_train ./compiled/matlab-version.txt -m "Updating compiled version." # may be empty, but that's okay
  if [ $? -eq 0 ]; then git push; fi;

  echo ""
  echo "All done. Copy $CHM_COMP_TARBALL and $CHM_SRC_TARBALL to appropriate location"
  echo ""
  echo ""
  echo "If this is an official release you should create a tag for it using the following commands:"
  echo "git tag v$VERSION"
  echo "git push origin v$VERSION"

else
  echo ""
  echo "$CHM_COMP_TARBALL and $CHM_SRC_TARBALL files have been generated"
  echo ""
  echo "BUT NO GIT OPERATIONS WERE PERFORMED. TO DO A FORMAL RELEASE RERUN THIS COMMAND OMITTING -s FLAG"
  echo ""
fi
