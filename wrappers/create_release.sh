#!/bin/bash

VERSION=2.1 # update manually
BUILD_NUM=`git log --pretty=format:'' | wc -l`  ## `git rev-list HEAD --count` # number of commits
VERSION=$VERSION.$BUILD_NUM

if [[ -d ./chm ]]; then echo "Temporary directory 'chm' exists, remove it before trying again." 1>&2; exit 1; fi;

echo "Creating release v$VERSION... (hopefully you did a git pull before this)"

echo ""
echo "* Creating 'source' version..."
mkdir ./chm
cp -r ../algorithm/* ./chm
cp ../LICENSE.txt ./chm
mv ./chm/ReadMe.txt ./chm/README-algorithm.txt
cp ./release-readmes/README-source.txt ./chm/README.txt
GZIP=-9 tar -czf chm-source-$VERSION.tar.gz chm
echo "Generated chm-source-$VERSION.tar.gz"
rm -rf chm

echo ""
echo "* Creating 'compiled' version..."
mkdir ./chm
matlab -nodisplay -singleCompThread -nojvm -r "run_from_shell('cd ./compiled;compile_exec;');";
cp ./compiled/CHM_test* ./chm
cp ./compiled/CHM_train* ./chm
cp ../LICENSE.txt ./chm
cp ./release-readmes/README-compiled.txt ./chm/README.txt
GZIP=-9 tar -czf chm-compiled-$VERSION.tar.gz chm
echo "Generated chm-compiled-$VERSION.tar.gz"
rm -rf chm

echo ""
echo "* Pushing compiled versions to GitHub..."
echo "(as soon as we get a better build system this step won't be necessary since the compiled versions won't be stored in the repo)"
git add ./compiled/CHM_test ./compiled/CHM_train ./compiled/matlab-version.txt
git commit ./compiled/CHM_test ./compiled/CHM_train ./compiled/matlab-version.txt -m "Updating compiled version." # may be empty, but that's okay
if [ $? -eq 0 ]; then git push; fi;

echo ""
echo "All done. Copy those tar-balls somewhere."

echo ""
echo "If this is an official release you should create a tag for it using the following commands:"
echo "git tag v$VERSION"
echo "git push origin v$VERSION"

