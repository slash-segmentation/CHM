% You need to run mbuild -setup before running this script for the first time!

% ? rm $prefdir/../mcr_*/CHM_*

cd ../../algorithm

% Compile the programs
% -m creates a standalone C command line application
% -C creates a seperate CTF archive from the executable (instead of packing them together)
% -R specifies matlab runtime arguments
% -N removes all toolbox paths
% -p readds a toolbox path
% -I includes a folder while compiling (does not necessary include all files in it if they are not found needed)
% -o gives the output filename
% -d gives the output directory
% Everything at the end are the functions to include/export
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -I FilterStuff -d ../wrappers/compiled CHM_test
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -p images/iptformats -I FilterStuff -d ../wrappers/compiled CHM_test_blocks
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -I FilterStuff -d ../wrappers/compiled CHM_train

cd ../wrappers/compiled

% Remove extraneous files
delete run_CHM_test.sh
delete run_CHM_test_blocks.sh
delete run_CHM_train.sh
delete readme.txt
delete mccExcludedFiles.log
