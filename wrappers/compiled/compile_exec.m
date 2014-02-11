% You need to run mbuild -setup before running this script for the first time!

cd ../../algorithm

% Compile the programs
% -m creates a standalone C command line application
% -R specifies matlab runtime arguments
% -N removes all toolbox paths
% -p readds a toolbox path
% -I includes a folder while compiling (does not necessary include all files in it if they are not found needed)
% -o gives the output filename
% -d gives the output directory
% Everything at the end are the functions to include/export
fprintf('Compiling CHM_test...\n');
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -I FilterStuff -d ../wrappers/compiled CHM_test
fprintf('\nCompiling CHM_test_blocks...\n');
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -p images/iptformats -p images/imuitools -I FilterStuff -d ../wrappers/compiled CHM_test_blocks
fprintf('\nCompiling CHM_train...\n');
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -I FilterStuff -d ../wrappers/compiled CHM_train

cd ../wrappers/compiled

% Remove extraneous files
delete run_CHM_test.sh
delete run_CHM_test_blocks.sh
delete run_CHM_train.sh
delete readme.txt
delete mccExcludedFiles.log

% Save MATLAB and compiler version which last compiled the code
fid = fopen('matlab-version.txt','w');
fprintf(fid,'Platform:         %s\n',computer);
fprintf(fid,'MATLAB Version:   %s\n',version);
[maj,min,rev] = mcrversion;
fprintf(fid,'Compiler Version: %d.%d.%d\n',maj,min,rev); 
fclose(fid);
