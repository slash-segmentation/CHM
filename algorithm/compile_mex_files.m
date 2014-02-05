% Compile mex files
mex -largeArrayDims UpdateDiscriminants.c
mex -v -largeArrayDims genOutput.c -lmwblas
mex -largeArrayDims genOutput_SB.c
mex -largeArrayDims UpdateDiscriminants_SB.c
cd FilterStuff/

mex cc_cmp_II.cc
mex cc_Haar_features.cc
mex -largeArrayDims HoG.cpp

cd ..
