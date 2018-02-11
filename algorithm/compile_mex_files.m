% Compile mex files
mex UpdateDiscriminants.c COPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer" LDOPTIMFLAGS="-O3"
mex genOutput.c -lmwblas COPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer" LDOPTIMFLAGS="-O3"
mex genOutput_SB.c COPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer" LDOPTIMFLAGS="-O3"
mex UpdateDiscriminants_SB.c COPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer" LDOPTIMFLAGS="-O3"
cd FilterStuff/

mex cc_cmp_II.cc CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer" LDOPTIMFLAGS="-O3"
mex cc_Haar_features.cc CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer" LDOPTIMFLAGS="-O3"
mex HoG.cpp CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer" LDOPTIMFLAGS="-O3"

cd ..
