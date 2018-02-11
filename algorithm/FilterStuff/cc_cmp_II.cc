// cc_cmp_II
// This function computes the integral image over the input image
// Michael Villamizar
// mvillami@iri.upc.edu
// 2009
//
// input:
// 	<- Input Image
// output:
//	-> Integral Image (II)
//

#include <math.h>
#include "mex.h"
#include <stdio.h>

// main function
mxArray *process(const mxArray *mxImg) {

	// input image	
	double *img = (double *)mxGetPr(mxImg);
	const size_t *imgSize = mxGetDimensions(mxImg);

	// integral image
	size_t out[2];
	out[0] = (int)imgSize[0];
	out[1] = (int)imgSize[1];
	mxArray *mxII = mxCreateNumericArray(2, out, mxDOUBLE_CLASS, mxREAL);
	double *II = (double *)mxGetPr(mxII);

	// temp variables
	double xo,yo,xoyo;
	
	// scanning
	for (int x = 0; x < imgSize[1]; x++) {
		for (int y = 0; y < imgSize[0]; y++) {

			if (x>0){
				if (y>0){
					xo   = *(II + (x-1)*imgSize[0] + y    );
					yo   = *(II + (x)*imgSize[0]   + (y-1));
					xoyo = *(II + (x-1)*imgSize[0] + (y-1));
				} 
				else {
					xo = *(II + (x-1)*imgSize[0] + y);
					yo = double(0);
					xoyo = double(0);
				}
			}
			else {
				if(y>0) {
					xo = double(0);
					yo = *(II + (x)*imgSize[0] + (y-1));
					xoyo = double(0);
				} 
				else {
					xo = double(0);
					yo = double(0);
					xoyo = double(0);
				}
			}

			// current integral image value
			*(II + x*imgSize[0] + y) = *(img + x*imgSize[0] + y) - xoyo + xo + yo;

		}//y
	}//x

	return mxII;
}

// compute integral image
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nlhs != 1)
    mexErrMsgTxt("incorrect input parameters : 1-> input image");
  plhs[0] = process(prhs[0]);
}



