// cc_Haar_features
// This function computes the Haar-like features in X and Y
// Michael Villamizar
// mvillami@iri.upc.edu
// 2009
//
// input:
// 	<- Integral Image (II)
//	<- Haar size
// output:
//	-> Haar maps : 1) Hx, 2) Hy
// tips:
// 	* Haar size must be even
//

#include <math.h>
#include "mex.h"
#include <stdio.h>
#define PI 3.1415
#define eps 0.00001

// main function 
mxArray *process(const mxArray *mxII, const mxArray *mxSize) {

  // input integral image (II)
  double *II = (double *)mxGetPr(mxII);
  const int *IIsize = mxGetDimensions(mxII);

  // Haar feature size
  int size = (int)mxGetScalar(mxSize);

  // mid point
  int mid = (int)round((double)size/2);
 
  // output map
  int out[3];
  out[0] = (int)(IIsize[0]-size);
  out[1] = (int)(IIsize[1]-size);
  out[2] = 2;
  mxArray *mxMap = mxCreateNumericArray(3, out, mxDOUBLE_CLASS, mxREAL);
  double *map = (double *)mxGetPr(mxMap);

  // variables
  double area, aLeft, aRight, aUp, aDown, Hx, Hy;
  int left, right, up, down; 	

  for (int x = 0; x < out[1]; x++) {
	for (int y = 0; y < out[0]; y++) {

		// region coordinates
		left  = x;
		right = left + size;	
		up    = y;
		down  = up + size;

		// total area
		area  =  *(II + right*IIsize[0] + down) + *(II + left*IIsize[0] + up) - *(II + right*IIsize[0] + up) - *(II + left*IIsize[0] + down);

		if (area>0) {

			// Haar X
			aRight = *(II + right*IIsize[0] + down) + *(II + (left+mid)*IIsize[0] + up) - *(II+ right*IIsize[0] + up) - *(II + (left+mid)*IIsize[0] + down);
			aLeft = *(II + (right-mid)*IIsize[0] + down) + *(II + left*IIsize[0] + up) - *(II + (right-mid)*IIsize[0] + up) - *(II + left*IIsize[0] + down);
			
			// Haar Y
			aUp = *(II + right*IIsize[0] + (down-mid)) + *(II + left*IIsize[0] + up) - *(II + right*IIsize[0] + up) - *(II + left*IIsize[0] + (down-mid));
			aDown = *(II + right*IIsize[0] + down) + *(II + left*IIsize[0] + (up+mid)) - *(II + right*IIsize[0] + (up+mid)) - *(II + left*IIsize[0] + down);
			
			// normalization
			Hx = (aRight-aLeft)/(area+eps);
			Hy = (aDown-aUp)/(area+eps);
		
			// Haar feature values
			*(map + x*out[0] + y + 0*out[0]*out[1]) = Hx;
			*(map + x*out[0] + y + 1*out[0]*out[1]) = Hy;
	
		}//area
	}// y
  }// x
  
  return mxMap;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
    if (nrhs != 2)
       mexErrMsgTxt("incorrect inputs : 1->II 2->HaarSize"); 
    plhs[0] = process(prhs[0],prhs[1]);
}



