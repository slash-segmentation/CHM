/* mex -v -largeArrayDims genOutput.c -lmwblas */
#if !defined(_WIN32)
#define dgemm dgemm_
#endif

#include "mex.h"
#include "math.h"
#include "blas.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    double *x, *disc, *output, *outAnd, sum_d_pts, mul_d_pts, AND_c_muls, *outputs_p;
    
    long int tN, nG, nD, dN, dM, tM;
    
    x = mxGetPr(prhs[0]); /* Input features */
    tN = mxGetN(prhs[0]); /* Number of data points */
    tM = mxGetM(prhs[0]);
    disc = mxGetPr(prhs[1]); /* discriminants */
    nG = mxGetScalar(prhs[2]); /* Number of ORs */
    nD = mxGetScalar(prhs[3]); /* Number of ANDs */
    dM = mxGetM(prhs[1]); 
    dN = mxGetN(prhs[1]);
    

    plhs[0] = mxCreateDoubleMatrix(1,tN,mxREAL);
    output = mxGetPr(plhs[0]); /* classifier prediction */
    
    /* outputs_p = mxCalloc(dN,sizeof(double)); */
    
    long int i,j,k;
    
    
    /* dgemm arguments */
    char *chn = "N", *chnn = "T";
    double one = 1.0, zero = 0.0;
    long int mm = 1, mmm = dM-1;
    mxArray *CC;
    CC = mxCreateDoubleMatrix(dN,tN,mxREAL);
    double *C;
    C = mxGetPr(CC);
    dgemm(chnn, chn, &dN, &tN, &tM, &one, disc, &dM, x, &tM, &zero, C, &dN);
    
/*
    for(i=0;i<dN;i++){
        for(j=0;j<tN;j++){
            C[i*tN+j] = 1/(1+exp(-C[i*tN+j]));
        }
    }
*/     
    mxArray *outAND;
    outAND = mxCreateDoubleMatrix(nG,tN,mxREAL);
    double *outANDp;
    outANDp = mxGetPr(outAND);
    
    for(j=0;j<tN;j++){
        mul_d_pts = 1;
        for(i=0;i<nG;i++){
            outANDp[j*nG+i] = 1;
            for(k=0;k<nD;k++){
                outANDp[j*nG+i] = outANDp[j*nG+i] /(1+exp(-C[j*dN + i*nD + k]));
            }
            mul_d_pts = mul_d_pts * (1-sqrt(outANDp[j*nG+i]));
        }
        output[j] = 1 - sqrt(mul_d_pts);
    }

    
}    
    