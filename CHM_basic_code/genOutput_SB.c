
#include "mex.h"
#include "math.h"
#include "mat.h"

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    
    mwSize nGroup;
    mwSize nDiscriminantPerGroup;    
    float *xtrain;        
    float *discriminants;
        
    mwSize tM, tN, dM, dN;
            
    float *outputs_p;
        
    mwIndex i, j, k;        
    
    float sum_d_pts, mul_d_pts, AND_c_muls;
    
    xtrain = (float*)mxGetData(prhs[0]);    
    discriminants = (float*)mxGetData(prhs[1]);
    
    nGroup = (mwSize)mxGetScalar(prhs[2]);
    nDiscriminantPerGroup = (mwSize)mxGetScalar(prhs[3]);    
    
    dM = (mwSize)mxGetM(prhs[1]);
    dN = (mwSize)mxGetN(prhs[1]);
    
    tM = (mwSize)mxGetM(prhs[0]);
    tN = (mwSize)mxGetN(prhs[0]);    
    
    plhs[0] = mxCreateNumericMatrix(1, tN, mxSINGLE_CLASS, mxREAL);  
        
    float *output;
    output = (float*)mxGetData(plhs[0]);
            
    outputs_p = mxCalloc(dN,sizeof(float));
                
    
    for (i=0; i<tN; i++){        

        for (j=0; j<dN; j++){
            sum_d_pts = 0;
            for (k=0; k<tM; k++){
                sum_d_pts = sum_d_pts + discriminants[j*dM+k]*xtrain[i*tM+k];                  
            }
            sum_d_pts = sum_d_pts + discriminants[j*dM+(dM-1)];
            outputs_p[j] = 1/(1+exp(-sum_d_pts));    
        }

        AND_c_muls = 1;
        for (j=0; j<nGroup; j++){
            mul_d_pts = 1;
            for (k=0; k<nDiscriminantPerGroup; k++){
                mul_d_pts = mul_d_pts * outputs_p[j*nDiscriminantPerGroup+k];
            }    
            
            AND_c_muls = AND_c_muls * (1-mul_d_pts);
        }

        output[i] = 1 - AND_c_muls;
    }

}