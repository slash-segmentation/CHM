
#include "mex.h"
#include "math.h"
#include "mat.h"

#define BSIZE 100

void shuffle(mwIndex *array, mwSize n)
{
    if (n > 1) 
    {
        mwIndex i;
        for (i = 0; i < n - 1; i++) 
        {
          mwIndex j = i + rand() / (RAND_MAX / (n - i) + 1);
          mwIndex t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    
    mwSize maxepoch;
    float epsilon;
    float momentumweight;
    mwSize nGroup;
    mwSize nDiscriminantPerGroup;    
    float *initial_discriminants;
    float *xtrain;
    float *ytrain;
        
    float *discriminants;
    float *totalerror;
    
    mwSize tM, tN, dM, dN;
            
    float *outputs_p, *outputsAND_c_p;
    float *outputsAND_p, *outputs_c_p, *term2_p;
    
    mxArray *prevupdates, *updates;
    float *prevupdates_p, *updates_p;
    
    mwIndex x, ii, i, j, k;    
    mwIndex *ptr_to_rnd; 
    
    MATFile *pmat;
    const char *file = "Data.mat";
    int s;

    float sum_d_pts, mul_d_pts, AND_c_muls, output, erro, mp;
    float *d_pnt;

    xtrain = (float*)mxGetData(prhs[0]);
    ytrain = (float*)mxGetData(prhs[1]);
    initial_discriminants = (float*)mxGetData(prhs[2]);
    
    maxepoch = (mwSize)mxGetScalar(prhs[3]);
    nDiscriminantPerGroup = (mwSize)mxGetScalar(prhs[4]);
    nGroup = (mwSize)mxGetScalar(prhs[5]);
    epsilon = (float)mxGetScalar(prhs[6]);
    momentumweight = (float)mxGetScalar(prhs[7]);    
    
    dM = (mwSize)mxGetM(prhs[2]);
    dN = (mwSize)mxGetN(prhs[2]);
    
    tM = (mwSize)mxGetM(prhs[0]);
    tN = (mwSize)mxGetN(prhs[0]);    
    
    plhs[0] = mxCreateNumericMatrix(dM, dN, mxSINGLE_CLASS, mxREAL);  
    plhs[1] = mxCreateNumericMatrix(maxepoch, 1, mxSINGLE_CLASS, mxREAL);            
        
    discriminants = (float*)mxGetData(plhs[0]);
    totalerror = (float*)mxGetData(plhs[1]);   
        
    outputs_p = mxCalloc(dN,sizeof(float));
    term2_p = mxCalloc(dN,sizeof(float));
    outputs_c_p = mxCalloc(dN,sizeof(float));
    outputsAND_p = mxCalloc(nGroup,sizeof(float));
    outputsAND_c_p = mxCalloc(nGroup,sizeof(float));
    
    prevupdates = mxCreateNumericMatrix(dM, dN, mxSINGLE_CLASS, mxREAL);  
    updates = mxCreateNumericMatrix(dM, dN, mxSINGLE_CLASS, mxREAL);  
    
    prevupdates_p = (float*)mxGetData(prevupdates);
    updates_p = (float*)mxGetData(updates);

    for (i=0;i<dM*dN;i++){
        updates_p[i] = 0;
    } 
    
    for (i=0;i<dM*dN;i++){
        discriminants[i] = initial_discriminants[i];
    } 
    
    ptr_to_rnd = mxCalloc(tN,sizeof(mwIndex));
    
    d_pnt = mxCalloc(tM,sizeof(float));
        
    for (i=0; i<tN; ptr_to_rnd[i] = i++);
        
    for (x=0;x<maxepoch;x++){
        shuffle(ptr_to_rnd, tN); 
        for (i=0; i<tN; i++){
            ii = ptr_to_rnd[i];
            
            for (j=0; j<dN; j++){
                sum_d_pts = 0;
                for (k=0; k<dM; k++){
                    sum_d_pts = sum_d_pts + discriminants[j*dM+k]*xtrain[ii*tM+k];                  
                }
                outputs_p[j] = 1/(1+exp(-sum_d_pts));
                outputs_c_p[j] = 1 - outputs_p[j];
            }
            
            AND_c_muls = 1;
            for (j=0; j<nGroup; j++){
                mul_d_pts = 1;
                for (k=0; k<nDiscriminantPerGroup; k++){
                    mul_d_pts = mul_d_pts * outputs_p[j*nDiscriminantPerGroup+k];
                }
                outputsAND_p[j] = mul_d_pts;
                outputsAND_c_p[j] = 1-mul_d_pts;
                AND_c_muls = AND_c_muls * outputsAND_c_p[j];
            }
            
            output = 1 - AND_c_muls;
            erro = 0.1+ 0.8*ytrain[ii] - output;                       
            totalerror[x] = totalerror[x] + erro*erro;                    
            
            for (j=0; j<nGroup; j++){                
                mp = ((AND_c_muls/outputsAND_c_p[j])*outputsAND_p[j])*erro;
                for (k=0; k<nDiscriminantPerGroup; k++){
                    term2_p[j*nDiscriminantPerGroup+k] = mp * outputs_c_p[j*nDiscriminantPerGroup+k];
                }                
            }
            
            for (j=0; j<dN; j++){
                for (k=0; k<tM; k++){
                   /* updates_p[j*tM+k] = xtrain[ii*tM+k] * term2_p[j] + momentumweight*prevupdates_p[j*tM+k];                    */
                    updates_p[j*tM+k] = xtrain[ii*tM+k] * term2_p[j] + updates_p[j*tM+k]; 
                    if ( (i+1)%BSIZE == 0 ){
                        discriminants[j*tM+k] = discriminants[j*tM+k] + epsilon*(updates_p[j*tM+k] + momentumweight*prevupdates_p[j*tM+k]);
                        prevupdates_p[j*tM+k] = updates_p[j*tM+k] + momentumweight*prevupdates_p[j*tM+k];
                        updates_p[j*tM+k] = 0;
                    }
                }
            }
        }
        totalerror[x] = sqrt(totalerror[x]/tN);
        mexPrintf("Epoch No. %d ... error = %f \n",x+1,totalerror[x]);
        mexEvalString("drawnow;"); // causes the print to happen now

 /*
        pmat = matOpen(file, "w");
        s = matPutVariable(pmat, "discriminants", plhs[0]);
        s = matPutVariable(pmat, "totalerror", plhs[1]);
        s = matClose(pmat); */
    }     
}