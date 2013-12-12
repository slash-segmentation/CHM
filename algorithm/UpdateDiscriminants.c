
#include "mex.h"
#include "math.h"
#include "mat.h"
#define EPSI 1e-14

void shuffle(mwIndex *array, mwSize n)
{
    if (n > 1) 
    {
        mwIndex i;
        for (i = 0; i < n - 1; i++) 
        {
          mwIndex j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    
    mwSize maxepoch;
    double epsilon;
    double momentumweight;
    mwSize nGroup;
    mwSize nDiscriminantPerGroup;
    double *initial_discriminants;
    double *xtrain;
    double *ytrain;
    
    double *discriminants;
    double *totalerror;
    double *totalerrorvalid;
    double lambda;
    mwSize tM, tN, dM, dN;
            
    xtrain = mxGetPr(prhs[0]);
    ytrain = mxGetPr(prhs[1]);
    initial_discriminants = mxGetPr(prhs[2]);
    
    maxepoch = mxGetScalar(prhs[3]);
    nDiscriminantPerGroup = mxGetScalar(prhs[4]);
    nGroup = mxGetScalar(prhs[5]);
    epsilon = mxGetScalar(prhs[6]);
    momentumweight = mxGetScalar(prhs[7]);    
    
    dM = mxGetM(prhs[2]);
    dN = mxGetN(prhs[2]);
    
    tM = mxGetM(prhs[0]);
    tN = mxGetN(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(dM, dN, mxREAL);            
    plhs[1] = mxCreateDoubleMatrix(maxepoch, 1, mxREAL); 
    
    discriminants = mxGetPr(plhs[0]);
    totalerror = mxGetPr(plhs[1]);
    /*totalerrorvalid = mxGetPr(plhs[2]);*/
    
    /* cross validation variables */
    /*double *xvalid, *yvalid;
    xvalid = mxGetPr(prhs[8]);
    yvalid = mxGetPr(prhs[9]);

    mwSize vN;
    vN = mxGetN(prhs[8]);
    double *voutputs_p;
    voutputs_p = mxCalloc(dN,sizeof(double));
    double  vmul_d_pts, vAND_c_muls, voutput, verro, CV;
    CV = 0;
    mxArray *discreserve, *discold;
    discreserve = mxCreateDoubleMatrix(dM, dN, mxREAL);
    discold = mxCreateDoubleMatrix(dM, dN, mxREAL);
    
    double *discreservep, *discoldp;
    discreservep = mxGetPr(discreserve);
    discoldp = mxGetPr(discold);*/
    
    /* end of validation variables */
    
    double *temp1_p, *outputs_p, *outputsAND_c_p;
    double *outputsAND_p, *term1_p, *outputs_c_p, *term2_p;

    temp1_p = mxCalloc(dM*dN,sizeof(double));
    outputs_p = mxCalloc(dN,sizeof(double));
    term2_p = mxCalloc(dN,sizeof(double));
    term1_p = mxCalloc(nGroup,sizeof(double));
    outputs_c_p = mxCalloc(dN,sizeof(double));
    outputsAND_p = mxCalloc(nGroup,sizeof(double));
    outputsAND_c_p = mxCalloc(nGroup,sizeof(double));
    
    mxArray *prevupdates, *updates;
    double *prevupdates_p, *updates_p;
            
    prevupdates = mxCreateDoubleMatrix(dM, dN, mxREAL);
    updates = mxCreateDoubleMatrix(dM, dN, mxREAL);    
    
    prevupdates_p = mxGetPr(prevupdates);
    updates_p = mxGetPr(updates);
    
    
    mwIndex x, ii, i, j, k, jj, kk;
    mwSize  DOind,DOindD;
    mwIndex *ptr_to_rnd; 
    mwIndex *DO_ptr,*DOD_ptr; 
    DO_ptr = mxCalloc(nGroup,sizeof(mwIndex));
    DOD_ptr = mxCalloc(nDiscriminantPerGroup,sizeof(mwIndex));
    DOind = nGroup/2;
    DOindD = nDiscriminantPerGroup/2;
    for (i=0;i<nGroup;DO_ptr[i] = i++);
    for (i=0;i<nDiscriminantPerGroup;DOD_ptr[i] = i++);
    
    MATFile *pmat;
    const char *file = "Outputs_v2.mat";
    int s;
    
    
    for (i=0;i<dM*dN;i++){
        discriminants[i] = initial_discriminants[i];
        
    }
    
    ptr_to_rnd = mxCalloc(tN,sizeof(mwIndex));

    
    double sum_d_pts, mul_d_pts, AND_c_muls, output, erro, mp;
    double *d_pnt;
    d_pnt = mxCalloc(tM,sizeof(double));

    
    for (i=0; i<tN; ptr_to_rnd[i] = i++);
        
    for (x=0;x<maxepoch;x++){
        
 /* cross validation */
        /*for (i=0; i<vN; i++){
            
            for (j=0; j<dN; j++){

                sum_d_pts = 0;
                for (k=0; k<dM; k++){
                    sum_d_pts = sum_d_pts + discriminants[j*dM+k]*xvalid[i*tM+k];                  
                }
                voutputs_p[j] = 1/(1+exp(-sum_d_pts));
            }            
            
            
            vAND_c_muls = 1;
            for (j=0; j<nGroup; j++){
                vmul_d_pts = 1;
                for (k=0; k<nDiscriminantPerGroup; k++){
                    vmul_d_pts = vmul_d_pts * voutputs_p[j*nDiscriminantPerGroup+k];
                }
                vAND_c_muls = vAND_c_muls * (1 - vmul_d_pts);
            }
             
            
            voutput = 1 - vAND_c_muls;
            verro = 0.1+ 0.8*yvalid[i] - voutput;
            totalerrorvalid[x] = totalerrorvalid[x] + verro*verro;
            }
            totalerrorvalid[x] = sqrt(totalerrorvalid[x]/vN);*/
/* end of computing validation outputs */       
        

        shuffle(ptr_to_rnd, tN); 
        for (i=0; i<tN; i++){
            ii = ptr_to_rnd[i];            
            shuffle(DO_ptr, nGroup);
            shuffle(DOD_ptr, nDiscriminantPerGroup);
            for (j=0; j<DOind; j++){
                for(kk=0; kk<DOindD; kk++){
                    jj = DO_ptr[j]*nDiscriminantPerGroup+DOD_ptr[kk];
                    sum_d_pts = 0;
                    for (k=0; k<dM; k++){
                        sum_d_pts = sum_d_pts + discriminants[jj*dM+k]*xtrain[ii*tM+k];                  
                    }
                    outputs_p[jj] = 1/(1+exp(-sum_d_pts));
                    outputs_c_p[jj] = 1 - outputs_p[jj];
                }
            }
            
            AND_c_muls = 1;
            for (j=0; j<DOind; j++){
                jj = DO_ptr[j];
                mul_d_pts = 1;
                for (k=0; k<DOindD; k++){

                   mul_d_pts = mul_d_pts * outputs_p[jj*nDiscriminantPerGroup+DOD_ptr[k]];

                }
                outputsAND_p[jj] = mul_d_pts;
                outputsAND_c_p[jj] = 1-mul_d_pts;
                AND_c_muls = AND_c_muls * outputsAND_c_p[jj];
            }
            
            output = 1 - AND_c_muls;
            erro = 0.1+ 0.8*ytrain[ii] - output;                       
            totalerror[x] = totalerror[x] + erro*erro;
            

          
            for (j=0; j<DOind; j++){
                jj = DO_ptr[j];
                mp = ((AND_c_muls/outputsAND_c_p[jj])*outputsAND_p[jj])*erro;              
                for (k=0; k<DOindD; k++){

                    term2_p[jj*nDiscriminantPerGroup+DOD_ptr[k]] = mp * outputs_c_p[jj*nDiscriminantPerGroup+DOD_ptr[k]];

                }                
            }
            
            for (j=0; j<DOind; j++){
                for(kk=0;kk<DOindD; kk++){
                    jj = DO_ptr[j]*nDiscriminantPerGroup+DOD_ptr[kk];
                    for (k=0; k<tM; k++){
                        updates_p[jj*tM+k] = xtrain[ii*tM+k] * term2_p[jj] + momentumweight * prevupdates_p[jj*tM+k];
                        discriminants[jj*tM+k] = discriminants[jj*tM+k] + epsilon * updates_p[jj*tM+k];
                        prevupdates_p[jj*tM+k] = updates_p[jj*tM+k];
                    }
                }
            }
            
            /*mxDestroyArray(prevupdates);
            prevupdates = mxDuplicateArray(updates);                
            prevupdates_p = mxGetPr(prevupdates);*/
            
        }
        totalerror[x] = sqrt(totalerror[x]/tN);
        mexPrintf("Epoch No. %d ... error = %f \n",x+1,totalerror[x]); 
/*        mexPrintf("Epoch No. %d ... error = %f (validation set)\n",x+1,totalerrorvalid[x]);      
        if(x>0){
            if(totalerrorvalid[x] > totalerrorvalid[x-1]){
                CV = CV + 1;
                if (CV==1){
                    mxDestroyArray(discreserve);
                    discreserve = mxDuplicateArray(discold);
                    discreservep = mxGetPr(discreserve);
                }
                if(CV==3){
                    mxDestroyArray(plhs[0]);
                    plhs[0] = mxDuplicateArray(discreserve);
                    discriminants = mxGetPr(plhs[0]);
                    break;
                }
            }
            else{
                CV = 0;
            }
        } */
        /*mxDestroyArray(discold);
        discold = mxDuplicateArray(plhs[0]);
        discoldp = mxGetPr(discold);*/
        pmat = matOpen(file,"w");
        s = matPutVariable(pmat,"discriminants",plhs[0]);
        s = matPutVariable(pmat,"totalerror",plhs[1]);
        /*s = matPutVariable(pmat,"totalerrorvalid",plhs[2]);*/
        s = matClose(pmat);
    }
    /*mxDestroyArray(discreserve);
    mxDestroyArray(discold);*/
    mxDestroyArray(prevupdates);
    mxDestroyArray(updates);
}
