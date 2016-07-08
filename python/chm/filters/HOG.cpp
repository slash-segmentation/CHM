// HOG - Histogram of oriented gradients features calculator
// Written by Jeffrey Bush, 2015-2016, NCMIR, UCSD
// Adapted from the code in HOG_orig.cpp with the following changes:
//  * Assumes that the image is always grayscale
//  * Params are hard-coded at compile time
//  * Uses doubles everywhere instead of float intermediates
//  * Does not require C++
//  * Much less memory is allocated
//  * Less looping in the second half
//  * Arrays are now used in C order instead of Fortran order (although a compile-time setting can switch this)
//  * Can divide initialization and running
// So overall, faster, more accurate, and less memory intensive.

#define _USE_MATH_DEFINES
#include "HOG.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <stdio.h>

//#define FORTRAN_ORDERING

#ifndef NB_BINS
#define NB_BINS    9
#define CWIDTH_INV 0.125 // 1/cwidth (so cwidth == 8)
#define BLOCK_SIZE 2
#define ORIENT     M_PI  // originally a bool, if true we need 2*PI, if false we need just PI
#define CLIP_VAL   0.2
#endif

#ifdef H_INDEX
#undef H_INDEX
#endif

// the x moves the fastest, followed by b, then finally y
#ifdef FORTRAN_ORDERING
#define H_INDEX(y,x,b) ((y) + ((x) + (b)*hist2)*hist1)
#else
#define H_INDEX(y,x,b) (((y)*hist2 + (x))*NB_BINS + (b))
#endif

ssize_t HOG(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, const ssize_t n)
{
	ssize_t N;
	ssize_t HN = HOG_init(w, h, &N);
    if (n < N) { return -1; }
	dbl_ptr_ar H = (dbl_ptr_ar)malloc(HN*sizeof(double));
	if (H == NULL) { return -2; }
	HOG_run(pixels, w, h, out, H);
	free(H);
	return N;
}

ssize_t HOG_init(const ssize_t w, const ssize_t h, ssize_t *n)
{
    const ssize_t hist1 = 2+(ssize_t)ceil(h*CWIDTH_INV - 0.5);
    const ssize_t hist2 = 2+(ssize_t)ceil(w*CWIDTH_INV - 0.5);
	*n = (hist1-BLOCK_SIZE-1)*(hist2-BLOCK_SIZE-1)*NB_BINS*BLOCK_SIZE*BLOCK_SIZE;
	return hist1*hist2*NB_BINS;
}

void HOG_run(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, dbl_ptr_ar H)
{
    const ssize_t hist1 = 2+(ssize_t)ceil(h*CWIDTH_INV - 0.5);
    const ssize_t hist2 = 2+(ssize_t)ceil(w*CWIDTH_INV - 0.5);
	//const ssize_t N = (hist1-BLOCK_SIZE-1)*(hist2-BLOCK_SIZE-1)*NB_BINS*BLOCK_SIZE*BLOCK_SIZE;
	const ssize_t HN = hist1*hist2*NB_BINS;
	
	memset(H, 0, HN*sizeof(double));

    //Calculate gradients (zero padding)
	// OPT: no padding? 
    for (ssize_t y = 0; y < h; ++y)
	{
		const double cy = y*CWIDTH_INV + 0.5;
        const ssize_t y1 = (ssize_t)cy, y2 = y1 + 1, yw = y*w;
        const double Yc = cy - y1 + 0.5*CWIDTH_INV;
		
        for (ssize_t x = 0; x < w; ++x)
		{
			const double cx = x*CWIDTH_INV + 0.5;
            const ssize_t x1 = (ssize_t)cx, x2 = x1 + 1;
            const double Xc = cx - x1 + 0.5*CWIDTH_INV;
			
			const double dx = ((x!=w-1) ? pixels[yw + (x+1)] : 0) - ((x!=0) ? pixels[yw + (x-1)] : 0);
			const double dy = ((y!=0) ? pixels[(y-1)*w + x] : 0) - ((y!=h-1) ? pixels[(y+1)*w + x] : 0);
            const double grad_mag = sqrt(dx*dx + dy*dy);
            const double grad_or = (atan2(dy,dx) + ((dy<0)*ORIENT)) * (NB_BINS/ORIENT) + 0.5;
            ssize_t b2 = (ssize_t)grad_or, b1 = b2 - 1;
            const double Oc = grad_or - b1 - 1;
            if (b2 == NB_BINS) { b2 = 0; } else if (b1 < 0) { b1 = NB_BINS-1; }
			
            H[H_INDEX(y1,x1,b1)] += grad_mag*(1-Xc)*(1-Yc)*(1-Oc);
            H[H_INDEX(y1,x1,b2)] += grad_mag*(1-Xc)*(1-Yc)*(  Oc);
            H[H_INDEX(y2,x1,b1)] += grad_mag*(1-Xc)*(  Yc)*(1-Oc);
            H[H_INDEX(y2,x1,b2)] += grad_mag*(1-Xc)*(  Yc)*(  Oc);
            H[H_INDEX(y1,x2,b1)] += grad_mag*(  Xc)*(1-Yc)*(1-Oc);
            H[H_INDEX(y1,x2,b2)] += grad_mag*(  Xc)*(1-Yc)*(  Oc);
            H[H_INDEX(y2,x2,b1)] += grad_mag*(  Xc)*(  Yc)*(1-Oc);
            H[H_INDEX(y2,x2,b2)] += grad_mag*(  Xc)*(  Yc)*(  Oc);
        }
    }
    
    //Block normalization
    ssize_t out_i = 0;
    for (ssize_t x = 1; x < hist2-BLOCK_SIZE; ++x)
	{
        for (ssize_t y = 1; y < hist1-BLOCK_SIZE; ++y)
		{
            
            double block_norm = 0.0;
            for (ssize_t i = 0; i < BLOCK_SIZE; ++i)
			{
                for (ssize_t j = 0; j < BLOCK_SIZE; ++j)
				{
                    for (ssize_t k = 0; k < NB_BINS; ++k)
					{
						const double val = H[H_INDEX(y+i,x+j,k)];
                        block_norm += val * val;
                    }
                }
            }
			
			const ssize_t out_start = out_i;
            double block_norm_2 = 0.0;
			if (block_norm > 0.0)
			{
				block_norm = 1.0 / sqrt(block_norm);
				for (ssize_t i = 0; i < BLOCK_SIZE; ++i)
				{
					for (ssize_t j = 0; j < BLOCK_SIZE; ++j)
					{
						for (ssize_t k = 0; k < NB_BINS; ++k)
						{
							double val = H[H_INDEX(y+i,x+j,k)] * block_norm;
							if (val > CLIP_VAL) { val = CLIP_VAL; }
							out[out_i++] = val;
							block_norm_2 += val * val;
						}
					}
				}
			}
			else { out_i += BLOCK_SIZE * BLOCK_SIZE * NB_BINS; }
			
			if (block_norm_2 > 0.0)
			{
				block_norm_2 = 1.0 / sqrt(block_norm_2);
				for (ssize_t i = out_start; i < out_i; ++i) { out[i] *= block_norm_2; }
			}
			else
			{
				memset(out+out_start, 0, (out_i-out_start) * sizeof(double));
			}
        }
    }
}
