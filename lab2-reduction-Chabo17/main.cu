/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    float *in_h, *naiveRout_h, *improvedRout_h;
    unsigned in_size;
    cudaError_t cuda_ret;

    // Allocate and initialize host memory
    if(argc == 1) {
        in_size = 1000000;
    } else if(argc == 2) {
        in_size = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./reduction          # Input of size 1,000,000 is used"
           "\n    Usage: ./reduction <m>      # Input of size m is used"
           "\n");
        exit(0);
    }
    initVector(&in_h, in_size);

    naiveRout_h = (float*)malloc(1 * sizeof(float));
    if(naiveRout_h == NULL) FATAL("Unable to allocate host");
 
    improvedRout_h = (float*)malloc(1 * sizeof(float));
    if(improvedRout_h == NULL) FATAL("Unable to allocate host");

    // Launch naive Reduction ---------------------------------------------------
    printf("Launching kernel naive Reduction..."); fflush(stdout);
    startTime(&timer);

    naiveReducion(naiveRout_h, in_h, in_size);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch improved Reduction ------------------------------------------------
    printf("Launching kernel improved Reduction..."); fflush(stdout);
    startTime(&timer);

    improvedReducion(improvedRout_h, in_h, in_size);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, naiveRout_h[0], improvedRout_h[0], in_size);

    // Free memory ------------------------------------------------------------

    free(in_h); 
    free(naiveRout_h);
    free(improvedRout_h);

    return 0;
}

