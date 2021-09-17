#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initVector(float **vec_h, unsigned size)
{
    *vec_h = (float*)malloc(size*sizeof(float));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%100)/100.00;
    }

}

void verify(float *in, float naiveRout, float improvedRout, unsigned in_size) {

    const float relativeTolerance = 2e-5;

    // Reduction.
    float sum = 0.0f;
    for(int i = 0; i < in_size; ++i) {
        sum += in[i];
    }

    // Naive Reduction Error
    float relativeError = (sum - naiveRout)/sum;
    printf("%f/%f ", sum, naiveRout);
    if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
        printf("Naive Reduction TEST FAILED, cpu = %0.3f, gpu = %0.3f\n\n", sum, naiveRout);
        exit(0);
    }
    printf("NAIVE REDUCTION TEST PASSED\n\n");

    // Improved Reduction Error
    relativeError = (sum - improvedRout)/sum;
    printf("%f/%f ", sum, improvedRout);
    if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
        printf("Improved Reduction TEST FAILED, cpu = %0.3f, gpu = %0.3f\n\n", sum, improvedRout);
        exit(0);
    }
    printf("IMPROVED REDUCTION TEST PASSED\n\n");
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

