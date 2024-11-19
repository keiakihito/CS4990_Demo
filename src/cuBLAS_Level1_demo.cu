#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/checks.h"
#include "../include/helper.h"

int main(){
    const int N = 5;
    const float vecX_h[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const float vecY_h[N] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float result_h = 0.0f;

    float* vecX_d = NULL;
    float* vecY_d = NULL;
    float* result_d = NULL;

    bool debug = true;

    // Allocate memory on device
    CHECK(cudaMalloc((void**)&vecX_d, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&vecY_d, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&result_d, sizeof(float)));

    // Copy data from host to device
    CHECK(cudaMemcpy(vecX_d, vecX_h, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vecY_d, vecY_h, N * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\nVector X (Device Memory):\n");
        print_vector_d(vecX_d, N);

        printf("\nVector Y (Device Memory):\n");
        print_vector_d(vecY_d, N);

    }

    // Setup cuBLAS handler
    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));

    // Calling cuBLAS API
    CHECK_CUBLAS(cublasSdot(cublasHandler, N, vecX_d, 1, vecY_d, 1, result_d));

    // Copy data from device to host
    CHECK(cudaMemcpy(&result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost));

    //Check the result
    printf("\nresult_h: %f",result_h);
    // Free memeory
    CHECK_CUBLAS(cublasDestroy(cublasHandler));
    CHECK(cudaFree(vecX_d));
    CHECK(cudaFree(vecY_d));
    CHECK(cudaFree(result_d));

    return 0;
}

/*
Sample Run

Vector X (Device Memory):
1.0000000000
2.0000000000
3.0000000000
4.0000000000
5.0000000000

Vector Y (Device Memory):
5.0000000000
4.0000000000
3.0000000000
2.0000000000
1.0000000000

result_h: 35.000000

*/