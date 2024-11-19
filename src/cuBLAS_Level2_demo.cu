#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/checks.h"  
#include "../include/helper.h" 

int main(){
    // Row-major Matrix A(3x3) and vectors x, y
    const int numOfRow = 3;
    const int numOfCol = 3;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float mtxA_h[numOfRow * numOfCol] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    float vecX_h[numOfRow] = {1.0f, 2.0f, 3.0f};
    float vecY_h[numOfRow] = {0.0f, 0.0f, 0.0f}; // It will be a result.

    float* mtxA_d = NULL;
    float* vecX_d = NULL;
    float* vecY_d = NULL;

    bool debug = true;

    // Allocate memory on the device
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRow * numOfCol * sizeof(float)));
    CHECK(cudaMalloc((void**)&vecX_d, numOfRow * sizeof(float)));
    CHECK(cudaMalloc((void**)&vecY_d, numOfRow * sizeof(float)));

    // Copy data from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, numOfRow * numOfCol * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vecX_d, vecX_h, numOfRow * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vecY_d, vecY_h, numOfRow * sizeof(float), cudaMemcpyHostToDevice));


    if (debug) {
        // Debug: Print device memory before operation
        printf("\nMatrix A (Device Memory, Row-Major):\n");
        print_mtx_row_d(mtxA_d, numOfRow, numOfCol);

        printf("\nVector X (Device Memory):\n");
        print_mtx_clm_d(vecX_d, numOfCol, 1);

        printf("\nVector Y (Device Memory, Before SGEMV):\n");
        print_mtx_clm_d(vecY_d, numOfRow, 1);
    }

    // Set up cuBLAS
    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));

    // Call cuBLAS API
    // The matrix A is row-major and the leasing dimenstion is the number of columns.
    // cuBLAS assumes column-major order, so we need to transpose when it handles Row-major matrics.
    CHECK_CUBLAS(cublasSgemv(cublasHandler, CUBLAS_OP_T, numOfRow, numOfCol, &alpha, mtxA_d, numOfRow, vecX_d, 1, &beta, vecY_d, 1));

    if (debug) {
        // Debug: Print result vector Y in device memory
        printf("\nVector Y (Device Memory, After SGEMV):\n");
        print_mtx_clm_d(vecY_d, numOfRow, 1);
    }

    // Copy result from device to host
    CHECK(cudaMemcpy(vecY_h, vecY_d, numOfRow * sizeof(float), cudaMemcpyDeviceToHost));

    // Check the result
    printf("\nvecY_h:\n");
    print_mtx_clm_h(vecY_h, numOfRow, 1);

    // Free memory
    CHECK_CUBLAS(cublasDestroy(cublasHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(vecX_d));
    CHECK(cudaFree(vecY_d));

    return 0;
}

/*
Sample Run

Matrix A (Device Memory, Row-Major):
1.000000 2.000000 3.000000 
4.000000 5.000000 6.000000 
7.000000 8.000000 9.000000 

Vector X (Device Memory):
1.000000 
2.000000 
3.000000 

Vector Y (Device Memory, Before SGEMV):
0.000000 
0.000000 
0.000000 

Vector Y (Device Memory, After SGEMV):
30.000000 
36.000000 
42.000000 

vecY_h:
30.000000 
36.000000 
42.000000 

*/
