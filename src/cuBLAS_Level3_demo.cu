#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/checks.h"  
#include "../include/helper.h" 

int main(){
    // Marix dimensions
    const int M = 3; // Rows of Matrix A and Matrix C
    const int N = 3; // Columns of Matrix B and Matrix C
    const int K = 3; // Columns of Matrix A and Rows of Matrx B

    const float alpha = 1.0f;
    const float beta = 0.0f;

    bool debug = true;

    // Row-major mattrices A, B and C
    float mtxA_h[M * K] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    float mtxB_h[K * N] = {
        9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f
    };

    float mtxC_h[M * N] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

    float* mtxA_d = NULL;
    float* mtxB_d = NULL;
    float* mtxC_d = NULL;

    // Allocate memeory on the device
    CHECK(cudaMalloc((void**)&mtxA_d, M * K * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxB_d, K * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxC_d, M * N * sizeof(float)));

    // Copy data from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxC_d, mtxC_h, M * N * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        // Debug: Print device memory before operation
        printf("\nMatrix A (Device Memory, Row-Major):\n");
        print_mtx_row_d(mtxA_d, M, K);

        printf("\nMatrix B (Device Memory, Row-Major):\n");
        print_mtx_row_d(mtxB_d, K, N);

        printf("\nMatrix C (Device Memory, Row-Major):\n");
        print_mtx_row_d(mtxC_d, M, N);
    }

    // Set up cuBLAS
    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));

    // Call cuBLAS API
    // Since matrices are Row-Major, the leading dimensnion is as follow
    // Matrix A: K, Matrix B: N, Matrix C: K
    // cuBLAS assumes column-major order, so we need to transpose when it handles Row-major matrics
    CHECK_CUBLAS(cublasSgemm(cublasHandler, CUBLAS_OP_T,CUBLAS_OP_T, M, N, K, &alpha, mtxA_d, K, mtxB_d, N, &beta, mtxC_d, K));

    if (debug){
        // Debug: Print device memory after operation
        printf("\nMatrix C (Device Memory, After SGEMM):\n");
        print_mtx_row_d(mtxC_d, M, N);
    }

    // Copy result from device to host
    CHECK(cudaMemcpy(mtxC_h, mtxC_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result Matrix C in host memory
    printf("\nMatrix C (Host Memory): \n");
    print_mtx_row_h(mtxC_h, M, N);

    // Free device memory
    CHECK_CUBLAS(cublasDestroy(cublasHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxB_d));
    CHECK(cudaFree(mtxC_d));





    return 0;
}
