/*
Reference page

Documentation
https://docs.nvidia.com/cuda/cusolver/index.html

Example code
https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/getrf/cusolver_getrf_example.cu

*/

#include <iostream>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include "../include/checks.h"
#include "../include/helper.h"

int main() {
    const int N = 2; // Matrix dimension (NxN)
    const int lda = N; // Leading dimension of A (since row-major, lda = N)

    /*
    Let matrix A
    |4.0 1.0|
    |3.0 1.0| 
    */ 

    // (1) Input matrix (Column-major order)
    float mtxA_h[lda * N] = {
        4.0f, 3.0f, 
        1.0f, 1.0f
    };


    bool debug = true;

    // (2) Device memory for matrix A
    float* mtxA_d = NULL;
    CHECK(cudaMalloc((void**)&mtxA_d, lda * N * sizeof(float)));
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, lda * N * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\nmtxA_d before cusolverDnSgetrf\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }
    
    // (4) Pivot indices and info
    int* pivot_d = NULL;
    int* info_d = NULL;
    CHECK(cudaMalloc((void**)&pivot_d, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&info_d, sizeof(int)));

    // (5) Set up cuSOLVER handle
    cusolverDnHandle_t cusolverHandler = NULL;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandler));

    // (6) Workspace size and allocation
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(cusolverHandler, N, N, mtxA_d, lda, &lwork));

    float* work_d = NULL;
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(float)));

    // (7) Perform LU factorization
    CHECK_CUSOLVER(cusolverDnSgetrf(cusolverHandler, N, N, mtxA_d, lda, work_d, pivot_d, info_d));


    if(debug){
        // (8) Check the LU decompostion success or not.
        int info_h = 0; // Host-side info variable
        CHECK(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

        // Check the info value
        if (info_h == 0) {
            printf("\nLU factorization successful.\n");
        } else if (info_h > 0) {
            printf("\nLU factorization failed. Matrix is singular at U[%d][%d].\n", info_h, info_h);
        } else if (info_h < 0) {
            printf("\nLU factorization failed. Illegal argument at position %d.\n", -info_h);
        }

        printf("\nmtxA_d after cusolverDnSgetrf\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }

    // (9) Feel free to explore
    // 1. solve AX = I to get inverse with cusolverDnSgetrs
    // 2. Extract L and U explicitly 

    // (10) Clean up
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(pivot_d));
    CHECK(cudaFree(info_d));
    CHECK(cudaFree(work_d));
    
    return 0;
}

/*
Sample Run

mtxA_d before cusolverDnSgetrf
4.000000 1.000000 
3.000000 1.000000 

LU factorization successful.

mtxA_d after cusolverDnSgetrf
4.000000 1.000000 
0.750000 0.250000 


*/