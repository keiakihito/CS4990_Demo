#include <iostream>
#include <cusparse.h>
#include <cuda_runtime.h>

#include "../include/checks.h"  
#include "../include/helper.h" 

int main(){
    // Matrix dimension
    const int numOfRow = 3;
    const int numOfClm = 3;
    const int nnz = 4; // Number of non-zero value

    const float alpha = 1.0f;
    const float beta = 0.0f;

  
    /*
    CSR format of Matrix A (Row-Major)
    Let Sparse Matrix A = 
    |1.0 0.0 0.0|
    |0.0 2.0 3.0|
    |0.0 0.0 4.0|
    */
    int csrOffsets_h[numOfRow+1] = {0, 1, 3, 4};
    int columns_h[nnz] = {0, 1, 2, 2};
    float values_h[nnz] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Dense vector x and y
    float vecX_h[] = {1.0f, 2.0f, 3.0f};
    float vecY_h[] = {0.0f, 0.0f, 0.0f}; // Store result later

    // Device memeory assignments
    int *csrOffsets_d = NULL;
    int *columns_d = NULL;
    float *values_d = NULL;
    float *vecX_d = NULL;
    float *vecY_d = NULL;

    bool debug = true;

    // Allocate device memory
    CHECK(cudaMalloc((void**)&csrOffsets_d, (numOfRow+1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&columns_d, nnz * sizeof(int)));
    CHECK(cudaMalloc((void**)&values_d, nnz * sizeof(float)));
    CHECK(cudaMalloc((void**)&vecX_d, numOfClm * sizeof(float)));
    CHECK(cudaMalloc((void**)&vecY_d, numOfClm * sizeof(float)));

    // Copy data from host to device
    CHECK(cudaMemcpy(csrOffsets_d, csrOffsets_h, (numOfRow + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(columns_d, columns_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(values_d, values_h, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vecX_d, vecX_h, numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vecY_d, vecY_h, numOfRow * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\ncsrOffsets_d:\n");
        print_vector_d(csrOffsets_d, numOfRow+1);

        printf("\ncolumns_d:\n");
        print_vector_d(columns_d, nnz);

        printf("\nvaluse_d:\n");
        print_vector_d(values_d, nnz);

        printf("\nvecX_d: \n");
        print_vector_d(vecX_d, numOfRow);
        
        printf("\nvecY_d: \n");
        print_vector_d(vecY_d, numOfRow);
    }

    // Create cuSPARSE handler
    cusparseHandle_t cusparseHandler = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandler));

    // Create matrix and vector descriptors
    cusparseSpMatDescr_t mtxA_des = NULL;
    cusparseDnVecDescr_t vecX_des = NULL;
    cusparseDnVecDescr_t vecY_des = NULL;

    // Define sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&mtxA_des, numOfRow, numOfClm, nnz, csrOffsets_d, columns_d, values_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Define dense vecotrs x and y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX_des, numOfRow, vecX_d, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY_des, numOfRow, vecY_d, CUDA_R_32F));

    // Allocate workspase for SpMV
    size_t bufferSize = 0;
    void *buffer_d = NULL;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,mtxA_des, vecX_des, &beta, vecY_des, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK(cudaMalloc(&buffer_d, bufferSize));

    if(debug){
        printf("\nBuffer size (bytes): %zu\n", bufferSize);
    }

    // Perform SpMV
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA_des, vecX_des, &beta, vecY_des, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer_d));

    // Copy refult from device to host
    CHECK(cudaMemcpy(vecY_h, vecY_d, numOfRow * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nResult Vector vecY_h:\n");
    print_mtx_clm_h(vecY_h, numOfRow, 1);

    // Free memory
    CHECK_CUSPARSE(cusparseDestroySpMat(mtxA_des));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX_des));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY_des));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandler));
    CHECK(cudaFree(csrOffsets_d));
    CHECK(cudaFree(columns_d));
    CHECK(cudaFree(values_d));
    CHECK(cudaFree(vecX_d));
    CHECK(cudaFree(vecY_d));
    CHECK(cudaFree(buffer_d));

    return 0;
}

/*
Sample Run

csrOffsets_d:
0
1
3
4

columns_d:
0
1
2
2

valuse_d:
1.0000000000
2.0000000000
3.0000000000
4.0000000000

vecX_d: 
1.0000000000
2.0000000000
3.0000000000

vecY_d: 
0.0000000000
0.0000000000
0.0000000000

Buffer size (bytes): 8

Result Vector vecY_h:
1.000000 
13.000000 
12.000000 


 */