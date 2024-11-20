#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <nvtx3/nvToolsExt.h> // For nsight profiling

#include "../include/checks.h"
#include "../include/helper.h"

// divide integers with rounding up, ensuring that any remainder results in an additional block.
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Naive cublasSgemm
__global__ void sgemm_naive(int m, int n, int k, float alpha, const float *mtxA_d,
                            const float *mtxB_d, float beta, float *mtxC_d);
void run_sgemm_naive(int m, int n, int k, float alpha, float *mtxA_d,
                     float *mtxB_d, float beta, float *mtxC_d);
void initializeRandom(float mtxB_h[], int numOfRow, int numOfClm);
void fillUpArray(int numOfRow, int numOfClm, float *ptr_h);
bool verify(float* cuBLAS_d, float* kernel_d, unsigned int nRows, unsigned int nCols);




int main()
{
    // Warm up kernel
    nvtxRangePush("Warm up kernel");
    cudaDeviceSynchronize();
    nvtxRangePop();


    double startTime, endTime;
    float alpha = 1.0f;
    float beta = 0.0f;
	
	// 4096 by 4096 matrix multiplication
    int m = 4096;
    int n = 4096;
    int k = 4096;

    bool debug = false;

    // Set up handler
    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));

    // Fill up random values
    float* mtxA_h = (float*)malloc((m*k) * sizeof(float));
    fillUpArray(m, k, mtxA_h);

    float* mtxB_h = (float*)malloc((k*n) * sizeof(float));
    fillUpArray(k, n, mtxB_h);

    //Initialize to 0
    float* mtxC_cuBLAS_h = (float*)calloc(m*n, sizeof(float));
    float* mtxC_kernel_h = (float*)calloc(m*n, sizeof(float));


    //(1) Allocate memory on the GPU
    float* mtxA_d = NULL;
    float* mtxB_d = NULL;
    float* mtxC_cuBLAS_d = NULL;
    float* mtxC_kernel_d = NULL;
    CHECK(cudaMalloc((void**)&mtxA_d, m * k * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxB_d, k * n * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxC_cuBLAS_d, m * n * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxC_kernel_d, m * n * sizeof(float)));
    
    
    //(2) Copy Data from CPU to GPU
    startTime = myCPUTimer();
    nvtxRangePush("Memory transfer from host to device");
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, k * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxC_cuBLAS_d, mtxC_cuBLAS_h, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxC_kernel_d, mtxC_kernel_h, m * n * sizeof(float), cudaMemcpyHostToDevice));
    nvtxRangePop();
    endTime = myCPUTimer();
    printf("\nMemory transfer from host to device: %f ms \n", endTime - startTime);

    // (3) Calling cuBLAS
    startTime = myCPUTimer();
    nvtxRangePush("Start cublasSgemm");
    CHECK_CUBLAS(cublasSgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, mtxA_d, m, mtxB_d, k, &beta, mtxC_cuBLAS_d, m));
    cudaDeviceSynchronize(); // Ensure cuBLAS operation is complete
    nvtxRangePop();
    endTime = myCPUTimer();
    printf("\ncuBLAS: %f ms \n", endTime - startTime);

    // (4) Calling kernels
    startTime = myCPUTimer();
    nvtxRangePush("1_naive_kernel");
    run_sgemm_naive(m, n, k, alpha, mtxA_d, mtxB_d, beta, mtxC_kernel_d);
    cudaDeviceSynchronize(); // Ensure kernel operation is complete
    nvtxRangePop();
    endTime = myCPUTimer();
    printf("\n1_naive: %f ms \n", endTime - startTime);

    //(5) Copy result GPU to CPU
    startTime = myCPUTimer();
    nvtxRangePush("Memory transfer from host to device");
    CHECK(cudaMemcpy(mtxC_cuBLAS_h, mtxC_cuBLAS_d, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(mtxC_kernel_h, mtxC_kernel_d, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();
    endTime = myCPUTimer();
    printf("\nMemory transfer from device to host: %f ms \n", endTime - startTime);
    
    // Verify manually
    if(debug){
        printf("\ncuBLAS_h\n");
        for (int i = 0; i < std::min(m, 10); i++) {
            for (int j = 0; j < std::min(n, 10); j++) {
                printf("%f ", mtxC_cuBLAS_h[i * n + j]);
            }
            printf("\n");
        }

        printf("\n\nmtxC_kernel_h\n");
        for (int i = 0; i < std::min(m, 10); i++) {
            for (int j = 0; j < std::min(n, 10); j++) {
                printf("%f ", mtxC_kernel_h[i * n + j]);
            }
            printf("\n");
        }
    }


    // Verification process
    bool check = verify(mtxC_cuBLAS_h, mtxC_kernel_h, m, n);
    if(check == true){printf("\nVERIFY: kernel PASSED👍👍👍\n");}
    else{printf("!!!Error Detected!!!"); return -1;}
    
    

  
    //(4) Free GPU and CPU memory
    CHECK_CUBLAS(cublasDestroy(cublasHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxB_d));
    CHECK(cudaFree(mtxC_cuBLAS_d));
    CHECK(cudaFree(mtxC_kernel_d));
    free(mtxA_h);
    free(mtxB_h);
    free(mtxC_cuBLAS_h);
    free(mtxC_kernel_h);
    

    return 0;
}

// Navive sgemm
__global__ void sgemm_naive(int m, int n, int k, float alpha, const float *mtxA_d,
                            const float *mtxB_d, float beta, float *mtxC_d)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // if statement is necessary to make thins work under title quantization
    if (x < m && y < n)
    {
        float temp = 0.0f;
        for (int i = 0; i < k; i++)
        {
            temp += mtxA_d[x * k + i] * mtxB_d[i * n + y];
        }
        // C = alpha(A@B) + beta*C
        mtxC_d[x * n + y] = alpha * temp + beta * mtxC_d[x * n + y];
    } // end of if
}

void run_sgemm_naive(int m, int n, int k, float alpha, float *mtxA_d,
                     float *mtxB_d, float beta, float *mtxC_d)
{
    dim3 gridDim(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    dim3 blockDim(32, 32); // Represent (row, col) 2D mapping
    sgemm_naive<<<gridDim, blockDim>>>(m, n, k, alpha, mtxA_d, mtxB_d, beta, mtxC_d);
}


//~~~Helper fundtions
// Initialize random values between -1 and 1
void initializeRandom(float mtxB_h[], int numOfRow, int numOfClm)
{
    srand(time(NULL));

    for (int wkr = 0; wkr < numOfRow * numOfClm; wkr++)
    {
        // Generate a random float between 0 and 1
        float rndVal = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        mtxB_h[wkr] = rndVal;
    }
} // end of initializeRandom


// Fill up random values
void fillUpArray(int numOfRow, int numOfClm, float *ptr_h)
{
    // srand(static_cast<unsigned int>(time(0))); //Random
    srand(42); // Set a fixed seed for reproducibility
    
    for (int wkr = 0; wkr < numOfRow * numOfClm; wkr++) {
        // ptr_h[wkr] = (float)wkr+ 1.0;
        ptr_h[wkr] = rand()%100/100.0;
    }
} // end of fillUpArray


// Verify calculate result
bool verify(float* cuBLAS_d, float* kernel_d, unsigned int nRows, unsigned int nCols)
{
    const double epsilon = 10e-3;
    double diff = 0.0f;
    for (int rWkr = 0; rWkr < nRows; rWkr++) {
        for (int cWkr = 0; cWkr < nCols; cWkr++) {
            diff = fabs(cuBLAS_d[rWkr*nCols + cWkr] - kernel_d[rWkr*nCols + cWkr]);
            if (diff > epsilon) {
                printf("\nrow: %d\n", rWkr);
                printf("column: %d\n", cWkr);
                printf("cuBLAS: %f\n", cuBLAS_d[rWkr*nCols + cWkr]);
                printf("kernel: %f\n", kernel_d[rWkr*nCols + cWkr]);
                return false;
            }
        } // end of inner loop
    }// end of outer loop
    return true;
} // end of verify
