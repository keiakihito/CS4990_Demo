#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <sys/time.h>

// helper function CUDA error checking and initialization
#include "checks.h"  



// Time tracker for each iteration
double myCPUTimer();

template<typename T>
void print_vector(const T *d_val, int size);

template<typename T>
void print_mtx_clm_d(const T *mtx_d, int numOfRow, int numOfClm);

//Print matrix column major
template <typename T>
void print_mtx_clm_d(const T *mtx_d, int numOfRow, int numOfClm);

// Print matrix in row-major order (Host)
template <typename T>
void print_mtx_row_h(const T *mtx_h, int numOfRow, int numOfClm);

template <typename T>
void print_mtx_clm_h(const T *mtx_h, int numOfRow, int numOfClm);

// // = = = Function signatures = = = = 

// Time tracker for each iteration
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}


template<typename T>
void print_vector_d(const T *d_val, int size) {
    // Allocate memory on the host
    T *check_r = (T *)malloc(sizeof(T) * size);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    // cudaError_t err = cudaMemcpy(check_r, d_val, size * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK(cudaMemcpy(check_r, d_val, size * sizeof(T), cudaMemcpyDeviceToHost));

    // Print the values to check them
    for (int i = 0; i < size; i++) {
        // Use the correct format based on the type of T
        if constexpr (std::is_same<T, float>::value) {
            printf("%.10f\n", check_r[i]); // For float
        } else if constexpr (std::is_same<T, int>::value) {
            printf("%d\n", check_r[i]); // For int
        } else {
            printf("Unsupported data type\n");
            break;
        }
    }

    // Free allocated memory
    free(check_r);
} // print_vector

// Print matrix in row-major order
template <typename T>
void print_mtx_row_d(const T *mtx_d, int numOfRow, int numOfClm) {
    // Allocate memory on the host
    T *check_r = (T *)malloc(sizeof(T) * numOfRow * numOfClm);

    if (check_r == NULL) {
        printf("Failed to allocate host memory\n");
        return;
    }

    // Copy data from device to host
    CHECK(cudaMemcpy(check_r, mtx_d, numOfRow * numOfClm * sizeof(T), cudaMemcpyDeviceToHost));

    // Print matrix in row-major format
    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++) {
        for (int clWkr = 0; clWkr < numOfClm; clWkr++) {
            // Access element in row-major order
            printf("%f ", check_r[rwWkr * numOfClm + clWkr]);
        } // end of column walker
        printf("\n");
    }// end of row walker

    // Free host memory
    free(check_r);
}

//Print matrix column major
template <typename T>
void print_mtx_clm_d(const T *mtx_d, int numOfRow, int numOfClm){
    //Allocate memory oh the host
    T *check_r = (T *)malloc(sizeof(T) * numOfRow * numOfClm);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    CHECK(cudaMemcpy(check_r, mtx_d, numOfRow * numOfClm * sizeof(T), cudaMemcpyDeviceToHost));

    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++){
        for(int clWkr = 0; clWkr < numOfClm; clWkr++){
            printf("%f ", check_r[clWkr*numOfRow + rwWkr]);
        }// end of column walker
        printf("\n");
    }// end of row walker
    
    // Free host memory
    free(check_r);
} // end of print_mtx_h

// Print matrix in row-major order (Host)
template <typename T>
void print_mtx_row_h(const T *mtx_h, int numOfRow, int numOfClm) {
    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++) {
        for (int clWkr = 0; clWkr < numOfClm; clWkr++) {
            // Access element in row-major order
            printf("%f ", mtx_h[rwWkr * numOfClm + clWkr]);
        }
        printf("\n");
    }
}

// Print matrix in column-major order (Host)
template <typename T>
void print_mtx_clm_h(const T *mtx_h, int numOfRow, int numOfClm) {
    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++) {
        for (int clWkr = 0; clWkr < numOfClm; clWkr++) {
            // Access element in column-major order
            printf("%f ", mtx_h[clWkr * numOfRow + rwWkr]);
        }
        printf("\n");
    }
}












#endif // HELPER_H