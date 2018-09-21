#pragma once
#include "commonHeader.cuh"
#include "Tools.cuh"


template<class T, class Ind> __global__
void CUDA_matrixSum_kernel(int M, int N, int totalANnz, T *dCscValA, Ind *dCscRowPtrA, Ind *dCscColIndA, float* result, bool isCol = false);

template __global__ void CUDA_matrixSum_kernel(int M, int N, int totalANnz, float *dCscValA, LARGEINDEX *dCscRowPtrA, LARGEINDEX *dCscColIndA, float* result, bool isCol = false);
template __global__ void CUDA_matrixSum_kernel(int M, int N, int totalANnz, float *dCscValA, int *dCscRowPtrA, int *dCscColIndA, float* result, bool isCol = false);
template __global__ void CUDA_matrixSum_kernel(int M, int N, int totalANnz, double *dCscValA, LARGEINDEX *dCscRowPtrA, LARGEINDEX *dCscColIndA, float* result, bool isCol = false);

template<class T>
void CUDA_matrixSum(int M, int N, int totalANnz, T *dCscValA, int *dCscRowPtrA, int *dCscColIndA, T* result, bool isCol);

//template void CUDA_matrixSum(int M, int N, int totalANnz, double *dCscValA, int *dCscRowPtrA, int *dCscColIndA, double* result, bool isCol = false);

template void CUDA_matrixSum(int M, int N, int totalANnz, double *dCscValA, int *dCscRowPtrA, int *dCscColIndA, double* result, bool isCol = false);



