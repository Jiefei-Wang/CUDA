#pragma once
#include "CommonHeader.cuh"
#include "Tools.cuh"


template<class T, class Ind> __global__
void CUDA_matrixSum_kernel(int M, int N, int totalANnz, T *dCscValA, Ind *dCscRowPtrA, Ind *dCscColIndA, float* result, bool isCol = false);


template<class T>
void CUDA_matrixSum(int M, int N, int totalANnz, T *dCscValA, int *dCscRowPtrA, int *dCscColIndA, T* result, bool isCol);






