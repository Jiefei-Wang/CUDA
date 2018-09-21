#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "CommonHeader.cuh"


void errorHandle(cudaError_t error,std::string msg);
__host__ __device__ void errorPrint(char*);
LARGEINDEX *  getIndexFromR(double* source, LARGEINDEX length);
//template<class T1, class T2>
//void  transformData(T1* target, T2* source, LARGEINDEX length);
template<class T1, class T2>
void  transformData(T1* target, T2* source, LARGEINDEX length) {
	for (LARGEINDEX i = 0; i < length; i++) {
		target[i] = source[i];
	}
}
template<class T> void fillWithNum(T* target, T number, int n);
template<class T> void print_partial_matrix(char* title, T *M, int nrows, int ncols, int max_row = -1,
	int max_col = -1);


template void transformData(double* target, LARGEINDEX* source, LARGEINDEX length);
template void transformData(LARGEINDEX* target, double* source, LARGEINDEX length);
template void transformData(LARGEINDEX* target, LARGEINDEX* source, LARGEINDEX length);

template void fillWithNum(double* target, double number, int n);
template void fillWithNum(float* target, float number, int n);
template void fillWithNum(int* target, int number, int n);

template void print_partial_matrix(char* title, double *M, int nrows, int ncols, int max_row = -1,
	int max_col = -1);
template void print_partial_matrix(char* title, float *M, int nrows, int ncols, int max_row = -1,
	int max_col = -1);
template void print_partial_matrix(char* title, int *M, int nrows, int ncols, int max_row = -1,
	int max_col = -1);


