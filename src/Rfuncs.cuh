#pragma once
#include <iostream>
#include "CommonHeader.cuh"
#include "MatrixOperation.cuh"
#include "SparseMatrix.cuh"


extern "C" __declspec(dllexport)
void upload(double* dataFrame, double * rowInd, double * colInd, double * size, double* offset, double* address);
extern "C" __declspec(dllexport)
void download(double* data, double * rowInd, double * colInd, double * address);
extern "C" __declspec(dllexport)
void colSums( int *direction, double * address,double *result);