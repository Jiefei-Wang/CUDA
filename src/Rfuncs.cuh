#pragma once
#include <iostream>
#include "CommonHeader.cuh"
#include "MatrixOperation.cuh"
#include "SparseMatrix.cuh"
#include "R_ext/libextern.h"

extern "C" #LibExport
void upload(double* dataFrame, double * rowInd, double * colInd, double * size, double* offset, double* address);
extern "C" #LibExport
void download(double* data, double * rowInd, double * colInd, double * address);
extern "C" #LibExport
void colSums( int *direction, double * address,double *result);