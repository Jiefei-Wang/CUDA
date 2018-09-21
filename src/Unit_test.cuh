#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include "CommonHeader.cuh"

#include "Test_tools.cuh"
#include "Rfuncs.cuh"


#include "MatrixOperation.cuh"



void test_matrixUpload();

void test_colsum(int M, int N, bool isCol);