
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cusparse_v2.h>


#include "SparseMatrix.cuh"
#include "Unit_test.cuh"
#include "MatrixOperation.cuh"

#include "Rfuncs.cuh"
#include "Test_tools.cuh"






int main(int argc, char **argv)
{
	bool isCol = false;

	int M = 10000;
	int N = 10000;
	test_colsum(M,N,isCol);
	//test_matrixUpload();
	
	return 0;
}