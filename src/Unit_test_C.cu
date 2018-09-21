#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include <ctime>
#include "CommonHeader.cuh"
#include <cusparse_v2.h>
#include "Test_tools.cuh"
#include "MatrixOperation.cuh"
#include "Unit_test.cuh"

/*
* Generate random dense matrix A in column-major order, while rounding some
* elements down to zero to ensure it is sparse.
*/
template<class T>
int generate_random_dense_matrix(int M, int N, T **outA)
{
	int i, j;
	double rMax = (double)RAND_MAX;
	T *A = (T *)malloc(sizeof(T) * M * N);
	int totalNnz = 0;
	double progress = 0;
	for (j = 0; j < N; j++)
	{
		for (i = 0; i < M; i++)
		{
			int r = rand();
			T *curr = A + (j * M + i);

			if (r % 100 >= 50)
			{
				*curr = 0.0f;
			}
			else
			{
				double dr = (double)r;
				*curr = (int)(dr / rMax * 100.0);
			}

			if (*curr != 0.0f)
			{
				totalNnz++;
			}
		}
		if (j > N*progress) {
			//std::cout << "Progress : " << progress << std::endl;
			progress = progress + 0.05;
		}
	}

	*outA = A;
	return totalNnz;
}


template<class T>
T* CPU_matrixSum(cusparseHandle_t handle, int M, int N, int totalANnz, int *dANnzPerCol, T *dCscValA, int *dCscRowPtrA, int *dCscColIndA, bool isCol = false) {
	int K1;
	int K2;
	cusparseOperation_t op;
	if (isCol) {
		//Col sum
		K1 = M;
		K2 = N;
	}
	else {
		//Row sum
		K1 = N;
		K2 = M;
	}
	float *C = new float[K2];

	//Download the CSC data
	float *hCscValA = new float[totalANnz];
	int *hCscRowPtrA = new int[totalANnz];
	int *hCscColIndA = new int[N + 1];
	cudaMemcpy(hCscValA, dCscValA, sizeof(*dCscValA) * totalANnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(hCscRowPtrA, dCscRowPtrA, sizeof(*dCscRowPtrA) * totalANnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(hCscColIndA, dCscColIndA, sizeof(*dCscColIndA) * (N + 1), cudaMemcpyDeviceToHost);


	/*
	print_partial_matrix("D:", hCscValA, 1, totalANnz);
	print_partial_matrix("D:", hCscRowPtrA, 1, totalANnz);
	print_partial_matrix("D:", hCscColIndA, 1, N + 1);*/

	clock_t begin = clock();
	for (int i = 0; i < K2; i++) {
		C[i] = 0;
	}
	if (isCol) {
		for (int colID = 0; colID < N; colID++) {
			int nonzero = hCscColIndA[colID + 1] - hCscColIndA[colID];
			for (int i = 0; i < nonzero; i++) {
				C[colID] = C[colID] + hCscValA[hCscColIndA[colID] + i];
			}
		}
	}
	else {
		int rowID;
		for (int colID = 0; colID < N; colID++) {
			int nonzero = hCscColIndA[colID + 1] - hCscColIndA[colID];
			for (int i = 0; i < nonzero; i++) {
				rowID = hCscRowPtrA[hCscColIndA[colID] + i];
				C[rowID] = C[rowID] + hCscValA[hCscColIndA[colID] + i];
			}
		}
	}
	clock_t end = clock();
	double timeSec = (end - begin) / static_cast<double>(CLOCKS_PER_SEC);
	std::cout << "CPU Elapsed time: " << timeSec << std::endl;
	return C;
}

void test_colsum(int M,int N,bool isCol) {
	int K1;
	int K2;
	cusparseOperation_t op;
	if (isCol) {
		//Col sum
		op = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;
		K1 = M;
		K2 = N;
	}
	else {
		//Row sum
		op = cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE;
		K1 = N;
		K2 = M;
	}

	// Create the cuSPARSE handle
	cusparseHandle_t handle = 0;
	cusparseCreate(&handle);

	// Generate input
	srand(9384);
	float *A, *dA;
	int trueANnz = generate_random_dense_matrix(M, N, &A);
	//print_partial_matrix("A:",A, M, N, M, N);

	// Allocate device memory for vectors and the dense form of the matrix A
	cudaMalloc((void **)&dA, sizeof(*A) * M * N);

	// Construct a descriptor of the matrix A
	cusparseMatDescr_t Adescr = 0;
	cusparseCreateMatDescr(&Adescr);
	cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO);
	// Transfer the input vectors and dense matrix A to the device
	cudaMemcpy(dA, A, sizeof(*A) * M * N, cudaMemcpyHostToDevice);

	// Compute the number of non-zero elements in A
	int totalANnz;
	int *dANnzPerCol;
	cudaMalloc((void **)&dANnzPerCol, sizeof(int) * N);
	cusparseSnnz(handle, CUSPARSE_DIRECTION_COLUMN, M, N, Adescr,
		dA, M, dANnzPerCol, &totalANnz);

	if (totalANnz != trueANnz)
	{
		fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
			"value: expected %d but got %d\n", trueANnz, totalANnz);
		return;
	}

	// Allocate device memory to store the sparse CSC representation of A
	float *dCscValA;
	int *dCscRowPtrA;
	int *dCscColIndA;
	cudaMalloc((void **)&dCscValA, sizeof(*A) * totalANnz);
	cudaMalloc((void **)&dCscRowPtrA, sizeof(*dCscRowPtrA) * totalANnz);
	cudaMalloc((void **)&dCscColIndA, sizeof(*dCscColIndA) * (N + 1));

	// Convert A from a dense formatting to a CSR formatting, using the GPU
	cusparseSdense2csc(handle, M, N, Adescr, dA, M, dANnzPerCol,
		dCscValA, dCscRowPtrA, dCscColIndA);
	//Delete the unused matrix
	free(A);
	cudaFree(dA);
	//std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;


	float *C_cpu = CPU_matrixSum(handle, M, N, totalANnz, dANnzPerCol, dCscValA, dCscRowPtrA, dCscColIndA, isCol);
	//print_partial_matrix("C cpu:", C_cpu, 1, K2);


	float * C_cuda = new float[K2];
	//CUDA_matrixSum(M, N, totalANnz, dCscValA, dCscRowPtrA, dCscColIndA, C_cuda,isCol);
	//print_partial_matrix("C cuda:", C_cuda, 1, K2);


	float *C_cuda_kernel = new float[K2];
	fillWithNum(C_cuda_kernel, (float)0, K2);
	float *dev_C_cuda = 0;
	cudaMalloc((void **)&dev_C_cuda, sizeof(*dev_C_cuda) * K2);
	cudaMemcpy(dev_C_cuda, C_cuda_kernel, sizeof(*dev_C_cuda) * K2, cudaMemcpyHostToDevice);

	clock_t begin = clock();
	CUDA_matrixSum_kernel << <BlockNum, ThreadNum >> >(M, N, totalANnz, dCscValA, dCscRowPtrA, dCscColIndA, dev_C_cuda, isCol);
	clock_t end = clock();
	double timeSec = (end - begin) / static_cast<double>(CLOCKS_PER_SEC);
	std::cout << "GPU Elapsed time: " << timeSec << std::endl;
	cudaDeviceSynchronize();
	cudaMemcpy(C_cuda_kernel, dev_C_cuda, sizeof(*dev_C_cuda) * K2, cudaMemcpyDeviceToHost);
	//print_partial_matrix("C cuda kernel:", C_cuda_kernel, 1, K2);
	cudaFree(dev_C_cuda);

	double error;
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	
	error = checkValue(C_cpu, C_cuda_kernel, K2);
	if (error <= 0.001&&error>=-0.001) {
		std::cout << "Matrix colsum test report: Pass" << std::endl;
	}
	else {
		std::cout << "Matrix colsum test report: Error is : " << error << std::endl;
	}

	delete[] C_cuda;
	delete[] C_cuda_kernel;
	delete[] C_cpu;

	cusparseDestroyMatDescr(Adescr);
	cudaFree(dANnzPerCol);
	cudaFree(dCscValA);
	cudaFree(dCscRowPtrA);
	cudaFree(dCscColIndA);
	cusparseDestroy(handle);
}