#include "MatrixOperation.cuh"

#include <cusparse_v2.h>

template<class T,class Ind> __global__
void CUDA_matrixSum_kernel(int M, int N, int totalANnz, T *dCscValA, Ind *dCscRowPtrA, Ind *dCscColIndA, float* result, bool isCol) {
	LARGEINDEX id = threadIdx.x + blockIdx.x*blockDim.x;
	LARGEINDEX step = gridDim.x*blockDim.x;
	if (isCol) {
		//Col sum
		for (LARGEINDEX colID = id; colID < N; colID = colID + step) {
			int nonzero = dCscColIndA[colID + 1] - dCscColIndA[colID];
			for (LARGEINDEX i = 0; i < nonzero; i++) {
				result[colID] = result[colID] + dCscValA[dCscColIndA[colID] + i];
			}
		}
	}
	else {
		//Row sum
		LARGEINDEX rowID;
		for (LARGEINDEX colID = id; colID < N; colID = colID + step) {
			int nonzero = dCscColIndA[colID + 1] - dCscColIndA[colID];
			for (LARGEINDEX i = 0; i < nonzero; i++) {
				rowID = dCscRowPtrA[dCscColIndA[colID] + i];
				atomicAdd(result + rowID,(float) dCscValA[dCscColIndA[colID] + i]);
			}
		}
	}
}




template<class T>
void CUDA_matrixSum(int M, int N, int totalANnz, T *dCscValA, int *dCscRowPtrA, int *dCscColIndA,T* result, bool isCol) {
	
	cusparseHandle_t handle=0;
	cusparseCreate(&handle);
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

	T *B, *dB;
	T *dC;
	B = new T[K1];
	for (size_t i = 0; i < K1; ++i) {
		B[i] = 1;
	}

	cudaMalloc((void **)&dB, sizeof(*B) * K1);
	cudaMalloc((void **)&dC, sizeof(*dC) * K2);

	cudaMemcpy(dB, B, sizeof(*B) * K1, cudaMemcpyHostToDevice);

	cusparseMatDescr_t Adescr = 0;
	cusparseCreateMatDescr(&Adescr);
	cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO);

	T alpha = 1;
	T beta = 0;

	clock_t begin = clock();
	cusparseDcsrmv(handle, op, N, M,
		totalANnz, &alpha, Adescr, dCscValA, dCscColIndA, dCscRowPtrA, dB, &beta, dC);

	cudaDeviceSynchronize();
	clock_t end = clock();
	double timeSec = (end - begin) / static_cast<double>(CLOCKS_PER_SEC);
	//std::cout << "Elapsed time: " << timeSec << std::endl;
	// Copy the result vector back to the host
	cudaMemcpy(result, dC, sizeof(*result) * K2, cudaMemcpyDeviceToHost);

	//print_partial_matrix("C:", C, 1, K2, 1, K2);

	delete[](B);
	cudaFree(dB);
	cudaFree(dC);
//	cudaFree(dANnzPerCol);
	cusparseDestroyMatDescr(Adescr);
	cusparseDestroy(handle);
}













template __global__ void CUDA_matrixSum_kernel(int M, int N, int totalANnz, float *dCscValA, LARGEINDEX *dCscRowPtrA, LARGEINDEX *dCscColIndA, float* result, bool isCol = false);
template __global__ void CUDA_matrixSum_kernel(int M, int N, int totalANnz, float *dCscValA, int *dCscRowPtrA, int *dCscColIndA, float* result, bool isCol = false);
template __global__ void CUDA_matrixSum_kernel(int M, int N, int totalANnz, double *dCscValA, LARGEINDEX *dCscRowPtrA, LARGEINDEX *dCscColIndA, float* result, bool isCol = false);
template void CUDA_matrixSum(int M, int N, int totalANnz, double *dCscValA, int *dCscRowPtrA, int *dCscColIndA, double* result, bool isCol = false);