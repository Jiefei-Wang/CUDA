#include "Rfuncs.cuh"
using namespace std;
template <class T> void print(T* a,int length, char* note) {
	cout << note<<" : ";
	for (int i = 0; i < length-1; i++) {
		cout << a[i] << "," ;
	}
	cout << a[length - 1]<<endl;
}
extern "C" LibExport
void upload(double* dataFrame, double * rowInd, double * colInd, double * size, double* offset, double* address) {
	LARGEINDEX * size_l = getIndexFromR(size, 5);
	LARGEINDEX * rowInd_l = getIndexFromR(rowInd, size_l[1]);
	LARGEINDEX * colInd_l = getIndexFromR(colInd, size_l[2]);
	LARGEINDEX * offset_l = getIndexFromR(offset, 1);

	SparseMatrix<double> dataMatrix(dataFrame, rowInd_l, colInd_l, size_l, offset_l, Deviceloc::Host);
	dataMatrix.HostToDevice();
	
	dataMatrix.setPackedInfo((LARGEINDEX *)address);
}
extern "C" LibExport
void download(double* data, double * rowInd, double * colInd, double * address) {
	SparseMatrix<double> dataMatrix((LARGEINDEX *)address);
	dataMatrix.deviceToHost();
	transformData(data, dataMatrix.matrixData, dataMatrix.matrixSize);
	transformData(rowInd, dataMatrix.rowInd, dataMatrix.rowIndexLen);
	transformData(colInd, dataMatrix.colInd, dataMatrix.colIndexLen);

	//dataMatrix.print();
}


int * cpyfunc(LARGEINDEX* src, int n) {
	int* tmp = new int[n];
	LARGEINDEX* tmp1 = new LARGEINDEX[n];

	cudaMemcpy(tmp1, src, sizeof(*tmp1) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		tmp[i] = tmp1[i];
	}
	int* target = 0;
	cudaMalloc((void **)&target, sizeof(*target) * n);
	cudaMemcpy(target, tmp, sizeof(*tmp) * n, cudaMemcpyHostToDevice);
	delete[] tmp;
	delete[] tmp1;
	return(target);

}
extern "C" LibExport
void colSums( int *direction,double * address, double * result)
{
	int nresult;
	SparseMatrix<double> dataMatrix((LARGEINDEX *)address);
	if (*direction == 1)
		nresult = dataMatrix.matrixRowDim;
	else
		nresult = dataMatrix.matrixColDim;

	fillWithNum(result, 0.0, nresult);

	
	

	float* dev_result = 0;
	float* host_result = new float[nresult];
	fillWithNum(host_result, (float)0, nresult);
	cudaMalloc((void **)&dev_result, sizeof(*dev_result) * nresult);
	cudaMemcpy(dev_result, host_result, sizeof(*host_result) * nresult, cudaMemcpyHostToDevice);

	CUDA_matrixSum_kernel << <BlockNum, ThreadNum >> >(dataMatrix.matrixRowDim, dataMatrix.matrixColDim, dataMatrix.matrixSize,
		dataMatrix.dev_matrixData, dataMatrix.dev_rowInd, dataMatrix.dev_colInd, dev_result, *direction == 2);

	cudaDeviceSynchronize();
	cudaMemcpy(host_result, dev_result, sizeof(*host_result) * nresult, cudaMemcpyDeviceToHost);
	for (int i = 0; i < nresult; i++) {
		result[i] = host_result[i];
	}
	delete[] host_result;
	cudaFree(dev_result);
/*
	int* dCscRowPtrA = cpyfunc(dataMatrix.dev_rowInd, dataMatrix.matrixSize);
	int* dCscColIndA = cpyfunc(dataMatrix.dev_colInd, dataMatrix.matrixColDim + 1);

	CUDA_matrixSum(dataMatrix.matrixRowDim, dataMatrix.matrixColDim, dataMatrix.matrixSize,
		dataMatrix.dev_matrixData, dCscRowPtrA, dCscColIndA, result, *direction == 2);
		*/
	//print(result, nresult, "result");


}

