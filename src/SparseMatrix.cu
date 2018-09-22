#include "SparseMatrix.cuh"
template<class T> __alldev__
SparseMatrix<T>::SparseMatrix(LARGEINDEX * size, LARGEINDEX * offset, Deviceloc location)
{
	 matrixSize = size[0];
	 rowIndexLen = size[1];
	 colIndexLen = size[2];
	 matrixRowDim = size[3];
	 matrixColDim = size[4];
	 this->offset = *offset;
	switch (location) {
	case Deviceloc::Host:
		createHostMatrix(matrixSize, rowIndexLen, colIndexLen);
		break;
	case Deviceloc::Device:
		createDevMatrix(matrixSize, rowIndexLen, colIndexLen);
		break;
	case Deviceloc::Auto:
#ifndef  __CUDA_ARCH__
		createHostMatrix(matrixSize, rowIndexLen, colIndexLen);
#else
		createDevMatrix(matrixSize, rowIndexLen, colIndexLen);
#endif
		break;
	}
}
template <class T> __alldev__
SparseMatrix<T>::SparseMatrix(T* dataFrame, LARGEINDEX* rowInd, LARGEINDEX* colInd, LARGEINDEX* size, LARGEINDEX * offset, Deviceloc location = Deviceloc::Auto) {
	 matrixSize = size[0];
	 rowIndexLen = size[1];
	 colIndexLen = size[2];
	 matrixRowDim = size[3];
	 matrixColDim = size[4];

	 this->offset = *offset;

	switch (location) {
	case Deviceloc::Host:
		matrixData = dataFrame;
		this->rowInd = rowInd;
		this->colInd = colInd;
		break;
	case Deviceloc::Device:
		dev_matrixData = dataFrame;
		dev_rowInd = rowInd;
		dev_colInd = colInd;
		break;
	case Deviceloc::Auto:
#ifndef  __CUDA_ARCH__
		matrixData = dataFrame;
		this->rowInd = rowInd;
		this->colInd = colInd;
#else
		dev_matrixData = dataFrame;
		dev_rowInd = rowInd;
		dev_colInd = colInd;
#endif
		break;
	}
}

template<class T>
SparseMatrix<T>::SparseMatrix(LARGEINDEX * address)
{
	
	matrixSize = address[3];
	rowIndexLen = address[4];
	colIndexLen = address[5];
	matrixRowDim = address[6];
	matrixColDim = address[7];
	this->offset = address[8];
	dev_matrixData = reinterpret_cast<T*>(address[0]);
	dev_rowInd = reinterpret_cast<LARGEINDEX*>(address[1]);
	dev_colInd = reinterpret_cast<LARGEINDEX*>(address[2]);
}

template<class T>
void SparseMatrix<T>::HostToDevice()
{
	if (matrixData != nullptr) {
		if (dev_matrixData == nullptr) {
			delDevMatrix();
			createDevMatrix(matrixSize, rowIndexLen, colIndexLen);
		}
		errorHandle(cudaMemcpy(dev_matrixData, matrixData, (matrixSize) * sizeof(T)
			, cudaMemcpyHostToDevice), std::string("Error in matrix synchronization to device"));	
		errorHandle(cudaMemcpy(dev_rowInd, rowInd, (rowIndexLen) * sizeof(double)
			, cudaMemcpyHostToDevice), std::string("Error in matrix synchronization to device"));
		errorHandle(cudaMemcpy(dev_colInd, colInd, (colIndexLen) * sizeof(double)
			, cudaMemcpyHostToDevice), std::string("Error in matrix synchronization to device"));
	}
	else {
		errorPrint("Error in matrix synchronization: The host matrix does not exist");
	}
}

template<class T>
void SparseMatrix<T>::deviceToHost()
{
	if (dev_matrixData != nullptr) {
		if (matrixData == nullptr) {
			delHostMatrix();
			createHostMatrix(matrixSize, rowIndexLen,colIndexLen);
		}
		errorHandle(cudaMemcpy(matrixData,dev_matrixData, (matrixSize ) * sizeof(T), cudaMemcpyDeviceToHost), std::string("Error in matrix synchronization to host"));
		errorHandle(cudaMemcpy(rowInd, dev_rowInd, (rowIndexLen) * sizeof(double), cudaMemcpyDeviceToHost), std::string("Error in matrix synchronization to host"));
		errorHandle(cudaMemcpy(colInd, dev_colInd, (colIndexLen) * sizeof(double), cudaMemcpyDeviceToHost), std::string("Error in matrix synchronization to host"));
	}
	else {
		errorPrint("Error in matrix synchronization: The device matrix does not exist");
	}
}



template<class T>
void SparseMatrix<T>::delHostMatrix()
{
	if (matrixData != nullptr) {
		delete[] matrixData;
		matrixData = nullptr;
		delete[] rowInd;
		rowInd = nullptr;
		delete[] colInd;
		colInd = nullptr;
	}
}

template<class T>
void SparseMatrix<T>::delDevMatrix()
{
	if (dev_matrixData != nullptr) {
		errorHandle(cudaFree(dev_matrixData), std::string("Error in deleting device matrix"));
		errorHandle(cudaFree(dev_rowInd), std::string("Error in deleting device matrix"));
		errorHandle(cudaFree(dev_colInd), std::string("Error in deleting device matrix"));
		dev_matrixData = nullptr;
	}
}

template<class T>
void SparseMatrix<T>::createHostMatrix(LARGEINDEX matrixSize, LARGEINDEX rowIndexLen, LARGEINDEX colIndexLen)
{
	this->matrixSize = matrixSize;
	this->rowIndexLen = rowIndexLen;
	this->colIndexLen = colIndexLen;
	matrixData = new T[matrixSize];
	rowInd = new LARGEINDEX[rowIndexLen];
	colInd = new LARGEINDEX[colIndexLen];
}
template<class T>
void SparseMatrix<T>::createDevMatrix(LARGEINDEX matrixSize, LARGEINDEX rowIndexLen, LARGEINDEX colIndexLen)
{
	this->matrixSize = matrixSize;
	this->rowIndexLen = rowIndexLen;
	this->colIndexLen = colIndexLen;
	errorHandle(cudaMalloc(&dev_matrixData,
		(matrixSize) *sizeof(T)), std::string("Error in create device matrix"));
	errorHandle(cudaMalloc(&dev_rowInd,
		(rowIndexLen) * sizeof(LARGEINDEX)), std::string("Error in create device matrix"));
	errorHandle(cudaMalloc(&dev_colInd,
		(colIndexLen) * sizeof(LARGEINDEX)), std::string("Error in create device matrix"));
}

template<class T>
void SparseMatrix<T>::setPackedInfo(LARGEINDEX* Info)
{
	Info[0] =(LARGEINDEX) dev_matrixData;
	Info[1] = (LARGEINDEX)dev_rowInd;
	Info[2] = (LARGEINDEX)dev_colInd;
	Info[3] = matrixSize;
	Info[4] = rowIndexLen;
	Info[5] = colIndexLen;
	Info[6] = matrixRowDim;
	Info[7] = matrixColDim;
	Info[8] = offset;
}
using namespace std;
template<class T>
void SparseMatrix<T>::print()
{
	cout << "Matrix Data: ";
	for (LARGEINDEX i = 0; i < matrixSize; i++) {
		cout << matrixData[i] << " ";
	}
	cout << endl;
	cout << "Row index: ";
	for (LARGEINDEX i = 0; i < rowIndexLen; i++) {
		cout << rowInd[i] << " ";
	}
	cout << endl;
	cout << "Column index: ";
	for (LARGEINDEX i = 0; i < colIndexLen; i++) {
		cout << colInd[i] << " ";
	}
	cout << endl;
	cout << "Matrix size: " << matrixSize <<
		", Row size: " << rowIndexLen <<
		", Column size: " << colIndexLen <<
		", Offset: " << offset << endl;
	cout <<"Row dimension: " << matrixRowDim <<
		", Column dimension: " << matrixColDim << endl;
}



