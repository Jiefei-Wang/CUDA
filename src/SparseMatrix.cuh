#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "Tools.cuh"
#include "CommonHeader.cuh"






enum Deviceloc { Host = 1, Device = 2, Both = 3, Auto = 4 };

template <class T>
class SparseMatrix {
public:
	LARGEINDEX matrixSize;
	LARGEINDEX rowIndexLen;
	LARGEINDEX colIndexLen;
	LARGEINDEX matrixRowDim;
	LARGEINDEX matrixColDim;
	LARGEINDEX offset;
	T * matrixData = nullptr;
	LARGEINDEX * rowInd = nullptr;
	LARGEINDEX * colInd = nullptr;

	T * dev_matrixData = nullptr;
	LARGEINDEX * dev_rowInd = nullptr;
	LARGEINDEX * dev_colInd = nullptr;
public:
	__alldev__
	SparseMatrix(LARGEINDEX* size, LARGEINDEX * offset, Deviceloc location = Deviceloc::Auto);
	__alldev__
	SparseMatrix(T*, LARGEINDEX* index, LARGEINDEX* rowInd, LARGEINDEX* colInd, LARGEINDEX * offset, Deviceloc location= Deviceloc::Auto);

	SparseMatrix(LARGEINDEX* Info);

	void HostToDevice();
	void deviceToHost();
	//Delete the host matrix
	void delHostMatrix();
	//Delete the device matrix
	void delDevMatrix();
	//Create the host matrix, if the matrix does not exist, it will create a new one
	void createHostMatrix(LARGEINDEX matrixSize, LARGEINDEX rowIndexLen, LARGEINDEX colIndexLen);
	//Create the device matrix, if the matrix does not exist, it will create a new one
	void createDevMatrix(LARGEINDEX matrixSize, LARGEINDEX rowIndexLen, LARGEINDEX colIndexLen);
	void setPackedInfo(LARGEINDEX* Info);
	void print();
};


