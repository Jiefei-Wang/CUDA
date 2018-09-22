#include "Unit_test.cuh"
#include <iostream>
#include <ctime>
using namespace std;



void test_matrixUpload() {
#include "read_test_data"
	double offset[] = { 10 };
	double* address = new double[9];
	upload(data, rowInd, colInd, size, offset, address);
	double* D_data = new double[(LARGEINDEX)size[0]];
	double* D_rowInd = new double[(LARGEINDEX)size[1]];
	double* D_colInd = new double[(LARGEINDEX)size[2]];
	download(D_data, D_rowInd, D_colInd, address);
	double error = 0;
	error+=checkValue(data, D_data,size[0]);
	error += checkValue(D_rowInd, D_rowInd, size[1]);
	error += checkValue(D_colInd, D_colInd, size[2]);

	if (error == 0) {
		std::cout << "Matrix upload and download test report: Pass" << std::endl;
	}
	else {
		std::cout << "Matrix upload and download test report: Error is : " << error << std::endl;
	}
	double* result = new double[10];
	int a = 1;
	colSums(&a, address, result);
	//print_partial_matrix("C cuda kernel:", result, 1, 10);
	error = checkValue(result, rowsum, size[0]);
	if (error == 0) {
		std::cout << "Matrix rowsum test report: Pass" << std::endl;
	}
	else {
		std::cout << "Matrix rowsum test report: Error is : " << error << std::endl;
	}
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;


}
