#include "Tools.cuh"

void errorHandle(cudaError_t error, std::string msg) {
	static int count = 0;
	if (error != cudaSuccess) {
		count++;
		std::cout << count << "." << cudaGetErrorString(error);
		if (msg.length() != 0) {
			std::cout << ":"<<msg.c_str()<<std::endl;
		}
	}
}


__host__ __device__
void errorPrint(char * msg)
{
#ifndef  __CUDA_ARCH__
	std::cout << msg << std::endl;
#endif
}


LARGEINDEX *  getIndexFromR(double* source, LARGEINDEX length) {
	LARGEINDEX * target = new LARGEINDEX[length];
	for (LARGEINDEX i = 0; i < length; i++) {
		target[i] = source[i];
	}
	return(target);
}

template<class T>
void fillWithNum(T* target, T number, int n) {
	for (int i = 0; i < n; i++) {
		target[i] = number;
	}
}

template<class T>
void print_partial_matrix(char* title, T *M, int nrows, int ncols, int max_row,
	int max_col)
{
	std::cout << title << std::endl;
	int row, col;
	if (max_row == -1) max_row = nrows;
	if (max_col == -1) max_col = ncols;
	for (row = 0; row < max_row; row++)
	{
		for (col = 0; col < max_col; col++)
		{
			std::cout << M[row + col*nrows] << " ";
		}
		printf("...\n");
	}
	printf("...\n");
}


template void transformData(double* target, LARGEINDEX* source, LARGEINDEX length);
template void transformData(LARGEINDEX* target, double* source, LARGEINDEX length);
template void transformData(LARGEINDEX* target, LARGEINDEX* source, LARGEINDEX length);

template void fillWithNum(double* target, double number, int n);
template void fillWithNum(float* target, float number, int n);
template void fillWithNum(int* target, int number, int n);

template void print_partial_matrix(char* title, double *M, int nrows, int ncols, int max_row,
	int max_col) ;
template void print_partial_matrix(char* title, float *M, int nrows, int ncols, int max_row,
	int max_col);
template void print_partial_matrix(char* title, int *M, int nrows, int ncols, int max_row,
	int max_col);


