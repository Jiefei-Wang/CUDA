#include "Test_tools.cuh"
template<class T>
double checkValue(T* a, T*b, int num) {
	double error = 0;
	double currentError = 0;
	for (int i = 0; i < 3; i++) {
		currentError = a[i] - b[i];
		error += currentError;
	}
	return error;
}