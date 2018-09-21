#Load the test functions
source("tests\\Support_functions.R")

#The file location, it should be different in Linux.
dll_lib="src\\kernel.dll"

#k is the dimension of the matrix
#test_upload:Test the communication between GPU and CPU
#test_matrixSum: Test the matrix row and column sum and compare the operation times
test_upload(k=100)
test_matrixSum(k=1000,rowSum=T)
test_matrixSum(k=1000,rowSum=F)







