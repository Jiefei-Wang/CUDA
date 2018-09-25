# #Load the test functions
# source("tests/Support_functions.R")
# 
# #The file location, it should be different in Linux.
# dll_lib="src/kernel.dll"
# SO_lib="src/CUDA.so"
# lib_file=dll_lib
# #k is the dimension of the matrix
# #test_upload:Test the communication between GPU and CPU
# #test_matrixSum: Test the matrix row and column sum and compare the operation times
# test_upload(k=100)
# test_matrixSum(k=1000,rowSum=T)
# test_matrixSum(k=1000,rowSum=F)



test_that("Matrix upload and download",{
  dyn.load(lib_file)
  k=100
  test.data=sparseData(row=k,col=k,nonzero=k*k/2)
  mycuda=CUDAispase(test.data$dataframe,test.data$rowNum,test.data$colNum,test.data$rowind,test.data$colind)
  mycuda=upload(mycuda)
  mycuda@data=as.double(rep(0,length(test.data$dataframe)))
  mycuda@rowInd=as.double(rep(0,length(test.data$dataframe)))
  mycuda@colInd=as.double(rep(0,length(test.data$colind)))
  mycuda=download(mycuda)
  all.equal(mycuda@data,test.data$dataframe)
  all.equal(mycuda@rowInd,test.data$rowind)
  all.equal(mycuda@colInd,test.data$colind)
  dyn.unload(lib_file)
})



