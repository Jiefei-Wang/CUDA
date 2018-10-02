# #Load the test functions
# source("tests/Support_functions.R")
# 
# #The file location, it should be different in Linux.

# #k is the dimension of the matrix
# #test_upload:Test the communication between GPU and CPU
# #test_matrixSum: Test the matrix row and column sum and compare the operation times
# test_upload(k=100)
# test_matrixSum(k=1000,rowSum=T)
# test_matrixSum(k=1000,rowSum=F)


test_that("Matrix upload and download",{
  lib_file="src/kernel.dll"
  dyn.load(lib_file)
  k=100
  test.data=sparseData(row=k,col=k,nonzero=k*k/2)
  mycuda=CUDAispase(test.data$dataframe,test.data$rowNum,test.data$colNum,test.data$rowind,test.data$colind)
  mycuda=upload(mycuda)
  mycuda@data=as.double(rep(0,length(test.data$dataframe)))
  mycuda@rowInd=as.double(rep(0,length(test.data$dataframe)))
  mycuda@colInd=as.double(rep(0,length(test.data$colind)))
  mycuda=download(mycuda)
  expect_equal(mycuda@data,test.data$dataframe)
  expect_equal(mycuda@rowInd,test.data$rowind)
  expect_equal(mycuda@colInd,test.data$colind)
  dyn.unload(lib_file)
})


test_that("Matrix sum",{
  lib_file="src/kernel.dll"
  dyn.load(lib_file)
  k=100
  test.data=sparseData(row=k,col=k,nonzero=k*k/2)
  mycuda=CUDAispase(test.data$dataframe,test.data$rowNum,test.data$colNum,test.data$rowind,test.data$colind)
  mycuda=upload(mycuda)
  col_result=colSums(mycuda)
  row_result=rowSums(mycuda)
  expect_equal(col_result,base::colSums(test.data$dataMatrix))
  expect_equal(row_result,base::rowSums(test.data$dataMatrix))
  dyn.unload(lib_file)
})

data=c(3,7,5,6,8,7,5,3,6,5,9,8,3,4,7,6,7,4,8,6,6,6,6,5,4,3,9,7,6,9,4,6,4,4,5,5,3,5,3,6,8,6,4,7,5,8,6,5,7,6 )
row=c(0,1,2,4,5,6,7,9,1,4,8,0,3,5,7,9,2,4,5,7,9,2,4,6,9,1,5,6,7,8,1,2,4,5,0,3,6,7,9,2,3,4,8,9,0,1,2,3,5,7)
col=c(0,8,11,16,21,25,30,34,39,44,50)
mycuda=CUDAispase(data,10,10,row,col)
mycuda=upload(mycuda)
col_result=colSums(mycuda)
row_result=rowSums(mycuda)


