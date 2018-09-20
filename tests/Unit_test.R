path=rstudioapi::getActiveDocumentContext()$path
setwd(dirname(path))
source("test2_generate_data.R")
library("tictoc")

test_upload<-function(k=10){

test.data=sparseData(row=k,col=k,nonzero=k*k/2)


dyn.load("cuda matrix\\x64\\Debug\\Matrix_class.dll")
offset=10
address=as.double(rep(0,9))
result=.C("upload",
         test.data$dataframe,test.data$rowind,test.data$colind,test.data$size,
         offset,address)

address=result[[length(result)]]

downloaded.data=as.double(rep(0,test.data$size[1]))
downloaded.rowind=as.double(rep(0,test.data$size[2]))
downloaded.colind=as.double(rep(0,test.data$size[3]))
result1=.C("download",
           downloaded.data,downloaded.rowind,downloaded.colind,address)
downloaded.data=result1[[1]]
downloaded.rowind=result1[[2]]
downloaded.colind=result1[[3]]

if(sum(test.data$dataframe-downloaded.data)+
  sum(test.data$index-c(downloaded.rowind,downloaded.colind))==0){
  print("Upload check passed")
}else{
  print("Upload check failure")
}

dyn.unload("cuda matrix\\x64\\Debug\\Matrix_class.dll")
}

test_upload(k=100)

#generate_test_data(test.data)

dyn.load("cuda matrix\\x64\\RCODE\\Matrix_class.dll")
k=10000
test.data=sparseData(row=k,col=k,nonzero=k*k/2)
m=test.data$sparseMatrix
offset=10
address=as.double(rep(0,9))
result=.C("upload",
          test.data$dataframe,test.data$rowind,test.data$colind,test.data$size,
          offset,address)

address=result[[length(result)]]
sumResult=as.double(rep(0,k))
simTime=100
#Row sum
tic()
for(i in 1:simTime){
CudaSum=.C("colSums",
          as.integer(1),address,sumResult)
}
toc()
tic()
for(i in 1:simTime){
denseSum=rowSums(test.data$dataMatrix)
}
toc()
tic()
for(i in 1:simTime){
sparseSum=rowSums(m)
}
toc()
sum(CudaSum[[3]]-denseSum)

#column sum
tic()
for(i in 1:simTime){
  CudaSum=.C("colSums",
             as.integer(2),address,sumResult)
}
toc()
tic()
for(i in 1:simTime){
  denseSum=colSums(test.data$dataMatrix)
}
toc()
tic()
for(i in 1:simTime){
  sparseSum=colSums(m)
}
toc()
sum(CudaSum[[3]]-denseSum)




dyn.unload("cuda matrix\\x64\\RCODE\\Matrix_class.dll")


