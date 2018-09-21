library("Matrix")
dll_lib="src\\kernel.dll"

sparseData<-function(row=10,col=10,nonzero=5){
mydata=matrix(0,row,col)
mydata[sample(1:length(mydata),nonzero)]=rbinom(nonzero,10,0.5)+1
m <- Matrix(mydata,sparse=T)

dataframe=m@x
rowind=m@i
colind=m@p
list(dataframe=as.double(dataframe),rowind=as.double(rowind),colind=as.double(colind),
     size=as.double(c(length(dataframe),length(rowind),length(colind),row,col)),dataMatrix=mydata,
     sparseMatrix=m
     )
}


generate_test_data<-function(test.data){
  tmp=test.data$dataframe
  message(paste("double data[]={",paste(tmp,sep="",collapse = ","),"};"))
  tmp=test.data$rowind
  message(paste("double rowInd[]={",paste(tmp,sep="",collapse = ","),"};"))
  tmp=test.data$colind
  message(paste("double colInd[]={",paste(tmp,sep="",collapse = ","),"};"))
  tmp=test.data$size
  message(paste("double size[]={",paste(tmp,sep="",collapse = ","),"};"))
}

#Upload a matrix of size k to GPU, and download it.
#Check if the CUDA function works.
test_upload<-function(k=10){
  test.data=sparseData(row=k,col=k,nonzero=k*k/2)
  dyn.load(dll_lib)
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
    message("Upload check passed")
  }else{
    message("Upload check failure")
  }
  
  dyn.unload(dll_lib)
}

test_matrixSum<-function(k=10,rowSum=T){
  dyn.load(dll_lib)
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
  if(rowSum){
    cuda.time=system.time({
      for(i in 1:simTime){
        CudaSum=.C("colSums",
                   as.integer(1),address,sumResult)
      }
    })
    message("CUDA time:")
    print(cuda.time)
    dense.time=system.time({
      for(i in 1:simTime){
        denseSum=rowSums(test.data$dataMatrix)
      }})
    message("Dense CPU sum time:")
    print(dense.time)
    sparse.time=system.time({
      for(i in 1:simTime){
        sparseSum=rowSums(m)
      }})
    message("Sparse CPU sum time:")
    print(sparse.time)
    CudaSum=.C("colSums",
               as.integer(1),address,sumResult)
    denseSum=rowSums(test.data$dataMatrix)
    error=sum(CudaSum[[3]]-denseSum)
    if(abs(error)<0.001){
      message("Row sum check passed")
    }
  }else{
    #column sum
    cuda.time=system.time({
      for(i in 1:simTime){
        CudaSum=.C("colSums",
                   as.integer(2),address,sumResult)
      }})
    message("CUDA time:")
    print(cuda.time)
    dense.time=system.time({
      for(i in 1:simTime){
        denseSum=colSums(test.data$dataMatrix)
      }})
    message("Dense CPU sum time:")
    print(dense.time)
    sparse.time=system.time({
      for(i in 1:simTime){
        sparseSum=colSums(m)
      }})
    message("Sparse CPU sum time:")
    print(sparse.time)
    CudaSum=.C("colSums",
               as.integer(2),address,sumResult)
    denseSum=colSums(test.data$dataMatrix)
    error=sum(CudaSum[[3]]-denseSum)
    if(abs(error)<0.001){
      message("Column sum check passed")
    }
  }
  dyn.unload(dll_lib)
}
