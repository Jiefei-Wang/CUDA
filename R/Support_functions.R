sparseData<-function(row=10,col=10,nonzero=5){
  library("Matrix")
  mydata=matrix(0,row,col)
  mydata[sample(1:length(mydata),nonzero)]=rbinom(nonzero,10,0.5)+1
  m <- Matrix(mydata,sparse=T)
  
  dataframe=m@x
  rowind=m@i
  colind=m@p
  list(dataframe=dataframe,rowind=rowind,colind=colind,
       rowNum=row,colNum=col,dataMatrix=mydata,
       sparseMatrix=m
  )
}


# 
# generate_test_data<-function(test.data){
#   tmp=test.data$dataframe
#   message(paste("double data[]={",paste(tmp,sep="",collapse = ","),"};"))
#   tmp=test.data$rowind
#   message(paste("double rowInd[]={",paste(tmp,sep="",collapse = ","),"};"))
#   tmp=test.data$colind
#   message(paste("double colInd[]={",paste(tmp,sep="",collapse = ","),"};"))
#   tmp=test.data$size
#   message(paste("double size[]={",paste(tmp,sep="",collapse = ","),"};"))
# }
# 
# test_matrixSum<-function(k=10,rowSum=T){
#   dyn.load(lib_file)
#   test.data=sparseData(row=k,col=k,nonzero=k*k/2)
#   m=test.data$sparseMatrix
#   offset=10
#   address=as.double(rep(0,9))
#   result=.C("upload",
#             test.data$dataframe,test.data$rowind,test.data$colind,test.data$size,
#             offset,address)
#   
#   address=result[[length(result)]]
#   sumResult=as.double(rep(0,k))
#   simTime=100
#   #Row sum
#   if(rowSum){
#     cuda.time=system.time({
#       for(i in 1:simTime){
#         CudaSum=.C("colSums",
#                    as.integer(1),address,sumResult)
#       }
#     })
#     message("CUDA time:")
#     print(cuda.time)
#     dense.time=system.time({
#       for(i in 1:simTime){
#         denseSum=rowSums(test.data$dataMatrix)
#       }})
#     message("Dense CPU sum time:")
#     print(dense.time)
#     sparse.time=system.time({
#       for(i in 1:simTime){
#         sparseSum=rowSums(m)
#       }})
#     message("Sparse CPU sum time:")
#     print(sparse.time)
#     CudaSum=.C("colSums",
#                as.integer(1),address,sumResult)
#     denseSum=rowSums(test.data$dataMatrix)
#     error=sum(CudaSum[[3]]-denseSum)
#     if(abs(error)<0.001){
#       message("Row sum check passed")
#     }
#   }else{
#     #column sum
#     cuda.time=system.time({
#       for(i in 1:simTime){
#         CudaSum=.C("colSums",
#                    as.integer(2),address,sumResult)
#       }})
#     message("CUDA time:")
#     print(cuda.time)
#     dense.time=system.time({
#       for(i in 1:simTime){
#         denseSum=colSums(test.data$dataMatrix)
#       }})
#     message("Dense CPU sum time:")
#     print(dense.time)
#     sparse.time=system.time({
#       for(i in 1:simTime){
#         sparseSum=colSums(m)
#       }})
#     message("Sparse CPU sum time:")
#     print(sparse.time)
#     CudaSum=.C("colSums",
#                as.integer(2),address,sumResult)
#     denseSum=colSums(test.data$dataMatrix)
#     error=sum(CudaSum[[3]]-denseSum)
#     if(abs(error)<0.001){
#       message("Column sum check passed")
#     }
#   }
#   dyn.unload(lib_file)
# }