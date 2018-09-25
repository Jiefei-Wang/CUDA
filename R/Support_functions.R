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
