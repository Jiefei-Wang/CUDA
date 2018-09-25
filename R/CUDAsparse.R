
.CUDAispase=setClass("CUDAispase",
                     slot=c(data="vector",rowNum="numeric",colNum="numeric",
                            rowInd="vector",colInd="vector",
                            offset="numeric",GPUaddress="vector",status="character"))
CUDAispase<-function(data,rowNum,colNum,rowInd,colInd){
  obj=.CUDAispase(data=as.double(data),rowNum=as.double(rowNum),colNum=as.double(colNum),rowInd=as.double(rowInd),colInd=as.double(colInd))
  obj@offset=as.double(0)
  obj@status="ready"
  obj@GPUaddress=vector()
  return(obj)
}


setGeneric(name="upload",def=function(obj){
  standardGeneric("upload")
})
setMethod(f="upload",signature = "CUDAispase",
          definition=function(obj){
            obj@GPUaddress=as.double(rep(0,9))
            result=.C("upload",
                      obj@data,obj@rowInd,obj@colInd,
                      as.double(c(length(obj@data),length(obj@rowInd),length(obj@colInd),obj@rowNum,obj@colNum)),
                      obj@offset,obj@GPUaddress)
            obj@GPUaddress=result[[length(result)]]
            return(obj)
          })

setGeneric(name="download",def=function(obj){
  standardGeneric("download")
})
setMethod(f="download",signature = "CUDAispase",
          definition=function(obj){
            if(length(obj@GPUaddress)==0){
              stop("The GPU data does not exist")
            }
            result=.C("download",
                      obj@data,obj@rowInd,obj@colInd,obj@GPUaddress)
            obj@data=result[[1]]
            obj@rowInd=result[[2]]
            obj@colInd=result[[3]]
            return(obj)
          })









