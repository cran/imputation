#use rank k svd approximation of x
#if x is large, set gpu=T to use CUDA accelerated svd function
.rankKapprox = function(x, k, gpu) {
  if(gpu) {
    x.svd = gpuSvd(x, nu=nrow(x),nv=ncol(x))
  }
  else {
    x.svd = svd(x, nu=k, nv=k)
  }
  x.svd$u %*% diag(x.svd$d[1:k],nrow=k,ncol=k) %*% t(x.svd$v)
}

SVDImpute = function(x, k, num.iters = 10, gpu=F, verbose=T) {
  if(gpu) {
    stop("no gpu support yet")
  }
  missing.matrix = is.na(x)
  numMissing = sum(missing.matrix)
  print(paste("imputing on", numMissing, "missing values with matrix size",
    length(as.vector(x)), sep=" "))
  if(numMissing == 0) {
    return (x)
  }
  missing.cols.indices = which(apply(missing.matrix, 2, function(i) {
    any(i)
  }))
  x.missing = (rbind(1:ncol(x), x))[,missing.cols.indices]
  x.missing.imputed = apply(x.missing, 2, function(j) {
    colIndex = j[1]
    j.original = j[-1]
    missing.rows = which(missing.matrix[,colIndex])
    if(length(missing.rows) == nrow(x))
      warning( paste("Column",colIndex,"is completely missing",sep=" ") )
    j.original[missing.rows] = mean(j.original[-missing.rows])
    j.original
  })
  x[,missing.cols.indices] = x.missing.imputed
  missing.matrix = is.na(x)
  x[missing.matrix] = 0
  for(i in 1:num.iters) {
    if(verbose) print(paste("Running iteration", i, sep=" "))
    x.svd = .rankKapprox(x, k, gpu)
    x[missing.matrix] = x.svd[missing.matrix]
  }
  return (x)
}

kNNImpute = function(x, k, verbose=T) {
  if(k >= nrow(x))
    stop("k must be less than the number of rows in x")
  missing.matrix = is.na(x)
  numMissing = sum(missing.matrix)
  print(paste("imputing on", numMissing, "missing values with matrix size",
    length(as.vector(x)), sep=" "))
  if(numMissing == 0) {
    return (x)
  }
  
  missing.rows.indices = which(apply(missing.matrix, 1, function(i) {
    any(i)
  }))
  x.missing = (cbind(1:nrow(x),x))[missing.rows.indices,]
  x.missing.imputed = t(apply(x.missing, 1, function(i) {
    rowIndex = i[1]
    i.original = i[-1]
    if(verbose) print(paste("Imputing row", rowIndex,sep=" "))
    missing.cols = which(missing.matrix[rowIndex,])
    if(length(missing.cols) == ncol(x))
      warning( paste("Row",rowIndex,"is completely missing",sep=" ") )
    imputed.values = sapply(missing.cols, function(j) {
      #find neighbors that have data on the jth column
      neighbor.indices = which(!missing.matrix[,j])
      #compute distance to these neighbors
      neighbor.dist = as.matrix(dist(rbind(i.original, x[neighbor.indices,]),upper=T))
      #order the neighbors to find the closest ones
      knn.ranks = order(neighbor.dist[1,])
      #identify the row number in the original data matrix of the knn
      knn = neighbor.indices[(knn.ranks[2:(k+1)])-1]
      mean(x[knn,j])
    })
    i.original[missing.cols] = imputed.values
    i.original
  }))
  x[missing.rows.indices,] = x.missing.imputed

  missing.matrix = is.na(x)
  x[missing.matrix] = 0

  return (x)
}
