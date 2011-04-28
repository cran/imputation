#use rank k svd approximation of x
#if x is large, set gpu=T to use CUDA accelerated svd function
.rankKapprox = function(x, k, gpu) {
  if(gpu) {
    x.svd = gpuSvd(x, nu=nrow(x),nv=ncol(x))
  }
  else {
    x.svd = svd(x, nu=k, nv=k)
  }
  x.svd$u[,1:k] %*% diag(x.svd$d[1:k]) %*% t(x.svd$v[,1:k])
}

SVDImpute = function(x, k, num.iters = 10, gpu=F) {
  if(gpu) {
    stop("no gpu support yet")
  }
  missing.matrix = is.na(x)
  if(sum(missing.matrix) == 0) {
    return (x)
  }
  x = apply(rbind(1:ncol(x),x), 2, function(j) {
    colindex = j[1]
    x.original = j[-1]
    missing.indices = which(missing.matrix[,colindex])
    x.original[missing.indices] = mean(x.original[-missing.indices])
    x.original
  })
  for(i in 1:num.iters) {
    x.svd = .rankKapprox(x, k, gpu)
    x[missing.matrix] = x.svd[missing.matrix]
  }
  return (x)
}

kNNImpute = function(x, k) {
  missing.matrix = is.na(x)
  if(sum(missing.matrix) == 0) {
    return (x)
  }
  missing.rows = apply(missing.matrix, 1, function(i) {
    any(i)
  })
  missing.rows.indices = which(missing.rows)

  x.dist = as.matrix(dist(x, upper=T))

  x.imputedrows = t(sapply(missing.rows.indices, function(i) {
    missing.cols = missing.matrix[i,]
    
    neighbors.ranks.indices = order(x.dist[i,])
    k.neighbors = neighbors.ranks.indices[2:(k+1)]

    #get neighbors' values for missing variable
    neighbors.missing = x[k.neighbors, missing.cols] 
    if(k > 1 ) {
      if(sum(missing.cols) > 1)  #multiple missing values
        x[i, missing.cols] = apply(neighbors.missing, 2, mean)
      else  #only 1 missing value
        x[i, missing.cols] = mean(neighbors.missing)
    }
    else {  #k = 1, just take missing values from nearest neighbor
      x[i, missing.cols] = neighbors.missing
    }
    x[i,]
  }))
  x[missing.rows.indices,] = x.imputedrows
  return (x)
}

