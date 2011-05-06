\name{cv.SVDImpute}
\alias{cv.SVDImpute}
\title{
Cross Validation for SVDImpute
}
\description{
Use cross-validation to determine optimal parameters for kNN, SVD, and SVD imputation
}
\usage{
cv.SVDImpute(x, k.max = 10)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
Data matrix containing missing values.
}
  \item{k.max}{
The maximum rank-k approximation that should be used
}
}
\details{
Erases data at random and attempts SVD imputation to restore the matrix.
The mean absolute error is used to determine what value of k gives the
best results.
}
\value{
The optimal value of k
}
\references{

}
\author{
Jeffrey Wong
}
\note{

}

\seealso{
cv.kNNImpute, cv.SVTImpute
}
\examples{
x = matrix(rnorm(100),10,10)
x[1,1] = NA
x[5,8] = NA
x[2,3] = NA
x[7,7] = NA
k = cv.SVDImpute(x)
SVDImpute(x, k)
}
