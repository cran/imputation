\name{cv.SVTImpute}
\alias{cv.SVTImpute}
\title{
Cross Validation for SVTImpute
}
\description{
Use cross-validation to determine optimal parameters for kNN, SVD, and SVD imputation
}
\usage{
cv.SVTImpute(x, lambda.range = seq(0, 1, length.out = 101))
}
\arguments{
  \item{x}{
Data matrix containing missing values.
}
  \item{lambda.range}{
A vector representing the range of lambdas that should be tested
}
}
\details{
Erases data at random and attempts SVT imputation to restore the matrix.
The mean absolute error is used to determine what value of lambda gives the
best results.
}
\value{
The optimal value of lambda
}
\references{

}
\author{
Jeffrey Wong
}
\note{

}

\seealso{
cv.kNNImpute, cv.SVDImpute
}
\examples{
x = matrix(rnorm(100),10,10)
x[1,1] = NA
x[5,8] = NA
x[2,3] = NA
x[7,7] = NA
lambda = cv.SVTImpute(x)
SVTImpute(x, lambda)
}
