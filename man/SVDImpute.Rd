\name{SVDImpute}
\alias{SVDImpute}
\title{
SVD Imputation
}
\description{
Multiple imputation using the SVD
}
\usage{
SVDImpute(x, k, num.iters = 10, gpu)
}
\arguments{
  \item{x}{
Numeric matrix containing some missing data.  Each row is an observation
and the columns are features of the data
}
  \item{k}{
The rank k approximation that should be used when calculating the SVD
}
  \item{num.iters}{
Maximum number of iterations to use
}
  \item{gpu}{
Logical.  If true, use CUDA accelerated functions to calculate the SVD.
Only useful for very large inputs
}
}
\details{
Identifies elements of x that are NA and initializes them to the mean
of that feature.  Computes the rank k SVD of the matrix, then replaces
the original missing values with those determined by the SVD.  Calculates
the SVD again on the imputed matrix and replaces missing values until 
convergence.
}
\value{
The imputed data matrix
}
\references{
}
\author{
Jeffrey Wong
}
\note{
}

\seealso{
kNNImpute
}
\examples{
x = matrix(rnorm(30),10,3)
x[2,3] = NA
x[5,1] = NA
x[2,2] = NA
x.imputed = SVDImpute(x,3)
}
\keyword{ svd }
\keyword{ impute }
