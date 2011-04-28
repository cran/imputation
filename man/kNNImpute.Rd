\name{kNNImpute}
\alias{kNNImpute}
\title{
k-NN Imputation
}
\description{
Imputation using k nearest neighbors
}
\usage{
kNNImpute(x, k)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
Numeric data matrix containing missing values
}
  \item{k}{
The number of neighbors to use for k-NN
}
}
\details{
Identifies observations, x, that have missing data.  Finds the k nearest 
observations to x and computes the mean of the feature that is missing.  
Replaces the missing value with this mean.
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
SVDImpute
}
\examples{
x = matrix(rnorm(30),10,3)
x[2,3] = NA
x[5,1] = NA
x[2,2] = NA
x.imputed = kNNImpute(x,3)
}
\keyword{ k-NN }
\keyword{ impute }
