\name{imputation-package}
\alias{imputation-package}
\alias{imputation}
\docType{package}
\title{
imputation
}
\description{
Provides methods for missing data imputation.  Currently, iterative
SVD imputation and kNN are supported.  SVD imputation initially imputes
missing values using the mean, then successively calculates a low rank-k
approximation for the data.  kNN finds the k nearest neighbors and fills
missing data by taking the mean of the missing variable among the k neighbors.
}
\details{
\tabular{ll}{
Package: \tab imputation\cr
Type: \tab Package\cr
Version: \tab 1.1\cr
Date: \tab 2011-04-30\cr
License: \tab GPL\cr
LazyLoad: \tab yes\cr
}
}
\author{
Jeffrey Wong
<jeffwong@stanford.edu>
}
\references{
}
\keyword{ package }
\seealso{
}
\examples{
}
