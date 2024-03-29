% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/iucnn_footprint_features.R
\name{iucnn_footprint_features}
\alias{iucnn_footprint_features}
\title{Extract Human Footprint Index Features from Occurrence Records}
\source{
https://wcshumanfootprint.org/
}
\usage{
iucnn_footprint_features(
  x,
  footp_input = NULL,
  species = "species",
  lon = "decimallongitude",
  lat = "decimallatitude",
  rescale = TRUE,
  year = c(1993, 2009),
  download_folder = tempdir(),
  breaks = c(0, 0.81, 1.6, 2.3, 100),
  verbose = FALSE
)
}
\arguments{
\item{x}{a data.frame of species occurrence records including three columns with
species name, longitudinal coordinates and latitudinal coordinates (both decimal).}

\item{footp_input}{an object of the class SpatRaster with values for the
human footprint index. If the SpatRaster object has two or more layers,
different layers are interpreted as different time-slices.}

\item{species}{a character string. The name of the column with the species names.}

\item{lon}{a character string. The name of the column with the longitude.}

\item{lat}{a character string. The name of the column with the latitude.}

\item{rescale}{logical. If TRUE, the values are rescaled using
natural logarithm transformation. If FALSE,
remember to change the breaks argument.}

\item{year}{numeric. The years for which to obtain the human footprint index.
The default is to the two layers available. Can be a either year, in case only
one slice is desired. Other time slices are currently not supported,}

\item{download_folder}{character string. The folder were to save the
data used for feature extraction. Relative to the working directory.
Set to NULL for the working directory}

\item{breaks}{numerical. The breaks to bin the human footprint index
for the final features. The defaults are
empirical values for the global footprint and rescale=TRUE.
For custom values ensure that they
cover the whole value range and are adapted to the value of rescale.}

\item{verbose}{logical. If TRUE, progress messages will be displayed. Default
= FALSE.}
}
\value{
a data.frame of human footprint features
}
\description{
Bins the human footprint index into a set of bins and the
fraction of occurrence records
of a species in each bin are the features.
By default the human footprint index is downloaded from
https://wcshumanfootprint.org/. THIS FUNCTION WILL DOWNLOAD DATA FROM
THE INTERNET AND SAVE IT TO THE  WORKING DIRECTORY. The data files
are >200 MB each and downloading may
take some time on first execution.
}
\details{
By default four categories of increasing human footprint index
( 1 = lowest, 4 = highest) are selected and rescaled.
}
\examples{
\dontrun{
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(200,10,15),
                  decimallatitude = runif(200,-5,5))

iucnn_footprint_features(dat)
}

}
\seealso{
Other Feature preparation: 
\code{\link{iucnn_bias_features}()},
\code{\link{iucnn_biome_features}()},
\code{\link{iucnn_climate_features}()},
\code{\link{iucnn_cnn_features}()},
\code{\link{iucnn_geography_features}()},
\code{\link{iucnn_phylogenetic_features}()},
\code{\link{iucnn_prepare_features}()},
\code{\link{iucnn_prepare_phy}()}
}
\concept{Feature preparation}
\keyword{Feature}
\keyword{preparation}
