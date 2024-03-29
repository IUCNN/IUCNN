% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/iucnn_climate_features.R
\name{iucnn_climate_features}
\alias{iucnn_climate_features}
\title{Extract Climatic Features from Occurrence Records}
\usage{
iucnn_climate_features(
  x,
  climate_input = NULL,
  species = "species",
  lon = "decimallongitude",
  lat = "decimallatitude",
  rescale = TRUE,
  res = 10,
  type = "selected",
  download_folder = tempdir()
)
}
\arguments{
\item{x}{a data.frame of species occurrence records including three columns with
species name, longitudinal coordinates and latitudinal coordinates (both decimal).}

\item{climate_input}{a SpatRaster object with climate data. Optional. If not
provided, the 19 bioclim variables from www.worldclim.org are used as
default.}

\item{species}{a character string. The name of the column with the species names.}

\item{lon}{a character string. The name of the column with the longitude.}

\item{lat}{a character string. The name of the column with the latitude.}

\item{rescale}{logical. If TRUE, the features are rescaled.
This is recommended to run IUCNN, and the default. If FALSE, raw (human readable)
feature values are returned.}

\item{res}{numeric. The resolution of the default climate rasters.
One of 2.5, 5, or 10. Only relevant if
climate_input is NULL}

\item{type}{character string. A selection of which variables to return.
If "all" all 19 bioclim variables
if "selected" only Annual Mean Temperature, Temperature Seasonality,
Mean temperature of the Coldest Quarter,
Annual Precipitation, Precipitation seasonality and
Precipitation of the Driest Quarter are returned}

\item{download_folder}{character string. The folder were to save the
data used for feature extraction. Relative to the working directory.
Set to NULL for the working directory}
}
\value{
a data.frame of climatic features
}
\description{
Extract median and range (95\% - 5\% quantiles) climate
features based on a table of species occurrence coordinates.
If no climate data is supplied via the input.climate
argument, 19 bioclim variables are
downloaded from  www.worldclim.org.
Rescaling is only done for these default variables.
}
\details{
All climate variables are summarized  to the species median.
}
\examples{
\dontrun{
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(200,10,15),
                  decimallatitude = runif(200,-5,5))

iucnn_climate_features(dat)
}

}
\seealso{
Other Feature preparation: 
\code{\link{iucnn_bias_features}()},
\code{\link{iucnn_biome_features}()},
\code{\link{iucnn_cnn_features}()},
\code{\link{iucnn_footprint_features}()},
\code{\link{iucnn_geography_features}()},
\code{\link{iucnn_phylogenetic_features}()},
\code{\link{iucnn_prepare_features}()},
\code{\link{iucnn_prepare_phy}()}
}
\concept{Feature preparation}
\keyword{Feature}
\keyword{preparation}
