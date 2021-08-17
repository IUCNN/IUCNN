#' Prepare Features for a CNN model
#'
#' Converts a data.frame of species occurrences into input features to use
#' a convolutional neural network to approximate species extinction risk.
#'
#' If y is not provided, assumes a lat/lon grid with extent equal to the respective minimum and maximum in x
#'
#'@param x a data.frame with at least three columns containing taxon name, decimal longitude and latitude values.
#'@param y a raster as reference to count the number of occurrence records in.
#' Can be of any resolution and CRS but the coordinates in x need to be in the same CRS.
#'@param crs_x a proj4string specifying the Coordinate Reference system of the coordinates in x.
#'Default is to lat/lon WGS84.
#'@param res_y numeric. The resolution for the raster in decimal degrees.
#'Only relevant if y is not provided.
#'@inheritParams iucnn_prepare_features

#'@return a list of matrices, one for each input species, where the cells represent the number
#'of occurrence records in this cell as input for the \dQuote{cnn} class of \code{\link{iucnn_train_model}}.
#'
#'@keywords Feature preparation
#'@family Feature preparation
#'
#'@examples
#' dat <- data.frame(species = c("A","B"),
#'                   decimallongitude = runif (200,10,15),
#'                   decimallatitude = runif (200,-5,5))
#'
#'iucnn_cnn_features(dat)
#'
#'
#'@export
#'@importFrom checkmate assert_data_frame assert_character assert_numeric assert_number
#'@importFrom sf st_as_sf st_crs
#'@importFrom terra crs ext<- rast rasterize res<- vect

iucnn_cnn_features <- function(x,
                               y = NULL,
                               species = "species",
                               lon = "decimallongitude",
                               lat = "decimallatitude",
                               crs_x = "+proj=longlat +datum=WGS84",
                               res_y = 1){

  # check if x is a data.frame with the relevant columns
  assert_data_frame(x)
  assert_character(x[[species]])
  assert_numeric(x[[lon]])
  assert_numeric(x[[lat]])
  assert_number(res_y)

  # convert data.frame to sf object
  pts <- split(x, f = x[[species]])

  pts <- lapply(pts,
                "st_as_sf",
                coords = c(lon, lat),
                crs = crs_x)

  # if no raster is provided assume lat/lon and do a lat/lon raster
  if(is.null(y)){
    warning("assuming lat/long coordinates.")
    warning("rasterizing using unprojected lat/lon. provide projected coordinates and raster in the same projection if possible")
    if(res_y == 1){
      warning("using 1 degree resolution")
    }

    # create lat/lon default raster
    y <- rast(crs = "+proj=longlat +datum=WGS84")
    ext(y) <- c(min(x[,lon]), max(x[,lon]), min(x[,lat]), max(x[,lat]))
    res(y) <- res_y

  }else{
    # if y is a raster raster convert to terra
    if(class(y) == "RasterLayer"){
      y <- rast(y)
    }

    #check that projections between x and y are similar
    t <- st_crs(pts)
    if(terra::crs(y) != t$wkt){
      stop("x and y have different CRS.")
    }
  }

  # perform rasterization
  out <- lapply(pts, function(k){
    # convert to terra object
    sub <- vect(k)
    #rasterize/count the number of records per grid cell
    sub_out <- terra::rasterize(sub, y = y, fun = length)

    #convert into amtrix
    sub_out <- matrix(sub_out, ncol = ncol(y), byrow = TRUE)

    # replace NAs with zeros
    sub_out[is.na(sub_out)] <- 0

    return(sub_out)
  })

  class(out) <- c("cnn_features", class(out))

  # return a list of per species rasters
  return(out)
}

