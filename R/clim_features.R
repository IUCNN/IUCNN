#' Extract Climatic Features from Occurrence Records
#'
#' Extract median climate features based on a table of  species occurrence coordinates.
#' If no climate data is supplied via the input.climate argument, 19 bioclim variables are
#' downloaded from  www.worldclim.org. Rescaling is only done for these default variables.
#'
#'
#' All climate variables are summarized  to the species median.
#'
#'@param climate.input a raster or rasterStack with climate data. Optional. If not provided,
#'the 19 bioclim variables from www.worldclim.org are used as default.
#'@param res numeric. The resolution of the default climate rasters. ONe of 2.5, 5, or 10. Only relevant if
#'climate.input is NULL
#'@inheritParams geo_features
#'
#'
#'@return a data.frame of climatic features
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#'\dontrun{
#'dat <- data.frame(species = "A",
#'                 decimallongitude = runif (200,-5,5),
#'                 decimallatitude = runif (200,-5,5))
#'
#'clim_features(dat)
#'}
#'
#'
#' @export
#' @importFrom dplyr as_tibble bind_cols .data distinct group_by left_join mutate select summarize_all
#' @importFrom raster extract getData
#' @importFrom magrittr %>%
#' @importFrom stats median
clim_features <- function(x,
                          climate.input = NULL,
                          species = "species",
                          lon = "decimallongitude",
                          lat = "decimallatitude",
                          rescale = TRUE,
                          res = 10){

  # Get climate data if non is provided
  if(is.null(climate.input)){
    climate.input <- raster::getData('worldclim', var='bio', res = res)
  }else{
    rescale <- FALSE
  }

  # Extract values
  bio <- raster::extract(x = climate.input, y = x[,c(lon, lat)])

  bio <- x %>%
    dplyr::select(.data$species) %>%
    dplyr::bind_cols(bio %>%  as_tibble()) %>%
    group_by(.data$species) %>%
    dplyr::summarize_all(median, na.rm = TRUE)

  # Rescale, I feel this needs some more though/justification, but Daniele is happy with it for now
  if(rescale){
    bio[, 2] <- bio[, 2]/15
    bio[, 3] <- bio[, 3]/10
    bio[, 4] <- bio[, 4]/50
    bio[, 5] <- log10(1+bio[, 5])
    bio[, 6:12] <- bio[, 6:12] / 15
    bio[, 13:20] <- log10(1+bio[, 13:20])
  }else{
    warning("Features not rescaled. Rescale manually")
  }

  # return
  return(bio)
}
