#' Extract Climatic Features from Occurrence Records
#'
#' Extract median and range (95% - 5% quantiles) climate
#' features based on a table of species occurrence coordinates.
#' If no climate data is supplied via the input.climate
#' argument, 19 bioclim variables are
#' downloaded from  www.worldclim.org.
#' Rescaling is only done for these default variables.
#'
#'
#' All climate variables are summarized  to the species median.
#'@inheritParams prep_features
#'@param climate.input a raster or rasterStack with climate data.
#'Optional. If not provided,
#'the 19 bioclim variables from www.worldclim.org are used as default.
#'@param res numeric. The resolution of the default climate rasters.
#'One of 2.5, 5, or 10. Only relevant if
#'climate.input is NULL
#'@param type character string. A selection of which variables to return.
#'If "all" all 19 bioclim variables
#'if "selected" only Annual Mean Temperature, Temperature Seasonality,
#'Mean temperature of the Coldest Quarter,
#'Annual Precipitation, Precipitation seasonality and
#' Precipitation of the Driest Quarter are returned
#'
#'@return a data.frame of climatic features
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#'\dontrun{
#' dat <- data.frame(species = c("A","B"),
#'                   decimallongitude = runif (200,10,15),
#'                   decimallatitude = runif (200,-5,5))
#'
#'ft_clim(dat)
#'}
#'
#'
#' @export
#' @importFrom dplyr as_tibble bind_cols .data distinct group_by left_join mutate select summarize_all
#' @importFrom raster extract getData
#' @importFrom magrittr %>%
#' @importFrom stats median
#' @importFrom checkmate assert_character assert_data_frame assert_logical assert_numeric

ft_clim <- function(x,
                    climate.input = NULL,
                    species = "species",
                    lon = "decimallongitude",
                    lat = "decimallatitude",
                    rescale = TRUE,
                    res = 10,
                    type = "selected",
                    download.folder = "feature_extraction"){

  # assertions
  assert_data_frame(x)
  assert_character(x[[species]], any.missing = FALSE, min.chars = 1)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)
  assert_logical(rescale)
  assert_numeric(res)
  assert_character(download.folder, null.ok = TRUE)

  # check arguments for the type
  match.arg(arg = type, choices = c("selected", "all"))

  # Get climate data if non is provided
  if(is.null(climate.input)){
    # set download path
    if(!dir.exists(download.folder)){
      dir.create(download.folder)
    }
    if(is.null(download.folder)){
      download.folder <- getwd()
    }else{
      download.folder <- file.path(getwd(), download.folder)
    }

    climate.input <- raster::getData('worldclim',
                                     var = 'bio',
                                     res = res,
                                     path = download.folder)
  }else{
    rescale <- FALSE
  }

  # Extract values
  bio_ex <- raster::extract(x = climate.input,
                            y = x[,c(lon, lat)])

  # absolute climate variables
  bio <- x %>%
    dplyr::select(.data$species) %>%
    dplyr::bind_cols(bio_ex %>%  as_tibble()) %>%
    group_by(.data$species) %>%
    dplyr::summarize_all(median, na.rm = TRUE)

  # climate range variables
  range_inp <- x %>%
    dplyr::select(.data$species) %>%
    dplyr::bind_cols(bio_ex %>%  as_tibble()) %>%
    group_by(.data$species)

  min <- range_inp %>%
    dplyr::summarize_all(quantile, probs = 0.05, na.rm = TRUE)
  max <- range_inp %>%
    dplyr::summarize_all(quantile, probs = 0.95, na.rm = TRUE)

  range <- data.frame(species = min$species,
                      max[, -1] - min[, -1])
  names(range)[-1] <- paste("range_", names(range)[-1], sep = "")

  # Rescale, I feel this needs some more though/justification, but Daniele is happy with it for now
  if(rescale){
    bio[, 2] <- bio[, 2]/150
    bio[, 3] <- bio[, 3]/10
    bio[, 4] <- bio[, 4]/50
    bio[, 5] <- log10(1+bio[, 5])
    bio[, 6:12] <- bio[, 6:12] / 15
    bio[, 13:20] <- log10(1+bio[, 13:20])

    range[, 2] <- range[, 2]/15
    range[, 3] <- range[, 3]/10
    range[, 4] <- range[, 4]/50
    range[, 5] <- log10(1+range[, 5])
    range[, 6:12] <- range[, 6:12] / 15
    range[, 13:20] <- log10(1+range[, 13:20])
  }else{
    warning("Features not rescaled. Rescale manually")
  }

  if(type == "selected"){
    out <- bio %>%
      select(.data$species,
             .data$bio1,
             .data$bio4,
             .data$bio11,
             .data$bio12,
             .data$bio15,
             .data$bio17) %>%
      bind_cols(range %>% select(.data$range_bio1,
                                 .data$range_bio4,
                                 .data$range_bio11,
                                 .data$range_bio12,
                                 .data$range_bio15,
                                 .data$range_bio17))
  }else{
    out <- left_join(bio, range, by = "species")
  }
  # return
  return(out)
}
