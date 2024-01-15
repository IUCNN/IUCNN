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
#'@inheritParams iucnn_prepare_features
#'@param climate_input a SpatRaster object with climate data. Optional. If not
#'  provided, the 19 bioclim variables from www.worldclim.org are used as
#'  default.
#'@param res numeric. The resolution of the default climate rasters.
#'One of 2.5, 5, or 10. Only relevant if
#'climate_input is NULL
#'@param type character string. A selection of which variables to return.
#'If "all" all 19 bioclim variables
#'if "selected" only Annual Mean Temperature, Temperature Seasonality,
#'Mean temperature of the Coldest Quarter,
#'Annual Precipitation, Precipitation seasonality and
#' Precipitation of the Driest Quarter are returned
#'@param rescale logical. If TRUE, the features are rescaled.
#'This is recommended to run IUCNN, and the default. If FALSE, raw (human readable)
#'feature values are returned.
#'
#'@return a data.frame of climatic features
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#' \dontrun{
#' dat <- data.frame(species = c("A","B"),
#'                   decimallongitude = runif(200,10,15),
#'                   decimallatitude = runif(200,-5,5))
#'
#' iucnn_climate_features(dat)
#' }
#'
#' @export
#' @importFrom dplyr as_tibble bind_cols .data distinct group_by left_join mutate select summarize_all
#' @importFrom geodata worldclim_global
#' @importFrom magrittr %>%
#' @importFrom terra extract
#' @importFrom stats median
#' @importFrom checkmate assert_character assert_data_frame assert_logical assert_numeric

iucnn_climate_features <- function(x,
                                   climate_input = NULL,
                                   species = "species",
                                   lon = "decimallongitude",
                                   lat = "decimallatitude",
                                   rescale = TRUE,
                                   res = 10,
                                   type = "selected",
                                   download_folder = "feature_extraction"){

  # assertions
  assert_data_frame(x)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)
  assert_logical(rescale)
  assert_numeric(res)
  assert_character(download_folder, null.ok = TRUE)

  # check arguments for the type
  match.arg(arg = type, choices = c("selected", "all"))

  # Get climate data if non is provided
  if (is.null(climate_input)) {
    # set download path
    if (is.null(download_folder)) {
      download_folder <- getwd()
    }
    # else{
    #   download_folder <- file.path(getwd(), download_folder)
    # }
    if (!dir.exists(download_folder)) {
      dir.create(download_folder)
    }

    climate_input <- geodata::worldclim_global(var = 'bio',
                                               res = res,
                                               path = download_folder)
  }else{
    rescale <- FALSE
  }

  # Extract values
  bio_ex <- terra::extract(x = climate_input,
                            y = x[, c(lon, lat)])

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

  # Rescale, I feel this needs some more though/justification,
  # but Daniele is happy with it for now
  if (rescale) {
    bio[, 2] <- bio[, 2]/150
    bio[, 3] <- bio[, 3]/10
    bio[, 4] <- bio[, 4]/50
    bio[, 5] <- log10(1 + bio[, 5])
    bio[, 6:12] <- bio[, 6:12] / 15
    bio[, 13:20] <- log10(1 + bio[, 13:20])

    range[, 2] <- range[, 2]/15
    range[, 3] <- range[, 3]/10
    range[, 4] <- range[, 4]/50
    range[, 5] <- log10(1 + range[, 5])
    range[, 6:12] <- range[, 6:12] / 15
    range[, 13:20] <- log10(1 + range[, 13:20])
  }else{
    warning("Features not rescaled. Rescale manually")
  }

  if (type == "selected") {
    colnames(bio) <- gsub("wc2.1_10m_", "", colnames(bio))
    colnames(range) <- gsub("wc2.1_10m_", "", colnames(range))
    out <- bio %>%
      select(.data$species,
             .data$bio_1,
             .data$bio_4,
             .data$bio_11,
             .data$bio_12,
             .data$bio_15,
             .data$bio_17) %>%
      bind_cols(range %>% select(.data$range_bio_1,
                                 .data$range_bio_4,
                                 .data$range_bio_11,
                                 .data$range_bio_12,
                                 .data$range_bio_15,
                                 .data$range_bio_17))
  }else{
    out <- left_join(bio, range, by = "species")
  }
  # return
  return(out)
}
