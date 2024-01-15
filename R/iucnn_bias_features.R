#' Extract Bias Features from Occurrence Records
#'
#'Use the sampbias method to assess the geographic sampling bias at the locations where a species is collected and the range of
#'sampling bias for all records per species.Values summarized per species are the median and the 0.05 to 0.95 percentiles.
#'
#'See the ?sampbias::calculate_bias for details.
#'
#'@param res numeric. The resolution of the default resolution to calculate sampling bias. In decimal degrees.
#'@param ras a SpatRaster object. Alternative to res, a sample SpatRaster to
#'  calculate sampling bias. Needs to use the same CRS as the coordinates in x.
#'@param plot logical. Should the results of the sampbias analysis be plotted for diagnostics?
#'@inheritParams iucnn_geography_features
#'
#'@return a data.frame of bias features
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#'\dontrun{
#' dat <- data.frame(species = c("A", "b"),
#'                   decimallongitude = runif(200, 10, 15),
#'                   decimallatitude = runif(200, -5, 5))
#'iucnn_bias_features(dat)
#'
#'}
#'
#'
#' @export
#' @importFrom dplyr group_by mutate select summarize
#' @importFrom terra extract
#' @importFrom magrittr %>%
#' @importFrom stats median quantile
#' @importFrom checkmate assert_character assert_data_frame assert_logical assert_numeric

iucnn_bias_features <- function(x,
                                species = "species",
                                lon = "decimallongitude",
                                lat = "decimallatitude",
                                res = 0.5,
                                ras = NULL,
                                plot = TRUE) {

  #check if sampbias is installed
  if (!requireNamespace("sampbias", character.only = TRUE)) {
    stop("Bias features require the 'sampbias' package. Install from https://github.com/azizka/sampbias")
  }
  # assertions
  assert_data_frame(x)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)
  assert_numeric(res)
  assert_logical(plot)


  # Prepare the input data
  inp <- x %>%
    dplyr::select(species = .data[[species]], decimalLongitude = .data[[lon]], decimalLatitude = .data[[lat]])

  # run sampbias analysis
  sampbias_out <- sampbias::calculate_bias(x = inp, res = res)

  # write summary of samp bias to screen
  summary(sampbias_out)

  # project bias through space
  proj <- sampbias::project_bias(sampbias_out)

  # plot results if plot argument is set
  if (plot) {
    plot(sampbias_out)
    sampbias::map_bias(proj)
  }

  # Extract values for each record
  bias_extract <- terra::extract(proj[["Total_percentage"]],
                                  inp[, c("decimalLongitude", "decimalLatitude")])

  # summarize the mean value and range for each species
  bias_feat <- inp %>%
    dplyr::select(.data[[species]]) %>%
    mutate(bias_feat = bias_extract) %>%
    mutate(bias_feat = .data$bias_feat / 100) %>%
    group_by(.data$species) %>%
    summarize(bias_median = median(.data$bias_feat, na.rm = TRUE),
              bias_min = quantile(.data$bias_feat, probs = 0.05, na.rm = TRUE),
              bias_max = quantile(.data$bias_feat, probs = 0.95, na.rm = TRUE)) %>%
    mutate(bias_range = .data$bias_max - .data$bias_min) %>%
    dplyr::select(.data$species, .data$bias_median, .data$bias_range)

  # return value
  return(bias_feat)
}
