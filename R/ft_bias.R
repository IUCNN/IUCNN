#' Extract Bias Features from Occurrence Records
#'
#'Use the sampbias method to assess the geographic sampling bias at the locations where a species is collected and the range of
#'sampling bias for all records per species.Values summarized per species are the median and the 0.05 to 0.95 percentiles.
#'
#'See the ?sampbias::calcualte_bias for details.
#'
#'@param res numeric. The resolution of the default resolution to calculate sampling bias. In decimal degrees.
#'@param ras a raster object. Alternative to res, a sample raster to calculate sampling bias. Needs to use the same CRS as
#'the coordinates in x.
#''@param plot logical. Should the results of the sampbias analysis be plotted for diagnostics?
#'@inheritParams ft_geo
#'
#'@return a data.frame of bias features
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
#'ft_bias(dat)
#'}
#'
#'
#' @export
#' @importFrom dplyr group_by mutate select summarize
#' @importFrom sampbias calculate_bias map_bias
#' @importFrom raster extract getData
#' @importFrom magrittr %>%
#' @importFrom stats median quantile
#' @importFrom checkmate assert_character assert_data_frame assert_logical assert_numeric

ft_bias <- function(x,
                    species = "species",
                    lon = "decimallongitude",
                    lat = "decimallatitude",
                    res = 0.5,
                    ras = NULL,
                    plot = TRUE){

  # assertions
  assert_data_frame(x)
  assert_character(x[[species]], any.missing = FALSE, min.chars = 1)
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
  if(plot){
    plot(sampbias_out)
    sampbias::map_bias(proj)
  }

  # Extract values for each record
  bias_extract <- raster::extract(proj[["Total_percentage"]], inp[, c("decimalLongitude", "decimalLatitude")])

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
