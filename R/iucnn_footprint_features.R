#' Extract Human Footprint Index Features from Occurrence Records
#'
#' Bins the human footprint index into a set of bins and the
#' fraction of occurrence records
#' of a species in each bin are the features.
#' By default the human footprint index is downloaded from
#' https://wcshumanfootprint.org/. THIS FUNCTION WILL DOWNLOAD DATA FROM
#' THE INTERNET AND SAVE IT TO THE  WORKING DIRECTORY. The data files
#' are >200 MB each and downloading may
#' take some time on first execution.
#'
#' By default four categories of increasing human footprint index
#'  ( 1 = lowest, 4 = highest) are selected and rescaled.
#' @inheritParams iucnn_prepare_features
#' @param footp_input an object of the class SpatRaster with values for the
#'   human footprint index. If the SpatRaster object has two or more layers,
#'   different layers are interpreted as different time-slices.
#' @param rescale logical. If TRUE, the values are rescaled using
#' natural logarithm transformation. If FALSE,
#' remember to change the breaks argument.
#' @param year numeric. The years for which to obtain the human footprint index.
#' The default is to the two layers available. Can be a either year, in case only
#' one slice is desired. Other time slices are currently not supported,
#' @param breaks numerical. The breaks to bin the human footprint index
#'  for the final features. The defaults are
#' empirical values for the global footprint and rescale=TRUE.
#' For custom values ensure that they
#' cover the whole value range and are adapted to the value of rescale.
#' @inheritParams iucnn_geography_features
#'
#' @source https://wcshumanfootprint.org/
#'
#' @return a data.frame of human footprint features
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
#' iucnn_footprint_features(dat)
#'}
#'
#' @export
#' @importFrom dplyr filter .data group_by tally mutate select summarize_all
#' @importFrom terra extract rast
#' @importFrom magrittr %>%
#' @importFrom sf st_as_sf st_coordinates st_transform st_crs
#' @importFrom curl has_internet
#' @importFrom readr parse_number
#' @importFrom tidyr pivot_longer pivot_wider
#' @importFrom checkmate assert_character assert_data_frame assert_logical assert_numeric
#' @importFrom geodata footprint

iucnn_footprint_features <- function(x,
                   footp_input = NULL,
                   species = "species",
                   lon = "decimallongitude",
                   lat = "decimallatitude",
                   rescale = TRUE,
                   year = c(1993, 2009),
                   download_folder = tempdir(),
                   breaks = c(0, 0.81, 1.6, 2.3, 100),
                   verbose = FALSE){

  # assertions
  assert_data_frame(x)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)
  assert_logical(rescale)
  assert_numeric(year)
  assert_character(download_folder , null.ok = TRUE)
  assert_numeric(breaks)

  # get human footprint
  if (is.null(footp_input)) {
    if (verbose) {
      message("Downloading Human Footprint data from https://wcshumanfootprint.org/")
    }

    # file path
    # set download path
    if (is.null(download_folder)) {
      download_folder <- getwd()
    }
    # else{
    #   download_folder <- file.path(getwd(), download_folder )
    # }
    if (!dir.exists(download_folder)) {
      dir.create(path = download_folder)
    }

    # test for internet
    if (!curl::has_internet()) {
      warning("No internet connection. Provide input raster via 'footp_inp'")
      return(NULL)
    }

    # download the human footprint raster from https://wcshumanfootprint.org/
    if (length(year) > 1) {
      year2 <- as.list(year)
      lapply(year2, FUN = geodata::footprint, path = download_folder)
    }else{
      geodata::footprint(x = year, path = download_folder)
    }

    # load raster
    footp_inp <-  terra::rast(file.path(download_folder, "landuse",
                                        paste("wildareas-v3-",
                                              year, "-human-footprint_geo.tif",
                                              sep = "")))

  }else{
    ## If no, download
    footp_inp <-  terra::rast(footp_input)
  }

  # extract values
  if (verbose) {
    message("Extracting_footprint_index for occurrence records")
  }
  pts <- sf::st_as_sf(x,
                      coords = c(lon, lat),
                      crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
  pts <- pts %>% sf::st_transform(sf::st_crs(footp_inp))
  pts <- sf::st_coordinates(pts)

  footp_ex <- terra::extract(x = footp_inp, y = pts)
  colnames(footp_ex) <- year

  if (rescale) {
    footp_ex <- log(footp_ex)
  }

  footp_ex <- data.frame(species = x[[species]],
                         footp_ex)

  # summarize per species and  create features object
  if (verbose) {
  message("Summarizing information per species")
  }

  ## classify the footprint into equal-sized bins
  footp_ex[, -1] <- apply(footp_ex[, -1],
                          2,
                          function(k){cut(k,
                                          breaks = breaks,
                                          labels = 1:(length(breaks) - 1),
                                          right = FALSE)})

  # prepare feature summary
  out <- footp_ex %>%
    pivot_longer(-.data$species,
                 names_to = "year",
                 values_to = "HFP")

  # check for NAs (i.e. records that did not yield any human footprint)
  nas <- sum(is.na(out$HFP))

  if (nas > 0) {
    warning(sprintf("Ignoring %s records without data in the input raster",
                    nas))
  }

  # summarize features
  out <- out %>%
    filter(!is.na(.data$HFP)) %>%
    group_by(.data$species, .data$year, .data$HFP) %>%
    tally() %>%
    group_by(.data$species, .data$year) %>%
    mutate(frac = round(.data$n / sum(.data$n), 2)) %>%
    mutate(label = paste("humanfootprint",
                         parse_number(.data$year),
                         .data$HFP, sep = "_")) %>%
    dplyr::ungroup() %>%
    dplyr::select(.data$species, .data$label, .data$frac) %>%
    pivot_wider(id_cols = .data$species,
                names_from = .data$label,
                values_from = .data$frac)

  # replace NAs
  out[is.na(out)] <- 0

  out <- out[, c("species", sort(names(out[-1]))) ]

  return(out)
}
