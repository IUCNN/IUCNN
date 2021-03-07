#' Extract Geographic Features from Occurrence Records
#'
#' Calculates the number of occurrences, number of unique occurrences,
#' mean latitude, mean longitude, latitudinal range, longitudinal range,
#' eoo, aoo and hemisphere as input features for IUCNN from a list of species occurrences.
#'
#' Coordinate ranges are 90% quantiles, for species with less than three occurrences EOO is set to AOO.
#'
#'@param x a data.frame of species occurrence records including three columns with
#'species name, longitudinal coordinates and latitudinal coordinates (both decimal).
#'@param species a character string. The name of the column with the species names.
#'@param lon a character string. The name of the column with the longitude.
#'@param lat a character string. The name of the column with the latitude.
#'@param rescale logical. If TRUE, the geographic features are rescaled.
#'This is recommended to run IUCNN, and the default. If FALSE, raw (human readable)
#'feature values are returned.
#'
#'@return a data.frame of geographic features
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#'
#' dat <- data.frame(species = "A",
#'                decimallongitude = runif (200,-5,5),
#'                 decimallatitude = runif (200,-5,5))
#'
#'ft_geo(dat)

#'
#' @export
#' @importFrom tidyselect all_of
#' @importFrom dplyr .data distinct left_join mutate summarize group_by n select
#' @importFrom rCAT ConBatch
#' @importFrom magrittr %>%
#' @importFrom stats quantile
#' @importFrom checkmate assert_character assert_data_frame assert_logical

ft_geo <- function(x,
                   species = "species",
                   lon = "decimallongitude",
                   lat = "decimallatitude",
                   rescale = TRUE){

  # assertions
  assert_data_frame(x)
  assert_character(x[[species]], any.missing = FALSE, min.chars = 1)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)
  assert_logical(rescale)

  # Total occurrences
  tot_occ <- x %>%
    group_by(species) %>%
    summarize(tot_occ = n())

  # Other geographic features
  uni <- x %>%
    select(all_of(species), all_of(lat), all_of(lon)) %>%
    distinct()

  geos <- uni%>%
    group_by(species) %>%
    summarize(
      # Unique occurrences
      uni_occ = n(),

      # Mean latitude
      mean_lat = mean(.data[[lat]]) %>%  round(3),

      # Mean longitude
      mean_lon = mean(.data[[lon]]) %>%  round(3),

      # Latitudinal range
      lat_range = (.data[[lat]] %>% quantile(probs = 0.95) + 180) -
        (.data[[lat]] %>% quantile(probs = 0.05) + 180),

      # Latitudinal range
      lon_range = (.data[[lon]] %>% quantile(probs = 0.95) + 180)-
        (.data[[lon]] %>% quantile(probs = 0.05) + 180) ) %>%

    #hemisphere
    mutate(lat_hemisphere = ifelse(.data$mean_lat < 0, 0, 1)) %>%
    mutate(mean_lat = abs(.data$mean_lat))

    # EOO and AOO
    spa <- rCAT::ConBatch(taxa = uni[species] %>%  unlist(),
                           lat = uni[lat] %>%  unlist(),
                           lon = uni[lon] %>%  unlist(),
                           cellsize = 2000) %>%
       dplyr::select(species = .data$taxa,
                     eoo = .data$EOOkm2,
                     aoo = .data$AOO2km) %>%
       mutate(eoo = ifelse(eoo == 0, aoo, eoo)) %>% # set EOO to AOO
       mutate(eoo = round(as.numeric(.data$eoo), 3)) %>%
       mutate(aoo = round(as.numeric(.data$aoo), 3))

    # combine
     out <- tot_occ %>%
       left_join(geos, by = species) %>%
       left_join(spa, by = species)

    # rescale
     if(rescale){
       out <- out %>%
         mutate(tot_occ = log10(1 + .data$tot_occ),
                uni_occ = log10(1 + .data$uni_occ),
                mean_lat = .data$mean_lat / 90,
                mean_lon = .data$mean_lon / 180,
                lat_range = log10(1 + .data$lat_range),
                lon_range = log10(1 + .data$lon_range),
                eoo = log10(1 + .data$eoo),
                aoo = log10(1 + .data$aoo))

     }
     # return
     return(out)
}


