#'Prepare features for an IUCNN model
#'
#'A wrapper function to prepare all default features included in IUcNN:
#'geographic, biomes, climate, human footprint.
#'If desired, bias features need to be calculated seperately with ft_bias.
#'For more control over feature preparation, you can use the
#'\code{\link{ft_geo}}, \code{\link{ft_biom}}, \code{\link{ft_clim}},\code{\link{ft_foot}} functions.
#'
#'Without internet access, only geographic features are calcualted,
#'if the sampbias package is not installed, the bias features are skipped.
#'
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
#'prep_features(dat)
#'}
#'
#'@export
#' @importFrom checkmate assert_character assert_data_frame assert_logical
#' @importFrom dplyr left_join

prep_features <- function(x,
                          species = "species",
                          lon = "decimallongitude",
                          lat = "decimallatitude"){

  # assertions
  assert_data_frame(x)
  assert_character(x[[species]], any.missing = FALSE, min.chars = 1)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)


  #prepare geographic features
  message("Calcualting geographic features.")
  out <- ft_geo(x,
                species = species,
                lon = lon,
                lat = lat)

  #if internet run biomes, climate and footprint
  if(curl::has_internet()){
    message("Calcualting biome features.")
    bio <- ft_biom(x,
                   species = species,
                   lon = lon,
                   lat = lat)

    message("Calcualting climate features.")
    clim <- ft_clim(x,
                    species = species,
                    lon = lon,
                    lat = lat)

    message("Calcualting human footprint features.")
    foot <- ft_foot(x,
                    species = species,
                    lon = lon,
                    lat = lat)

    out <- out %>%
      left_join(bio, by = species) %>%
      left_join(clim, by = species) %>%
      left_join(foot, by = species)
    }else{
    warning("No internet connection, only geographic features created")
    }

  # #if sampbias is installed, run bias features
  # if(!require("sampbias", quietly = TRUE)){
  #   warning("sampbias not fund, skipping bias features. Install package sampbias.")
  # }else{
  #   samp <- t_bias(x)
  #
  #   out <- out %>%
  #     left_join(geo, out, by = species)
  # }
  return(out)
}
