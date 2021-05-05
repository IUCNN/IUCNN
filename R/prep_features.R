#'Prepare features for an IUCNN model
#'
#'A wrapper function to prepare all default features included in IUCNN:
#'geographic, biomes, climate, human footprint.
#'If desired, bias features need to be calculated separately with ft_bias.
#'For more control over feature preparation, you can use the
#'\code{\link{ft_geo}}, \code{\link{ft_biom}}, \code{\link{ft_clim}},
#'\code{\link{ft_foot}} functions.
#'
#'Without internet access, only geographic features are calculated,
#'if the sampbias package is not installed, the bias features are skipped.
#'
#'@inheritParams ft_geo
#'@param type character. The type of features to calculate. Possible options are
#'\dQuote{geographic}, \dQuote{biome}, \dQuote{climate},
#'\dQuote{human footprint}.
#'All except bias are the default.
#'"\dQuote{bias} is only recommended up to the regional scale or below.
#'
#'@return a data.frame of bias features
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
#'prep_features(dat)
#'}
#'
#'@export
#' @importFrom checkmate assert_character assert_data_frame assert_logical
#' @importFrom dplyr left_join

prep_features <- function(x,
                          species = "species",
                          lon = "decimallongitude",
                          lat = "decimallatitude",
                          type = c("geographic", "biomes", "climate", "humanfootprint")){

  # assertions
  assert_data_frame(x)
  assert_character(x[[species]], any.missing = FALSE, min.chars = 1)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)
  assert_character(type)


  #prepare geographic features
  if("geographic" %in% type){
    message("Calculating geographic features.")
    out <- ft_geo(x,
                  species = species,
                  lon = lon,
                  lat = lat)
  }



  if(curl::has_internet()){

    #biomes
    if("biomes" %in% type){
      message("Calculating biome features.")
      bio <- ft_biom(x,
                     species = species,
                     lon = lon,
                     lat = lat)

      if(exists("out")){
        out <- out %>%
          left_join(bio, by = species)
      }else{
        out <- bio
      }
    }

    #climate
    if("climate" %in% type){
      message("Calculating climate features.")
      clim <- ft_clim(x,
                      species = species,
                      lon = lon,
                      lat = lat)

      if(exists("out")){
        out <- out %>%
          left_join(clim, by = species)
      }else{
        out <- clim
      }
    }

    # human footprint
    if("humanfootprint" %in% type){
      message("Calculating human footprint features.")
      foot <- ft_foot(x,
                      species = species,
                      lon = lon,
                      lat = lat)

      if(exists("out")){
        out <- out %>%
          left_join(foot, by = species)
      }else{
        out <- foot
      }
    }
  }else{
    warning("No internet connection, only geographic features created")
  }
  return(out)
}
