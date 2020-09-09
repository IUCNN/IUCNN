#' Obtain Biome Features from Occurrence Records
#'
#' Will code all species in the input file into biomes based on an
#' intersection of the coordinates with a shape file. The biome scheme can
#' be user provided or by default will download the wWf biomes.
#'
#'
#' If biome.input is NULL this will download  the WWF biome scheme from
#' https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world
#' and save them in the working directory
#'
#'@param biome.input s simple features collection of geometry type polygon, contain polygons of different biomes.
#'If NULL, the WWF biome scheme is downloaded from
#'https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world
#'@param biome.id a character string. The name of the column with the biome names in biome.input.
#'Default is "BIOME"
#'@param download.path character string. The path were to save the WWF polygons
#'if biome.input is NULL. Default is the working directory
#'@inheritParams geo_features
#'
#'
## biomes from
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
#'biom_features(dat)
#'}
#'
#'
#' @export
#' @importFrom dplyr bind_cols select distinct mutate
#' @importFrom sf st_as_sf st_crs st_geometry st_intersects st_read
#' @importFrom magrittr %>%
#' @importFrom tidyr pivot_wider
#' @importFrom utils download.file unzip
biome_features <- function(x,
                           species = "species",
                           lon = "decimallongitude",
                           lat = "decimallatitude",
                           biome.input = NULL,
                           biome.id = "BIOME",
                           download.path = NULL){

  # get biome data if necessary
  if(is.null(biome.input)){
    # set download path
    if(is.null(download.path)){
      download.path <- getwd()
    }

    # Download biomes shape
    if(!file.exists(file.path(download.path, "WWF_ecoregions",  "official", "wwf_terr_ecos.shp"))){
      download.file("http://assets.worldwildlife.org/publications/15/files/original/official_teow.zip",
                    destfile = file.path(download.path, "wwf_ecoregions.zip"))
      unzip(file.path(download.path, "wwf_ecoregions.zip"),
            exdir = file.path(download.path, "WWF_ecoregions"))
      file.remove(file.path(download.path, "wwf_ecoregions.zip"))
    }

    #load biomes
    biome.input <- sf::st_read(dsn = file.path(download.path, "WWF_ecoregions", "official"),
                               layer = "wwf_terr_ecos")
  }

  # The point in polygon test for the point records
  pts <- sf::st_as_sf(x,
                      coords = c(lon, lat),
                      crs= st_crs(biome.input))

  biom <- st_intersects(pts, biome.input) # this gives the rownames of each point in wwf

  sf::st_geometry(biome.input) <- NULL
  biom2 <-biome.input[as.numeric(biom), biome.id]

  biom <- dplyr::bind_cols(x %>% dplyr::select(species = .data$species),
                   BIOME =  biom2) %>%
    dplyr::distinct() %>%
    dplyr::mutate(presence = 1) %>%
    tidyr::pivot_wider(id_cols = .data$species, names_from = .data$BIOME, values_from = .data$presence) %>%
    replace(is.na(.), 0) %>%
    dplyr::select(-`NA`)

}
