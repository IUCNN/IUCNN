#' Obtain Biome Features from Occurrence Records
#'
#' Will code all species in the input file into biomes based on an
#' intersection of the coordinates with a shape file. The biome scheme can
#' be user provided or by default will download the WWF biomes.
#'
#'
#' If biome_input is NULL this will download  the WWF biome scheme from
#' https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world
#' and save them in the working directory
#'
#'@inheritParams iucnn_prepare_features
#'@param biome_input a simple features collection of geometry type polygon,
#'contain polygons of different biomes.
#'If NULL, the WWF biome scheme is downloaded from
#'https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world
#'@param biome_id a character string. The name of the column
#'with the biome names in biome_input.
#'Default is "BIOME"
#'@param remove_zeros logical. If TRUE biomes without occurrence of
#'any species are removed from the features.
#'Default = FALSE
#'
#'@return a data.frame of climatic features
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#' \dontrun{
#' dat <- data.frame(species = c("A", "b"),
#'                   decimallongitude = runif(200, 10, 15),
#'                   decimallatitude = runif(200, -5, 5))
#'
#' iucnn_biome_features(dat)
#'}
#'
#' @export
#' @importFrom dplyr bind_cols select distinct mutate matches
#' @importFrom sf st_as_sf st_crs st_geometry st_geometry<- st_intersects st_is_valid st_make_valid st_read
#' @importFrom magrittr %>%
#' @importFrom tidyr pivot_wider
#' @importFrom tibble as_tibble
#' @importFrom utils download.file unzip
#' @importFrom checkmate assert_character assert_data_frame assert_logical assert_numeric

iucnn_biome_features <- function(x,
                                 species = "species",
                                 lon = "decimallongitude",
                                 lat = "decimallatitude",
                                 biome_input = NULL,
                                 biome_id = "BIOME",
                                 download_folder = tempdir(),
                                 remove_zeros = FALSE){

  #assertions
  assert_data_frame(x)
  assert_numeric(x[[lon]], any.missing = FALSE, lower = -180, upper = 180)
  assert_numeric(x[[lat]], any.missing = FALSE, lower = -90, upper = 90)
  assert_character(download_folder, null.ok = TRUE)
  assert_logical(remove_zeros)

  # get biome data if necessary
  if (is.null(biome_input)) {
    # set download path
    if (is.null(download_folder)) {
      download_folder <- getwd()
    }
    if (!dir.exists(download_folder)) {
      dir.create(download_folder)
    }

    # Download biomes shape
    if (!file.exists(file.path(
      download_folder,
      "WWF_ecoregions",
      "official",
      "wwf_terr_ecos.shp"
    ))) {
      download.file(
        "https://files.worldwildlife.org/wwfcmsprod/files/Publication/file/6kcchn7e3u_official_teow.zip",
        destfile = file.path(download_folder, "wwf_ecoregions.zip")
      )
      unzip(
        file.path(download_folder, "wwf_ecoregions.zip"),
        exdir = file.path(download_folder, "WWF_ecoregions")
      )
      file.remove(file.path(download_folder, "wwf_ecoregions.zip"))
    }
    #load biomes
    biome_input <- sf::st_read(dsn = file.path(download_folder,
                                               "WWF_ecoregions",
                                               "official"),
                               layer = "wwf_terr_ecos", quiet = TRUE)
  }

  # check if input is valid
  valid_test <- all(st_is_valid(biome_input))

  if (!valid_test) {
    biome_input <- st_make_valid(biome_input)

    valid_test2 <- all(st_is_valid(biome_input))
    if (!valid_test2) {
      stop("input biome polygon contains invalid geometries")
    }else{
      warning("input biome polygon contained invalid geometries. Fixed using 'st_make_valid'")
    }
  }

  # prepare input points
  pts <- sf::st_as_sf(x,
                      coords = c(lon, lat),
                      crs = st_crs(biome_input))

  # The point in polygon test for the point records
  biom <- suppressMessages(sf::st_join(pts, biome_input))

  # prepare output data
  st_geometry(biom) <- NULL

  biom <- biom %>%
    tidyr::as_tibble() %>%
    #select relevant columns
    dplyr::select(species = .data[[species]],
                  biome = .data[[biome_id]]) %>%
    #only one entry per species per biome
    distinct() %>%
    # pivot wider
    dplyr::mutate(presence = 1) %>%
    tidyr::pivot_wider(id_cols = .data$species,
                       names_from = .data$biome,
                       values_from = .data$presence) %>%
  dplyr::select(!dplyr::matches("NA"))

  # replace NAs
  biom[is.na(biom)] <- 0

  # If desired by the user add biomes without any occurrences
  if (!remove_zeros) {
    #Get a list of all biomes in the dataset
    all_biomes <- unique(biome_input[[biome_id]])

    test <- all_biomes[!all_biomes %in% names(biom)]

    if (length(test) > 0) {
      add <- data.frame(t(test))
      names(add) <- test
      add[] <- 0

      biom <- bind_cols(biom, add)
    }
  }else{
    biom <- biom[,c(TRUE, colSums(biom[,-1]) > 0)]
  }
  # order biomes

  # change names
  names(biom)[-1] <- paste("biome_", names(biom)[-1], sep = "")

  # return output
  class(biom) <- c("iucnn_features", class(biom))

  return(biom)
}
