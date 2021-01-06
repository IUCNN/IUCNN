



#'@export
#'@importFrom


assess_species <- function(x = "Boreava aptera",
                           training_taxon = "Boreava",
                           IUCNkey = NULL,
                           method = "nn",
                           accuracy_target = 0.8,
                           reference_target = NULL,
                           reference_occurrences = NULL){
  # assertions
  assert_character(x)
  assert_character(training_taxon)
  assert_character(IUCNkey)
  aasert_character(method)
  assert_numeric(accuracy_target)
  # Obtain IUCN status, check out how many species if possible

  # check how much occurrences and species are availble, if excessive, warning

  if(length(x) > 1){
    target_check <-  sum(unlist(lapply(lapply(rgbif::occ_search(scientificName = x,
                                                                limit = 0,
                                                                hasCoordinate = TRUE),
                                              "[[", "meta"), "[[", "count")))
  }else{
    target_check <-  rgbif::occ_search(scientificName = x, limit = 0, hasCoordinate = TRUE)$meta$count
  }

  if(length(training_taxon) > 1){
    reference_check <-  sum(unlist(lapply(lapply(rgbif::occ_search(scientificName = training_taxon,
                                                                   limit = 0,
                                                                   hasCoordinate = TRUE),
                                                 "[[", "meta"), "[[", "count")))
  }else{
    reference_check <- rgbif::occ_search(scientificName = training_taxon,
                                         limit = 0,
                                         hasCoordinate = TRUE)$meta$count
  }

  if(sum(target_check, reference_check) > 100000){
    warning(sprintf("Large number of occurence records (%s).", sum(target_check, reference_check)))
  }

  # get occurrences from GBIF
  message("Downloading occurrence data")

  target_occ <- bRacatus::getOcc(x)
  reference_occ <- bRacatus::getOcc(training_taxon)

  inp <- dplyr::bind_rows(target_occ, reference_occ) %>%
    select(species,
           decimallongitude = decimalLongitude,
           decimallatitude = decimalLatitude) %>%
    filter(!is.na(species))

  # Cleaning I CoordinateCleaner
  message("Removing problematic records based on gazetteers")

  dat <- CoordinateCleaner::clean_coordinates(inp) %>%
    filter(.summary)

  # Generate features
  message("Generating features.")

  if(method == "cnn"){

  }else{
    geo <- geo_features(dat)
    bio <- biome_features(dat)
    clim <- clim_features(dat)
    foot <- footprint_features(dat)

  }

  # run assessment
  if(method == "nn"){

  }else if(method == "bnn"){

  }else if(method == "cnn"){

  }

  # write results to screen, warning for accuracy

  # return output object
}


