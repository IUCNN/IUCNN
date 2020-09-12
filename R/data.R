#' IUCN threat categories for 884 orchid species
#'
#' A dataset containing the International Union for the Conservation of Nature's Global
#'  Red List conservation assessments for 884 species of orchids.
#'
#' @format A data frame with 884 rows and 2 variables:
#' \describe{
#'   \item{species}{The canonical species name}
#'   \item{labels}{The IUCN conservation assessment, converted to numerical for use with IUCNN.
#'   0 =  Critically Endangered (CR), 1 = Endangered (EN), 2 = Vulnerable (VU), 3 = Near Threatened (NT), 4 = Least Concern (LC)}
#' }
#' @source \url{https://www.iucnredlist.org/}
"labels_detail"


#' Binary Threat Status for 884 orchid species
#'
#' A dataset containing the International Union for the Conservation of Nature's
#' Global Red List conservation assessments for 884 species of orchids.
#' Converted to the Binary "threatened/not threatened" level.
#'
#' @format A data frame with 884 rows and 2 variables:
#' \describe{
#'   \item{species}{The canonical species name}
#'   \item{labels}{The IUCN conservation assessment, converted to numerical for use with IUCNN.
#'   0 =  Threatened, 1 = Not Threatened}
#' }
#' @source \url{https://www.iucnredlist.org/}
"labels_broad"


#' Geographic Occurrence Records for Not Evaluated Orchids
#'
#' A dataset containing geo-referenced occurrences of 100 Orchid species without
#' existing IUCN Red List assessment ("Not Evaluated"). This is example data
#' to predict the IUCN status.
#'
#' @format A data frame with 14,900 rows and 3 variables:
#' \describe{
#'   \item{species}{The canonical species name}
#'   \item{decimallongitude}{longitudinal coordinates}
#'   \item{decimallatitude}{latitudinal coordinates}
#' }
#' @source \url{https://www.gbif.org/}
"orchid_target"


#' Geographic Occurrence Records for Orchids with IUCN assessment
#'
#' A dataset containing geo-referenced occurrences of 884 Orchid species with
#' existing IUCN Red List assessment ("CR", "EN", "VU", "NT", "LC"). This is example data
#' to train an IUCNN model.
#'
#' @format A data frame with 125,412 rows and 3 variables:
#' \describe{
#'   \item{species}{The canonical species name}
#'   \item{decimallongitude}{longitudinal coordinates}
#'   \item{decimallatitude}{latitudinal coordinates}
#' }
#' @source \url{https://www.gbif.org/}
"orchid_occ"


#' Geographic Occurrence Records for Orchids with IUCN assessment
#'
#' A dataset containing geo-referenced occurrences of 884 Orchid species with
#' existing IUCN Red List assessment ("CR", "EN", "VU", "NT", "LC"). This is example data
#' to train an IUCNN model.
#'
#' @format A data frame with 884 rows and 3 variables:
#' \describe{
#'   \item{species}{The canonical species name}
#'   \item{category_broad}{The IUCN conservation assessment, converted to numerical for use with IUCNN.
#'   Threatened ("CR", "EN", or "VU") Not Threatened ("NT" or "LC")}
#'   \item{category_detail}{The IUCN conservation assessment, converted to numerical for use with IUCNN.
#'   0 =  Critically Endangered (CR), 1 = Endangered (EN), 2 = Vulnerable (VU), 3 = Near Threatened (NT), 4 = Least Concern (LC)}
#' }
#' @source \url{https://www.iucnredlist.org/}
"orchid_classes"
