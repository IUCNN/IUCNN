#' IUCN threat categories for 884 orchid species
#'
#' A dataset containing the International Union for the Conservation of Nature's Global
#'  Red List conservation assessments for 884 species of orchids.
#'
#' @format A data frame with 53940 rows and 10 variables:
#' \describe{
#'   \item{species}{The canonical species name}
#'   \item{labels}{The IUCN conseration assessment, converted to numerical for use with IUCNN.
#'   0 =  Critically ENdangered (CR), 1 = Endangered (EN), 2 = Vulnerable (VU), 3 = Near Threatened (NT), 4 = Least Concern (LC)}
#' }
#' @source \url{https://www.iucnredlist.org/}
"labels_detail"

#' Binary Threat Status for 884 orchid species
#'
#' A dataset containing the International Union for the Conservation of Nature's
#' Global Red List conservation assessments for 884 species of orchids.
#' Converted to the Binary "threatened/not threatened" level.
#'
#' @format A data frame with 53940 rows and 10 variables:
#' \describe{
#'   \item{species}{The canonical species name}
#'   \item{labels}{The IUCN conseration assessment, converted to numerical for use with IUCNN.
#'   0 =  Threatened, 1 = Not Threatened}
#' }
#' @source \url{https://www.iucnredlist.org/}
"labels_broad"
