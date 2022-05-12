#' Extract Phylogenetic Features Based on Phylogenteic Eigenvectors
#'
#' Extract features for a list of species based on phylogenetic Eigenvectors
#'  prepared with the \\code{iucnn_prepare_phy} function.
#'  Tip labels need to match the species names in x.
#'
#'@inheritParams iucnn_prepare_features
#'@param phy.eigen a matrix of phylogenetic Eigenvector calculated
#'from a phylogenetic tree including trainng and prediction species
#'using the \code{iucnn_prepare_phy} function.
#'
#'@return a data.frame of phylogenetic features
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#' dat <- data.frame(species = c("A","B", "X"),
#'                   decimallongitude = runif (180,10,15),
#'                   decimallatitude = runif (180,-5,5))
#'
#' tree <- rphylo(n = 10, birth=0.1, death=0)
#' phy <- iucnn_prepare_phy(phy = tree)
#'
#'iucnn_phylogenetic_features(x = dat,
#'                            phy.eigen = phy)

#'
#' @export
#' @importFrom dplyr left_join


iucnn_phylogenetic_features <- function(x,
                                        species = "species",
                                        phy.eigen){

  # assertions
  assert_data_frame(x)
  assert_data_frame(phy.eigen)

  #extract features in the correct order
  out <- left_join(data.frame(species = unique(x[[species]])),
                              phy.eigen,
                              by = "species")

  #return features
  return(out)
}
