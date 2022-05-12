#' Prepare Phylogenetic Eigenvectors to Extract Phylogenetic Features
#'
#'Extract Phylogenetic Eigenvectors to represent the phylogeentic relationsships
#'in a matrix format ot use as NN features.
#'

#'@param species
#'the 19 bioclim variables from www.worldclim.org are used as default.
#'@param variance_fraction
#'@param numeigen
#'
#'@return a matrix of phylogenetic Eigenvectors
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#'
#'
#' @export
#' @importFrom PVR PVRdecomp

iucnn_prepare_phy <- function(phy,
                              variance_fraction = 0.8,
                              numeigen = NULL){

  #assertions
  assert_class(phy, classes = "phylo")
  assert_numeric(variance_fraction)
  assert_numeric(numeigen, null.ok = TRUE)

  #produces object of class 'PVR' with eigenvectors
  decomp <- PVRdecomp(tree, type = "newick")
  label.decomp <- as.data.frame(decomp@phylo$tip.label)
  egvec <- as.data.frame(decomp@Eigen$vectors)


  if (is.null(numeigen)){
    egval <- decomp@Eigen$values #extract eigenvalues
    eigPerc <- egval / (sum(egval)) #calculate % of variance
    eigPercCum <- t(cumsum(eigPerc)) #cumulated variance
    #eigenvectors representing more than X% variance
    numeigen <- sum(eigPercCum < variance_fraction)
  }

  # alternatively indicate the number of Eigenvectors
  egOK <- egvec[, 1:numeigen]

  # Change 'numeigen' on above line to a number if you want to specify number of eigenvectors
  eigenTobind <- cbind(label.decomp,egOK) #add names, these are the eigenvectors to merge with trait database

  #Eigenvectors generated in object 'eigenTobind'
  #rename eigenTobind species column so it matches trait dataset species column
  names(eigenTobind)[1] <- "species"

  #return matrix of eigenvectors
  return(eigenTobind)

}
