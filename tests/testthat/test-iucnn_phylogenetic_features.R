dat <- data.frame(species = c("A","B", "X"),
                  decimallongitude = runif(180,10,15),
                  decimallatitude = runif(180,-5,5))

tree <- ape::rphylo(n = 10, birth = 0.1, death = 0)
tree$tip.label <- LETTERS[1:10]
phy <- iucnn_prepare_phy(phy = tree)


test_that("iucnn_phylogenetic_features works", {
  skip_on_cran()
  res <- iucnn_phylogenetic_features(x = dat,
                              phy.eigen = phy)

  expect_type(res, "list")
  expect_equal(nrow(res), 3)
})
