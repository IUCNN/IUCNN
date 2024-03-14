require(ape)
skip_on_cran()

dat <- data.frame(species = c("A","B", "X"),
                  decimallongitude = runif(180,10,15),
                  decimallatitude = runif(180,-5,5))

tree <- ape::rphylo(n = 10, birth = 0.1, death = 0)
tree$tip.label <- c(LETTERS[1:9], "X")
res <- iucnn_prepare_phy(phy = tree, numeigen = 5)


test_that("iucnn_prepare_phy works", {
  expect_type(res, "list")
  expect_equal(dim(res), c(10, 6))
})

test_that("iucnn_phylogenetic_features works", {
  res <- iucnn_phylogenetic_features(x = dat,
                              phy.eigen = res)

  expect_type(res, "list")
  expect_equal(nrow(res), 3)
})
