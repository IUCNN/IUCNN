test_that("iucnn_prepare_phy works", {
  skip_on_cran()
  require(ape)
  tree <- rphylo(n = 10, birth = 0.1, death = 0)
  res <- iucnn_prepare_phy(phy = tree, numeigen = 5)
  expect_type(res, "list")
  expect_equal(dim(res), c(10, 6))
})
