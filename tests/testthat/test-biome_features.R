test_that("biome features work", {
  skip_on_cran()
  skip_if_offline(host = "r-project.org")

  data("prediction_occ")
  biom <- ft_biom(prediction_occ)
  expect_equal(ncol(biom), 17)
  expect_equal(nrow(biom), 100)

  biom <- ft_biom(prediction_occ, remove_zeros = TRUE)
  expect_equal(ncol(biom), 16)
})
