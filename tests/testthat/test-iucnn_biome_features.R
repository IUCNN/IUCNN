n <- sample(10:200, 2)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

test_that("iucnn_biome_features works", {
  skip_on_cran()
  expect_warning(res <- iucnn_biome_features(dat))
  expect_type(res, "list")
  expect_equal(dim(res), c(2, 17))
})
