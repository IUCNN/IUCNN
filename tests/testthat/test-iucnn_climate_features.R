n <- sample(seq(10, 200, 2), 1)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

test_that("iucnn_climate_features works", {
  skip_on_cran()
  res <- iucnn_climate_features(dat)
  expect_type(res, "list")
  expect_equal(dim(res), c(2, 13))
})

test_that("iucnn_climate_features rescale = FALSE works", {
  skip_on_cran()
  expect_warning(res <- iucnn_climate_features(dat, rescale = FALSE))
  expect_type(res, "list")
  expect_equal(dim(res), c(2, 13))
})

test_that("iucnn_climate_features type = all works", {
  skip_on_cran()
  res <- iucnn_climate_features(dat, type = "all")
  expect_type(res, "list")
  expect_equal(dim(res), c(2, 41))
})
