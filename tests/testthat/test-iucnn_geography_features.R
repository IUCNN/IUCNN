n <- sample(10:200, 2)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

test_that("iucnn_geography_features works", {
  skip_on_cran()
  res <- iucnn_geography_features(dat)
  expect_type(res, "list")
  expect_equal(dim(res), c(2, 10))
})

test_that("iucnn_geography_features rescale = TRUE works", {
  skip_on_cran()
  res <- iucnn_geography_features(dat, rescale = FALSE)
  expect_type(res, "list")
  expect_equal(dim(res), c(2, 10))
})
