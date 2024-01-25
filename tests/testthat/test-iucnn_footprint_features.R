n <- sample(seq(10, 200, 2), 1)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

test_that("iucnn_footprint_features works", {
  skip_on_cran()
  res <- iucnn_footprint_features(dat)
  expect_type(res, "list")
  expect_equal(nrow(res), 2)
})

