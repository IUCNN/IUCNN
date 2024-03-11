n <- sample(seq(10, 200, 2), 1)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

test_that("iucnn_bias_features works", {
  skip_on_cran()
  res <- iucnn_bias_features(dat)
  expect_s3_class(res, "tbl")
  expect_equal(ncol(res), 3)
})

