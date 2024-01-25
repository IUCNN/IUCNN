n <- sample(seq(10, 200, 2), 1)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

test_that("iucnn_cnn_features works", {
  skip_on_cran()
  expect_warning(res <- iucnn_cnn_features(dat))
  expect_type(res, "list")
  expect_equal(length(res), 2)
})

