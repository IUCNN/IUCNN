n <- sample(10:200, 2)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

test_that("iucnn_bias_features works", {
  skip_on_cran()
  if (!require(sampbias, quietly = TRUE)) {
  expect_error(res <- iucnn_bias_features(dat))
  } else {
    #Input after fixing sampbias
  }
})

