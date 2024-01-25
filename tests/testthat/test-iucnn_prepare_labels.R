n <- sample(seq(10, 200, 2), 1)
dat <- data.frame(species = c("A","B"),
                  decimallongitude = runif(n, 10, 15),
                  decimallatitude = runif(n, -5, 5))

labs <- data.frame(species = c("A","B"),
                   labels = c("CR", "LC"))
suppressMessages(suppressWarnings(
  features <- iucnn_prepare_features(dat,
                                     type = "geographic")
))


test_that("iucnn_prepare_labels works", {

  res <- iucnn_prepare_labels(x = labs, y = features)
  expect_type(res, "list")
  expect_equal(length(res), 2)

})
