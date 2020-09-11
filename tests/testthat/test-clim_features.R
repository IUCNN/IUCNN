data("orchid_occ")
orchid<- CLIM_features(orchid_occ)

test_that("is a data,frame", {
  expect_true(is.data.frame(orchid))
})

test_that("right amount of columns", {
  expect_s3_class(orchid, "data.frame")
  expect_equal(ncol(orchid), 20)
})

test_that("all columns numeric", {
  expect_type(orchid$species, "character")
})

