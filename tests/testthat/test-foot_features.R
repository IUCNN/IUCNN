test_that("footprint features work", {
  skip_on_cran()
  skip_if_offline(host = "r-project.org")

  data("prediction_occ")
  foot <- ft_foot(prediction_occ)
  expect_equal(ncol(foot), 9)
  expect_equal(nrow(foot), 98)
})
