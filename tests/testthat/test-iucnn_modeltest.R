data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments

# 1. Feature and label preparation
features <- iucnn_prepare_features(training_occ, type = "geographic") # Training features
labels_train <- iucnn_prepare_labels(training_labels, features) # Training labels


# Model-testing


test_that("iucnn_modeltest detailed works", {
  skip_on_cran()

  ## train the model
  mod_test <- iucnn_modeltest(x = features,
                              lab = labels_train,
                              logfile = "model_testing_results-2.txt",
                              model_outpath = "iucnn_modeltest-2",
                              mode = "nn-class",
                              dropout_rate = c(0.0, 0.1, 0.3),
                              n_layers = c("30", "40_20", "50_30_10"),
                              cv_fold = 2,
                              init_logfile = TRUE)
  expect_equal(length(mod_test), 42)
  expect_s3_class(mod_test, "data.frame")
})
