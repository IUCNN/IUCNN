

test_that("nn-class detailed works", {
  skip_on_cran()

  data("training_occ") #geographic occurrences of species with IUCN assessment
  data("training_labels")# the corresponding IUCN assessments
  data("prediction_occ")
  # Training
  ## Generate features
  features <- iucnn_prepare_features(training_occ)
  features_predict <- iucnn_prepare_features(prediction_occ)

  ## Prepare training labels
  labels_train <- iucnn_prepare_labels(x = training_labels,
                                       y = features)

  ## train the model
  m <- iucnn_train_model(x = features,
                   lab = labels_train,
                   mode = "nn-class",
                   overwrite = TRUE)

  p <- iucnn_predict_status(x = features_predict,
                     model = m)

  expect_equal(length(m), 43)
  expect_s3_class(m, "iucnn_model")
})

test_that("nn-class broad works", {
  skip_on_cran()

  data("training_occ") #geographic occurrences of species with IUCN assessment
  data("training_labels")# the corresponding IUCN assessments
  data("prediction_occ")
  # Training
  ## Generate features
  features <- iucnn_prepare_features(training_occ)
  features_predict <- iucnn_prepare_features(prediction_occ)

  ## Prepare training labels
  labels_train <- iucnn_prepare_labels(x = training_labels,
                                       y = features,
                                       level = "broad")

  ## train the model
  m <- iucnn_train_model(x = features,
                   lab = labels_train,
                   mode = "nn-class",
                   overwrite = TRUE)

  p <- iucnn_predict_status(x = features_predict,
                     model = m)

  expect_equal(length(m), 43)
  expect_s3_class(m, "iucnn_model")
})
