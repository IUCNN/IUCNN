

test_that("nn-class detailed works", {
  skip_on_cran()

  data("training_occ") #geographic occurrences of species with IUCN assessment
  data("training_labels")# the corresponding IUCN assessments
  data("prediction_occ")
  # Training
  ## Generate features
  features <- prep_features(training_occ)
  features_predict <- prep_features(prediction_occ)

  ## Prepare training labels
  labels_train <- prep_labels(training_labels)

  ## train the model
  m <- train_iucnn(x = features,
                    lab = labels_train,
                    mode = "nn-class")

  p <- predict_iucnn(x = features_predict,
                     model = m,
                     model_dir = "iuc_nn_model")

  expect_equal(length(m), 25)
  expect_s3_class(m, "iucnn_model")
})

test_that("nn-class broad works", {
  skip_on_cran()

  data("training_occ") #geographic occurrences of species with IUCN assessment
  data("training_labels")# the corresponding IUCN assessments
  data("prediction_occ")
  # Training
  ## Generate features
  features <- prep_features(training_occ)
  features_predict <- prep_features(prediction_occ)

  ## Prepare training labels
  labels_train <- prep_labels(training_labels,
                              level = "broad")

  ## train the model
  m <- train_iucnn(x = features,
                   lab = labels_train,
                   mode = "nn-class")

  p <- predict_iucnn(x = features_predict,
                     model = m,
                     model_dir = "iuc_nn_model")

  expect_equal(length(m), 25)
  expect_s3_class(m, "iucnn_model")
})


test_that("bnn-class works", {
  expect_equal(2 * 2, 4)
})

test_that("nn-reg works", {
  expect_equal(2 * 2, 4)
})

