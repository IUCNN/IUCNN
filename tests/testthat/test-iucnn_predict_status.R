data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments
data("prediction_occ") #occurrences from Not Evaluated species to prdict

# Training
## Generate features
features <- iucnn_prepare_features(training_occ,
                                   type = "geographic")

features_predict <- iucnn_prepare_features(prediction_occ,
                                          type = "geographic") # Prediction features


## Prepare training labels
labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features)


test_that("iucnn_predict_status detailed works", {
  skip_on_cran()

  ## train the model
  expect_warning(m <- iucnn_train_model(
    x = features,
    lab = labels_train,
    mode = "nn-class",
    overwrite = TRUE
  ))
  p <- iucnn_predict_status(x = features_predict, model = m)

  expect_equal(length(p), 6)
  expect_s3_class(p, "iucnn_predictions")
})
