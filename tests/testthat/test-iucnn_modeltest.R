skip_on_cran()

data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments

# 1. Feature and label preparation
features <- iucnn_prepare_features(training_occ,
                                   type = "geographic") # Training features


# Model-testing
test_that("iucnn_prepare_features works", {
  skip_on_cran()
  expect_type(features, "list")
  expect_equal(nrow(features), 889)
})

labels_train <- iucnn_prepare_labels(training_labels, features) # Training labels

test_that("iucnn_prepare_labels works", {

  expect_type(labels_train, "list")
  expect_equal(length(labels_train), 2)

})

mod_test <- iucnn_modeltest(x = features,
                            lab = labels_train,
                            logfile = "model_testing_results-2.txt",
                            model_outpath = paste0(tempdir(), "/a"),
                            mode = "nn-class",
                            dropout_rate = c(0.0),
                            n_layers = c("30"),
                            cv_fold = 1,
                            test_fraction = 0.2,
                            init_logfile = TRUE)

test_that("iucnn_modeltest detailed works", {
  skip_on_cran()

  ## train the model
  expect_equal(length(mod_test), 42)
  expect_s3_class(mod_test, "data.frame")
})

test_that("iucnn_best_model works", {
  # Selecting best model based on chosen criterion
  best_iucnn_model <- iucnn_best_model(mod_test,
                                       criterion = 'val_acc',
                                       require_dropout = FALSE)
  expect_equal(length(best_iucnn_model), 44)
  expect_s3_class(best_iucnn_model, "iucnn_model")
})


m <- iucnn_train_model(
  x = features,
  lab = labels_train,
  mode = "nn-class",
  overwrite = TRUE
)

test_that("nn-class detailed works", {
  skip_on_cran()

  ## train the model
  expect_equal(length(m), 44)
  expect_s3_class(m, "iucnn_model")
})


# Precit_statu
data("prediction_occ") #occurrences from Not Evaluated species to prdict
features_predict <- iucnn_prepare_features(prediction_occ,
                                          type = "geographic") # Prediction features



test_that("iucnn_predict_status detailed works", {
  skip_on_cran()
  ## train the model
  p <- iucnn_predict_status(x = features_predict, model = m)
  expect_equal(length(p), 6)
  expect_s3_class(p, "iucnn_predictions")
})



test_that("nn-class broad works", {
  skip_on_cran()

  ## Prepare training labels
  labels_train <- iucnn_prepare_labels(x = training_labels,
                                       y = features,
                                       level = "broad")

  ## train the model
  m <- iucnn_train_model(
    x = features,
    lab = labels_train,
    mode = "nn-class",
    overwrite = TRUE
  )

  expect_equal(length(m), 44)
  expect_s3_class(m, "iucnn_model")
})

