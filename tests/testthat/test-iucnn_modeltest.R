skip_on_cran()
skip_if_offline()

data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments

# 1. Feature and label preparation
# Training
## Generate features

# Model-testing
features <- iucnn_prepare_features(training_occ)
test_that("iucnn_prepare_features works", {
  skip_on_cran()
  skip_if_offline()
  expect_type(features, "list")
  expect_equal(nrow(features), 889)
})

labels_train <- iucnn_prepare_labels(training_labels, features) # Training labels
test_that("iucnn_prepare_labels works", {
  skip_on_cran()
  skip_if_offline()
  expect_type(labels_train, "list")
  expect_equal(length(labels_train), 2)
})

mod_test <- iucnn_modeltest(
  x = features,
  lab = labels_train,
  logfile = tempfile(),
  model_outpath = paste0(tempdir(), "/a"),
  mode = "nn-class",
  dropout_rate = c(0.0),
  n_layers = c("30"),
  cv_fold = 1,
  test_fraction = 0.2,
  init_logfile = TRUE)

test_that("iucnn_modeltest detailed works", {
  skip_on_cran()
  skip_if_offline()
  ## train the model
  expect_equal(length(mod_test), 42)
  expect_s3_class(mod_test, "data.frame")
})



if (!file.exists(mod_test$model_outpath)) {
  best_iucnn_model <- iucnn_best_model(mod_test,
                                       criterion = 'val_acc',
                                       require_dropout = FALSE)

  test_that("iucnn_best_model works", {
    skip_on_cran()
    skip_if_offline()
    # Selecting best model based on chosen criterion
    expect_equal(length(best_iucnn_model), 44)
    expect_s3_class(best_iucnn_model, "iucnn_model")
  })
}

m <- iucnn_train_model(
  x = features,
  lab = labels_train,
  mode = "nn-class",
  overwrite = TRUE)

test_that("nn-class detailed works", {
  skip_on_cran()
  skip_if_offline()
  ## train the model
  expect_equal(length(m), 44)
  expect_s3_class(m, "iucnn_model")
})


# Precit_statu
data("prediction_occ") #occurrences from Not Evaluated species to prdict

if (file.exists(m$trained_model_path)) {
  features_predict <- iucnn_prepare_features(prediction_occ)
  p <- iucnn_predict_status(x = features_predict, model = m)

  test_that("iucnn_predict_status detailed works", {
    skip_on_cran()
    skip_if_offline()
    ## train the model
    expect_equal(length(p), 6)
    expect_s3_class(p, "iucnn_predictions")
  })
}

## Prepare training labels
labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features,
                                     level = "broad")
## train the model
m <- iucnn_train_model(
  x = features,
  lab = labels_train,
  mode = "nn-class",
  overwrite = TRUE)

test_that("nn-class broad works", {
  skip_on_cran()
  skip_if_offline()
  expect_equal(length(m), 44)
  expect_s3_class(m, "iucnn_model")
})


if (file.exists(m$trained_model_path)) {
  imp_def <- iucnn_feature_importance(x = m)
  test_that("iucnn_featureimportance works", {
    skip_on_cran()
    skip_if_offline()
    expect_equal(length(imp_def), 3)
    expect_s3_class(imp_def, "iucnn_featureimportance")
  })
}

if (file.exists(m$trained_model_path)) {
  imp_cust <- iucnn_feature_importance(
    x = m,
    feature_blocks = list(block1 = c(1, 2, 3, 4),
                          block2 = c(5, 6, 7, 8)),
    provide_indices = TRUE
  )
  test_that("iucnn_featureimportance feature blocks works", {
    skip_on_cran()
    skip_if_offline()


    expect_equal(length(imp_cust), 3)
    expect_s3_class(imp_cust, "iucnn_featureimportance")

  })
}
