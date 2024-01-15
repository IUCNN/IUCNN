
data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments

# 1. Feature and label preparation
features <- iucnn_prepare_features(training_occ, type = "geographic")
labels <- iucnn_prepare_labels(training_labels, features) # Training labels

# Model-testing
logfile <- paste0("model_testing_results.txt")
model_testing_results <- iucnn_modeltest(features,
                                       labels,
                                       logfile,
                                       model_outpath = 'iucnn_modeltest',
                                       mode = 'nn-class',
                                       seed = 1234,
                                       dropout_rate = c(0.0,0.1,0.3),
                                       n_layers = c('30','40_20','50_30_10'),
                                       cv_fold = 2,
                                       init_logfile = TRUE)


test_that("iucnn_best_model works", {
  # Selecting best model based on chosen criterion
  best_iucnn_model <- iucnn_best_model(model_testing_results,
                                       criterion = 'val_acc',
                                       require_dropout = TRUE)
  expect_equal(length(best_iucnn_model), 44)
  expect_s3_class(best_iucnn_model, "iucnn_model")
})
