data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments

cnn_training_features <- iucnn_cnn_features(training_occ)
cnn_labels <- iucnn_prepare_labels(x = training_labels,
                                   y = cnn_training_features)



test_that("multiplication works", {
  trained_model <- iucnn_cnn_train(cnn_training_features,
                                   cnn_labels,
                                   cv_fold = 1,
                                   overwrite = TRUE,
                                   dropout = 0.1)
  expect_equal(length(trained_model), 45)
  expect_s3_class(trained_model, "iucnn_model")
})
