data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments

# Training
## Generate features
features <- iucnn_prepare_features(training_occ,
                                   type = "geographic")
## Prepare training labels
labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features)
expect_warning(
  m <- iucnn_train_model(
    x = features,
    lab = labels_train,
    mode = "nn-class",
    overwrite = TRUE
  )
)

test_that("iucnn_featureimportance works", {
  skip_on_cran()
  imp_def <- iucnn_feature_importance(x = m)

  expect_equal(length(imp_def), 3)
  expect_s3_class(imp_def, "iucnn_featureimportance")
})

test_that("iucnn_featureimportance feature blocks works", {
  skip_on_cran()
  imp_cust <- iucnn_feature_importance(x = m,
                                       feature_blocks = list(block1 = c(1,2,3,4),
                                                             block2 = c(5,6,7,8)),
                                       provide_indices = TRUE)

  expect_equal(length(imp_cust), 3)
  expect_s3_class(imp_cust, "iucnn_featureimportance")
})


