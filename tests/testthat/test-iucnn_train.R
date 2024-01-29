data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments
# Training
## Generate features
features <- iucnn_prepare_features(training_occ,
                                   type = "geographic")

## Prepare training labels
labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features)


