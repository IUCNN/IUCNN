## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
 collapse = TRUE,
 comment = "#>",
 warning = FALSE,
 message = FALSE,
 fig.width = 8
)

## ----setup--------------------------------------------------------------------
library(IUCNN)
library(magrittr)
library(dplyr)

## ---- eval = FALSE------------------------------------------------------------
#  install.packages("devtools")
#  library(devtools)
#  library(IUCNN)

## ---- eval = FALSE------------------------------------------------------------
#  install.packages(reticulate)
#  library("reticulate")
#  install_miniconda()

## ---- eval = FALSE------------------------------------------------------------
#  reticulate::conda_install("r-reticulate","tensorflow=2.4")
#  reticulate::py_install("https://github.com/dsilvestro/npBNN/archive/v0.1.10.tar.gz",
#                         pip = TRUE)

## ---- results='hide'----------------------------------------------------------
data("training_occ") #geographic occurrences of species with IUCN assessment
data("prediction_occ")

features_train <- iucnn_prepare_features(training_occ) # Training features
features_predict <- iucnn_prepare_features(prediction_occ) # Prediction features


## -----------------------------------------------------------------------------
data("training_labels")

labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features_train) # Training labels

## -----------------------------------------------------------------------------
res_1 <- iucnn_train_model(x = features_train,
                           lab = labels_train, 
                           path_to_output = "iucnn_model_1")

## -----------------------------------------------------------------------------
summary(res_1)
plot(res_1)

## -----------------------------------------------------------------------------
predictions <- iucnn_predict_status(x = features_predict, 
                                    model = res_1)

plot(predictions)

## ---- eval = FALSE------------------------------------------------------------
#  features_train2 <- iucnn_prepare_features(training_occ,
#                                            type = c("geographic",
#                                                     "climate",
#                                                     "humanfootprint"))

## ---- eval = FALSE------------------------------------------------------------
#  clim_features <- iucnn_climate_features(x = training_occ,
#                                          type = "selected")
#  
#  clim_features2 <- iucnn_climate_features(x = training_occ,
#                                           type = "all")

## ---- eval = FALSE------------------------------------------------------------
#  feat <- data.frame(species = c("Adansonia digitata", "Ceiba pentandra"),
#   max_plant_size_m = c(25, 50),
#   africa = c(1,1),
#   south_america = c(0,1),
#   fraction_of_records_in_protected_area = c(25, 75))

## ---- eval = FALSE------------------------------------------------------------
#  labels_train <- iucnn_prepare_labels(training_labels,
#                                       y = features,
#                                       level = "broad")

## -----------------------------------------------------------------------------
res_2 <- iucnn_train_model(x = features_train,
 lab = labels_train, 
 dropout_rate = 0.3,
 path_to_output= "iucnn_model_2",
 n_layers = "60",
 use_bias = FALSE,
 act_f = "sigmoid")

## ---- eval = FALSE------------------------------------------------------------
#  res_3 <- iucnn_train_model(x = features_train,
#   lab = labels_train,
#   path_to_output = "iucnn_model_3",
#   mode = 'bnn-class')

## -----------------------------------------------------------------------------
res_4 <- iucnn_train_model(x = features_train,
 lab = labels_train, 
 path_to_output = "iucnn_model_4",
 mode = 'nn-reg',
 rescale_features = TRUE)

## ---- eval = TRUE-------------------------------------------------------------
fi <- iucnn_feature_importance(x = res_1)
plot(fi)

## ---- eval = FALSE------------------------------------------------------------
#  modeltest_results <- iucnn_modeltest(features,
#   labels,
#   dropout_rate = c(0.0,0.1,0.3),
#   n_layers = c('30','40_20','50_30_10'))

## ---- eval = FALSE------------------------------------------------------------
#  best_m <- iucnn_best_model(modeltest_results, criterion='val_acc')

## ---- eval = FALSE------------------------------------------------------------
#  # Train the best model on all training data for prediction
#  m_prod <- iucnn_train_model(train_feat,
#                        train_lab,
#                        production_model = m_best,
#                        overwrite = TRUE)

## ---- eval = FALSE------------------------------------------------------------
#  # Predict RL categories for target species
#  pred <- iucnn_predict_status(pred_feat,
#                        m_prod)
#  plot(pred)

## -----------------------------------------------------------------------------
pred_2 <- iucnn_predict_status(x = features_predict, 
 target_acc = 0.7,
 model = res_2)
plot(pred_2)

## ---- eval = FALSE------------------------------------------------------------
#  pred_3 <- iucnn_predict_status(x = features_predict,
#   model = res_2,
#   return_IUCN = FALSE)

## -----------------------------------------------------------------------------
# preapre custom raster, you can split this step if training and test occurrences have the same extent
library(terra)
data("training_occ")
data("prediction_occ")

# find the minimum latitude and longitude values for the extent of the raster
min_lon <- min(c(min(training_occ$decimallongitude), 
                 min(prediction_occ$decimallongitude)))
max_lon <- max(c(max(training_occ$decimallongitude), 
                 max(prediction_occ$decimallongitude)))
min_lat <- min(c(min(training_occ$decimallatitude), 
                 min(prediction_occ$decimallatitude)))
max_lat <- max(c(max(training_occ$decimallatitude), 
                 max(prediction_occ$decimallatitude)))
## set the coordinate reference system
ras <- rast(crs = "+proj=longlat +datum=WGS84")
## set raster extent
ext(ras) <- c(min_lon,
              max_lon, 
              min_lat, 
              max_lat)
## set raster resolution
res(ras) <- 1 # the resolution in CRS units, in this case degrees lat/lon


 # Training features
cnn_features <- iucnn_cnn_features(x = training_occ, 
                                   y = ras)

# Prediction features
cnn_features_predict <- iucnn_cnn_features(x = prediction_occ,
                                           y = ras)
 # Training labels
cnn_labels <- iucnn_prepare_labels(x = training_labels,
                                   y = cnn_features)


## ---- eval = FALSE------------------------------------------------------------
#  trained_model <- iucnn_cnn_train(cnn_features,
#                                  cnn_labels,
#                                  overwrite = TRUE,
#                                  dropout_rate = 0.1,
#                                  optimize_for = 'accuracy')
#  
#  plot(trained_model)
#  summary(trained_model)

## ---- eval = FALSE------------------------------------------------------------
#  pred <- iucnn_predict_status(cnn_features_predict,
#                               trained_model,
#                               target_acc = 0.0
#                               )

