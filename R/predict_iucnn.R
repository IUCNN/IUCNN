#' Predict IUCN Categories from Features
#'
#' Uses a model generated with \code{\link{train_iucnn}}
#' to predict the IUCN status of
#' Not Evaluated or Data Deficient species based on features,
#' for instance generated
#' from species occurrence records with \code{\link{ft_geo}},
#' \code{\link{ft_clim}}, and \code{\link{ft_biom}}.
#' The same features in the same order must be
#' used for training and fitting.
#'
#'
#'@param x a data.set, containing a column "species" with the species names, and
#'subsequent columns with different features,
#'in the same order as used for \code{\link{train_iucnn}}
#'@param model the information on the NN model returned by \code{\link{train_iucnn}}
#'@param target_acc numerical, 0-1. The target accuracy of the overall model.
#' Species that cannot be classified with
#'enough certainty to reach this accuracy are classified as DD (Data Deficient).
#'@param return_raw logical. Should the probabilities for the labels be returned?
#'Default is FALSE.
#'Note that the probabilities are the direct output of the
#'SoftMax function in the output layer
#'of the neural network and might be an unreliable measure of statistical support for
#'the result of the classification (e.g. https://arxiv.org/abs/2005.04987).
#'@param return_IUCN logical. If TRUE the predicted labels are translated
#' into the original labels.
#'If FALSE numeric labels as used by the model are returned
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")} for a
#'tutorial on how to run IUCNN.
#'
#'@return a vector with the predicted labels for the input species.
#'
#' @examples
#'\dontrun{
#'data("training_occ") #geographic occurrences of species with IUCN assessment
#'data("training_labels")# the corresponding IUCN assessments
#'data("prediction_occ") #occurrences from Not Evaluated species to prdict
#'
#'# 1. Feature and label preparation
#'features <- prep_features(training_occ) # Training features
#'labels_train <- prep_labels(training_labels) # Training labels
#'features_predict <- prep_features(prediction_occ) # Prediction features
#'
#'# 2. Model training
#'m1 <- train_iucnn(x = features, lab = labels_train)
#'
#'# 3. Prediction
#'predict_iucnn(x = features_predict,
#'              model = m1)
#'}
#'
#'
#' @export
#' @importFrom reticulate source_python
#' @importFrom magrittr %>%
#' @importFrom dplyr select
#' @importFrom stats complete.cases

predict_iucnn <- function(x,
                          model,
                          target_acc = 0.0,
                          return_IUCN = TRUE){

  # assertions
  assert_class(x, classes = "data.frame")
  assert_class(model, classes = "iucnn_model")


  if (model$cv_fold > 1){
    stop("Provided model consists of multiple cross-validation (CV) folds.\n
          CV models are only used for model evaluation in IUCNN.
          Retrain your chosen model without using CV.
          To do this you can use the train_iucnn function and simply
          provide your CV model under the \'production_model\' flag.")
  }




  # check that the same features are in training and prediction
  test1 <- all(names(x)[-1] %in% model$input_data$feature_names)
  if(!test1){
    mis <- names(x)[-1][!names(x)[-1] %in% model$input_data$feature_names]
    stop("Feature mismatch, missing in training features: \n",
         paste0(mis, collapse = ", "))
  }

  test2 <- all(model$input_data$feature_names %in% names(x))
  if(!test2){
    mis <- model$input_data$feature_names[!model$input_data$feature_names %in% names(x)]
    stop("Feature mismatch, missing in prediction features: \n",
         paste0(mis, collapse = ", "))
  }

  if (target_acc == 0){
    confidence_threshold <- NULL
  }else{
    acc_thres_tbl <- model$accthres_tbl
    if (class(acc_thres_tbl)[1] == "matrix"){
     confidence_threshold <- acc_thres_tbl[min(which(acc_thres_tbl[,2] >
                                                       target_acc)),][1]
    }else{
     stop('Table with accuracy thresholds required when choosing target_acc > 0.
          This is only available for models where
          \'mc_dropout=TRUE\' and \'dropout_rate\' > 0.')
   }
  }

  data_out <- process_iucnn_input(x,mode = mode, outpath = '.',
                                  write_data_files = FALSE)

  dataset <- data_out[[1]]
  instance_names <- data_out[[3]]

  message("Predicting conservation status")
  pred_out <- NULL
  pred_out$names <- NULL

  if(model$model == 'bnn-class'){
    postpr <- bnn_predict(features = as.matrix(dataset),
                          instance_id = as.matrix(instance_names),
                          model_path = model$trained_model_path,
                          post_cutoff = confidence_threshold,
                          filename = 'prediction',
                          post_summary_mode = 0
                          )
    not_nan_boolean <- complete.cases(postpr$post_prob_predictions)
    predictions_tmp <- apply(postpr$post_prob_predictions[not_nan_boolean,],
                             1,
                             which.max)-1
    predictions <- rep(NA, dim(postpr$post_prob_predictions)[1])
    predictions[not_nan_boolean] <- predictions_tmp

    pred_out$raw_predictions <- postpr$post_prob_predictions
    pred_out$class_predictions <- predictions

  }else{
    # source python function
    reticulate::source_python(system.file("python", "IUCNN_predict.py",
                                          package = "IUCNN"))

    # run predict function
    pred_out <- iucnn_predict(
                   feature_set = as.matrix(dataset),
                   model_dir = model$trained_model_path,
                   iucnn_mode = model$model,
                   dropout = model$mc_dropout,
                   dropout_reps = 100,
                   confidence_threshold = confidence_threshold,
                   rescale_factor = model$label_rescaling_factor,
                   min_max_label = model$min_max_label_rescaled,
                   stretch_factor_rescaled_labels = model$label_stretch_factor)
  }

  # Translate prediction to original labels
  if(return_IUCN){
    lu <- model$input_data$lookup.labels
    names(lu) <- model$input_data$lookup.lab.num.z

    predictions <- lu[pred_out$class_predictions+1]
    names(predictions) <- NULL
    pred_out$class_predictions <- predictions
  }

  pred_out$names <- instance_names
  class(pred_out) <- "iucnn_predictions"

  return(pred_out)

}


