#' Predict IUCN Categories from Features
#'
#' Uses a model generated with \code{\link{iucnn_train_model}}
#' to predict the IUCN status of
#' Not Evaluated or Data Deficient species based on features, generated
#' from species occurrence records with \code{\link{iucnn_prepare_features}}.
#' These features should be of the same type as those used for training the
#' model.
#'
#'@param x a data.set, containing a column "species" with the species names, and
#'subsequent columns with different features,
#'in the same order as used for \code{\link{iucnn_train_model}}
#'@param model the information on the NN model returned by
#'\code{\link{iucnn_train_model}}
#'@param target_acc numerical, 0-1. The target accuracy of the overall model.
#' Species that cannot be classified with
#'@param dropout_reps integer, (default = 100). The number of how often the
#'predictions are to be repeated (only for dropout models). A value of 100 is
#'recommended to capture the stochasticity of the predictions, lower values
#'speed up the prediction time.
#'@param return_IUCN logical. If TRUE the predicted labels are translated
#' into the original labels.
#'If FALSE numeric labels as used by the model are returned
#'@param return_raw logical. If TRUE, the raw predictions of the model will be
#'returned, which in case of MC-dropout and bnn-class models includes the class
#'predictions across all dropout prediction reps (or MCMC reps for bnn-class).
#'Note that setting this to TRUE will result in large output objects that can
#'fill up the memory allocated for R and cause the program to crash.
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")} for a
#'tutorial on how to run IUCNN.
#'
#'@return outputs an \code{iucnn_predictions} object containing the predicted
#'labels for the input species.
#'
#' @examples
#'\dontrun{
#'data("training_occ") #geographic occurrences of species with IUCN assessment
#'data("training_labels")# the corresponding IUCN assessments
#'data("prediction_occ") #occurrences from Not Evaluated species to prdict
#'
#'# 1. Feature and label preparation
#'features <- iucnn_prepare_features(training_occ, type = "geographic") # Training features
#'labels_train <- iucnn_prepare_labels(training_labels, features) # Training labels
#'features_predict <- iucnn_prepare_features(prediction_occ,
#'                                           type = "geographic") # Prediction features
#'
#'# 2. Model training
#'m1 <- iucnn_train_model(x = features, lab = labels_train)
#'
#'# 3. Prediction
#'iucnn_predict_status(x = features_predict, model = m1)
#'}
#'
#'
#' @export
#' @importFrom reticulate source_python
#' @importFrom magrittr %>%
#' @importFrom dplyr select
#' @importFrom stats complete.cases

iucnn_predict_status <- function(x,
                          model,
                          target_acc = 0.0,
                          dropout_reps = 100,
                          return_IUCN = TRUE,
                          return_raw = FALSE){

  # assertions


  assert_class(model, classes = "iucnn_model")

  if (model$cv_fold > 1) {
    stop("Provided model consists of multiple cross-validation (CV) folds.\n
          CV models are only used for model evaluation in IUCNN.
          Retrain your chosen model without using CV.
          To do this you can use the iucnn_train_model function and simply
          provide your CV model under the \'production_model\' flag.")
  }

  # only run tests for models other than cnn
  if (!model$model == 'cnn') {
    assert_class(x, classes = "data.frame")
    # check that the same features are in training and prediction
    test1 <- all(names(x)[-1] %in% model$input_data$feature_names)
    if (!test1) {
      mis <- names(x)[-1][!names(x)[-1] %in% model$input_data$feature_names]
      stop("Feature mismatch, missing in training features: \n",
           paste0(mis, collapse = ", "))
    }

    test2 <- all(model$input_data$feature_names %in% names(x))
    if (!test2) {
      mis <- model$input_data$feature_names[!model$input_data$feature_names %in% names(x)]
      stop("Feature mismatch, missing in prediction features: \n",
           paste0(mis, collapse = ", "))
    }
    data_out <- process_iucnn_input(x,
                                    mode = mode,
                                    outpath = '.',
                                    write_data_files = FALSE)

    dataset <- as.matrix(data_out[[1]])
    instance_names <- data_out[[3]]

  }else{
    dataset = x
    instance_names = names(x)
  }


  if (target_acc == 0) {
    confidence_threshold <- NULL
  }else{
    acc_thres_tbl <- model$accthres_tbl
    if (class(acc_thres_tbl)[1] == "matrix") {
     confidence_threshold <- acc_thres_tbl[min(which(acc_thres_tbl[,2] >
                                                       target_acc)),][1]
    }else{
     stop('Table with accuracy thresholds required when choosing target_acc > 0.
          This is only available for models where
          \'mc_dropout=TRUE\' and \'dropout_rate\' > 0.')
   }
  }



  message("Predicting conservation status")

  if (model$model == 'bnn-class') {
    # source python function
    reticulate::source_python(system.file("python",
                                          "IUCNN_helper_functions.py",
                                          package = "IUCNN"))
    pred_out <- predict_bnn(features = as.matrix(dataset),
                            model_path = model$trained_model_path,
                            posterior_threshold = confidence_threshold,
                            post_summary_mode = 0
                          )


  }else{
    # source python function
    reticulate::source_python(system.file("python", "IUCNN_predict.py",
                                          package = "IUCNN"))

    # run predict function
    pred_out <- iucnn_predict(
                   input_raw = dataset,
                   model_dir = model$trained_model_path,
                   iucnn_mode = model$model,
                   dropout = model$mc_dropout,
                   dropout_reps = dropout_reps,
                   confidence_threshold = confidence_threshold,
                   rescale_factor = model$label_rescaling_factor,
                   min_max_label = model$min_max_label_rescaled,
                   stretch_factor_rescaled_labels = model$label_stretch_factor)
  }

  cat_count = get_cat_count(pred_out$class_predictions,
                max_cat = length(model$input_data$lookup.lab.num.z)-1,
                include_NA = TRUE)
  pred_out$pred_cat_count = cat_count

  # Translate prediction to original labels
  if (return_IUCN) {
    lu <- model$input_data$lookup.labels
    names(lu) <- model$input_data$lookup.lab.num.z

    predictions <- lu[pred_out$class_predictions + 1]
    names(predictions) <- NULL
    pred_out$class_predictions <- predictions
  }
  if (return_raw == FALSE) {
    pred_out$raw_predictions <- NaN
  }
  pred_out$names <- instance_names
  class(pred_out) <- "iucnn_predictions"

  return(pred_out)

}


