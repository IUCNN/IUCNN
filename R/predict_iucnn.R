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
#'@param verbose logical, if TRUE generate additional output.
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
                          validation_model=NULL,
                          target_acc = 0.0,
                          verbose = 0,
                          return_raw = FALSE,
                          return_IUCN = TRUE){

  # assertions
  assert_class(x, classes = "data.frame")


  # check that the same features are in training and prediction
  test1 <- all(names(x)[-1] %in% model$input_data$feature_names)
  if(!test1){
    mis <- names(x)[-1][!names(x)[-1] %in% model$input_data$feature_names]
    stop("Feature mismatch, missing in training features: \n", paste0(mis, collapse = ", "))
  }

  test2 <- all(model$input_data$feature_names %in% names(x))
  if(!test2){
    mis <- model$input_data$feature_names[!model$input_data$feature_names %in% names(x)]
    stop("Feature mismatch, missing in prediction features: \n", paste0(mis, collapse = ", "))
  }

  if (target_acc == 0){
    confidence_threshold = NULL
  }else{
   if (class(validation_model) == "iucnn_model"){
     if (validation_model$dropout == FALSE){
       stop('target_acc argument can only be used for models trained with mc_dropout. Retrain your model and specify a dropout rate > 0 to use this option')
     }else{
       confidence_threshold = validation_model$accthres_tbl[min(which(validation_model$accthres_tbl[,2] > target_acc)),][1]
     }
   }else{
     stop('When choosing target_acc > 0 you need to provide the validation model (output of the evaluate_model() function)')
   }
  }


  data_out = process_iucnn_input(x,mode = mode, outpath = '.', write_data_files = FALSE)

  dataset = data_out[[1]]
  instance_names = data_out[[3]]

  message("Predicting conservation status")

  pred_out = NULL
  pred_out$names = instance_names

  if(model$model == 'bnn-class'){
    postpr <- bnn_predict(features = as.matrix(dataset),
                          instance_id = as.matrix(instance_names),
                          model_path = model$trained_model_path,
                          target_acc = target_acc,
                          filename = 'prediction',
                          post_summary_mode = 1
                          )

    if (return_raw==TRUE){
      pred_out$predictions = postpr$post_prob_predictions
      return(pred_out)
    }else{
      not_nan_boolean <- complete.cases(postpr$post_prob_predictions)
      predictions_tmp <- apply(postpr$post_prob_predictions[not_nan_boolean,],1,which.max)-1
      predictions <- rep(NA, dim(postpr$post_prob_predictions)[1])
      predictions[not_nan_boolean] <- predictions_tmp
      pred_out$predictions = predictions
      return(pred_out)
    }


  }else{
    # source python function
    reticulate::source_python(system.file("python", "IUCNN_predict.py", package = "IUCNN"))

    # run predict function
    out <- iucnn_predict(feature_set = as.matrix(dataset),
                         model_dir = model$trained_model_path,
                         verbose = verbose,
                         iucnn_mode = model$model,
                         dropout = validation_model$mc_dropout,
                         dropout_reps = 100,
                         confidence_threshold = confidence_threshold,
                         rescale_labels_boolean = model$rescale_labels_boolean,
                         rescale_factor = model$label_rescaling_factor,
                         min_max_label = model$min_max_label_rescaled,
                         stretch_factor_rescaled_labels = model$label_stretch_factor)
    if (return_raw==TRUE){
      pred_out$predictions = out
      return(pred_out)
    }else{
      if (model$model == 'nn-reg'){
        predictions <- round(out)
      }else{
        not_nan_boolean <- complete.cases(out)
        predictions_tmp <- apply(out[not_nan_boolean,],1,which.max)-1
        predictions <- rep(NA, dim(out)[1])
        predictions[not_nan_boolean] <- predictions_tmp
      }

      # Translate prediction to original labels
      if(return_IUCN){
        lu <- model$input_data$lookup.labels
        names(lu) <- model$input_data$lookup.lab.num.z

        predictions <- lu[predictions+1]
        names(predictions) <- NULL
      }
      pred_out$predictions = predictions
      return(pred_out)
    }
  }
}


