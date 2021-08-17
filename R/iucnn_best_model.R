#' Select the Best Model After Model-testing

#'
#' Uses a data-frame of model-testing results generated with
#' \code{\link{iucnn_modeltest}} as input, and finds the best model
#' based on the chosen criterion.
#'
#'
#'@param x a data.frame of model-testing results as produced
#'by \code{\link{iucnn_modeltest}}.
#'@param criterion name the criterion to rank models by (default="val_acc").
#'Valid options are
#'"val_acc","val_loss","weighted_error", or "total_class_matches"
#'(see details below):
#'- val_acc: highest validation accuracy
#'- val_loss: lowest validation loss
#'- weighted_error: lowest weighted error, e.g. an LC species misclassified as
#'                  CR has a weighted error of 4-0 = 4, while an LC species
#'                  misclassified as NT has a weighted error of 1-0 = 1.
#'                  These error scores are summed across all validation
#'                  predictions
#'- total_class_matches: picks the model that best reproduces the class
#'                       distribution in the validation data. When picking
#'                       this criterion it is not considered whether or not
#'                       individual instances are predicted correctly, but
#'                       instead it only looks at the overall class distribution
#'                       in the predicted data.
#'
#'@param require_dropout logical (default=FALSE). If set to TRUE, the best model
#'that contains a dropout rate of > 0 will be picked, even if other non-dropout
#'models scored higher given the chosen criterion. Dropout models are required
#'for certain functionalities within IUCNN, such as e.g. choosing a target
#'accuracy when using predict_iucnn.
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")}
#'for a tutorial on how to run IUCNN.
#'
#'@return outputs an \code{iucnn_model} object containing all
#'information about the best model.
#'
#' @examples
#'\dontrun{
#'# Model-testing
#'logfile = paste0("model_testing_results.txt")
#'model_testing_results = modeltest_iucnn(features,
#'                                        labels,
#'                                        logfile,
#'                                        model_outpath = 'iucnn_modeltest',
#'                                        mode = 'nn-class',
#'                                        seed = 1234,
#'                                        dropout_rate = c(0.0,0.1,0.3),
#'                                        n_layers = c('30','40_20','50_30_10'),
#'                                        cv_fold = 5,
#'                                        init_logfile = TRUE)
#'
#'# Selecting best model based on chosen criterion
#'best_iucnn_model = bestmodel_iucnn(model_testing_results,
#'                                    criterion = 'val_acc',
#'                                    require_dropout = TRUE)
#'}
#'
#'
#' @export

iucnn_best_model<- function(x,
                            criterion = "val_acc",
                            require_dropout = FALSE) {
<<<<<<< HEAD:R/iucnn_best_model.R
  ranked_models <- rank_models(x,
                               rank_by = criterion)

=======

  if(criterion == "val_loss" & length(unique(x$mode))>1){
stop("The chosen criterion val_loss can't be used to compare across
  different model types (e.g. nn-class and nn-reg). Choose different criterion or
  provide modeltesting results that are restricted to only one mode.")
  }

  ranked_models <- rank_models(x, rank_by = criterion)
>>>>>>> 0dc3a45afd1a000a6277834d895a180dac8ea6b5:R/bestmodel_iucnn.R
  if (require_dropout) {
    best_model <- ranked_models[ranked_models$dropout_rate > 0, ][1, ]
  } else {
    best_model <- ranked_models[1, ]
  }

  cat("Best model:\n")
  cat("", sprintf("%s: %s\n", names(best_model), best_model))
  cat("\n")

  iucnn_model <-  readRDS(best_model$model_outpath)
  return(iucnn_model)
}
