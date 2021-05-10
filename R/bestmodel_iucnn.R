#' Select the best IUCNN Model after Model-Testing
#'
#' Uses a data-frame of model-testing results generated with
#' \code{\link{modeltest_iucnn}} as input, and finds the best model
#' based on the chosen criterion.
#'
#'
#'@param x a data.frame of model-testing results as produced
#'by \code{\link{modeltest_iucnn}}.
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
#'@return outputs an \code{iucnn_model} object containing all information
#'about the best model.
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

bestmodel_iucnn <- function(x,
                             criterion = "val_acc",
                             require_dropout = FALSE) {
  ranked_models <- rank_models(x, rank_by = criterion)
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
