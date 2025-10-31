#' Get partial dependence probabilities for IUCN Categories
#'
#'@param x iucnn_model object, as produced as output
#'when running \code{\link{iucnn_train_model}}
#'@param dropout_reps integer, (default = 100). The number of how often the
#'predictions are to be repeated (only for dropout models). A value of 100 is
#'recommended to capture the stochasticity of the predictions, lower values
#'speed up the prediction time.
#'@param feature_blocks a list of features for which the PDP should be obtained.
#'One hot encoded features should be grouped (see example)
#'@param include_all_features logical (default = FALSE). IF FALSE, PDPs will be
#'only obtained for the features in the feature_blocks. If TRUE, PDPs will be
#'calculated for all feature.
#'@param provide_indices logical. Set to TRUE if custom \code{feature_blocks}
#'are provided as indices. Default is FALSE.
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")}
#'  for a tutorial on how to run IUCNN and \code{\link{plot.iucnn_pdp}} for plotting
#'  options.
#'
#'@return A list named according to the feature block with the gradient of the
#'feature and the partial dependence probabilities for the IUCN category.
#'When the model has been trained with dropout, the output includes a
#'95% prediction interval.
#'
#' @examples
#'\dontrun{
#'data("training_occ")
#'data("training_labels")
#'
#'train_feat <- iucnn_prepare_features(training_occ, type = "geographic")
#'
#'# create fake one-hot encoded categorical feature
#'train_feat$yellow <- 0
#'train_feat$yellow[1:300] <- 1
#'train_feat$blue <- 0
#'train_feat$blue[301:600] <- 1
#'train_feat$red <- 0
#'train_feat$red[601:nrow(train_feat)] <- 1
#'labels_train <- iucnn_prepare_labels(training_labels, train_feat,
#'                                     level = 'detail')
#'
#'train_output <- iucnn_train_model(x = train_feat,
#'                                  lab = labels_train,
#'                                  patience = 10,
#'                                  overwrite = TRUE)
#'
#'
#'feature_blocks <- list(color = c('yellow', 'blue', 'red'), aoo = 'aoo')
#'pdp <- iucnn_get_pdp(x = train_output,
#'                     dropout_reps = 10,
#'                     feature_blocks = feature_blocks)
#'}
#'
#'# plot partial dependence probabilities
#'plot(pdp)
#'
#' @export
#' @importFrom reticulate import source_python
#' @importFrom checkmate assert_class assert_numeric assert_character
#'   assert_logical
#'
iucnn_get_pdp <- function(x,
                          dropout_reps = 100,
                          feature_blocks = list(),
                          include_all_features = FALSE,
                          provide_indices = FALSE){

  if (!any(file.exists(x$trained_model_path))) {
    stop("Model path doesn't exists.
         Please check if you saved it in a temporary directory.")
  }

  # assertions
  assert_class(x, "iucnn_model")
  assert_numeric(dropout_reps)
  assert_class(feature_blocks, "list")
  assert_logical(include_all_features)
  assert_logical(provide_indices)

  dropout_reps <- as.integer(dropout_reps)

  # features for which to obtain PDP
  fb <- make_feature_block(x = x,
                           feature_blocks = feature_blocks,
                           include_all_features = include_all_features,
                           provide_indices = provide_indices)
  feature_block_indices <- fb$feature_block_indices
  num_feature_blocks <- length(feature_block_indices)



  reticulate::source_python(system.file("python", "IUCNN_pdp.py",
                                        package = "IUCNN"))
  if (x$model == 'bnn-class') {
    # source python function
    reticulate::source_python(system.file("python",
                                          "IUCNN_helper_functions.py",
                                          package = "IUCNN"))
  }


  model_dir <- x$trained_model_path
  iucnn_mode <- x$model
  dropout <- x$mc_dropout
  rescale_factor <- x$label_rescaling_factor
  min_max_label <- as.integer(x$min_max_label_rescaled)
  stretch_factor_rescaled_labels <- x$label_stretch_factor

  if (x$cv_fold == 1) {
    data_pdp <- rbind(x$input_data$data, x$input_data$test_data)
  }
  else {
    data_pdp <- x$input_data$data
  }

  pdp <- vector(mode = "list", length = num_feature_blocks)
  names(pdp) <- names(feature_block_indices)
  for (i in 1:num_feature_blocks) {
    pdp[[i]] <- iucnn_pdp(input_features = data_pdp,
                          focal_features = feature_block_indices[[i]],
                          model_dir = model_dir,
                          iucnn_mode = iucnn_mode,
                          cv_fold = as.integer(x$cv_fold),
                          dropout = dropout,
                          dropout_reps = dropout_reps,
                          rescale_factor = rescale_factor,
                          min_max_label = min_max_label,
                          stretch_factor_rescaled_labels = stretch_factor_rescaled_labels)

    if (length(feature_block_indices[[i]]) == 1) {
      colnames(pdp[[i]][[1]]) <- names(feature_block_indices)[i]
    }
    else {
      df <- data.frame(A = feature_blocks[[i]])
      colnames(df) <- names(feature_block_indices)[i]
      pdp[[i]][[1]] <- df
    }
  }

  class(pdp) <- "iucnn_pdp"
  return(pdp)
}
