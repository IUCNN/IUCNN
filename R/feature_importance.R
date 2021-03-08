#' Evaluate relative importance of training features
#'
#' Uses a model generated with \code{\link{train_iucnn}} to evaluate how much each feature or
#' group of features contributes to the accuracy of the test set predictions. The function
#' implements the concept of permutation feature importance, in which the values in a given
#' feature column of the test set are shuffled randomely among all samples. Then the feature
#' data manipulated in this manner are used to predict labels for the test set and the accuracy
#' is compared to that of the original feature data. The difference (delta accuracy) can be
#' interpreted as a measure of how important a given feature or group of features is for the
#' trained NN to make accuracte predictions.
#'
#' By default this function groups the features into geographic, climatic, biome, and human
#' footprint features and determines the importance of each of these blocks of features. The
#' feature blocks can be manually defined using the feature_blocks argument.
#'
#'@param iucnn_model iucnn_model object, as produced as output when running \code{\link{train_iucnn}}
#'@param n_permutations an integer. Defines how many iterations of shuffling feature values and
#'predicting the resulting accuracy are being executed. The mean and standard deviation of the
#'delta accuracy are being summarized from these permutations.
#'@param predictions_outdir a character string. The path to the output folder where feature importance
#'dataframe will be stored.
#'@param feature_blocks a list. Default behaviour is to group the features into geographic, climatic,
#'biome, and human footprint features. Provide custom list of feature names or indices to define other
#'feature blocks, e.g. \code{feature_blocks = list(block1 = c(1,2,3,4),block2 = c(5,6,7,8))}. If feature
#'indices are provided as in this example, turn provide_indices flag to TRUE.
#'@param provide_indices logical. Set to TRUE if custom \code{feature_blocks} are provided as indices. Default is FALSE.
#'@param verbose logical. Set to TRUE to print screen output while calculating feature importance. Default is FALSE.
#'@param unlink_features_within_block logical. If TRUE, the features within each defined block arre shuffled independently.
#'If FALSE, each feature column within a block is resorted in the same manner. Default is TRUE
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")} for a
#'tutorial on how to run IUCNN.
#'
#'@return a data.frame with the relative importance of each feature block (see delta_acc_mean column).
#'
#' @examples
#'\dontrun{
#'data("training_occ")
#'data("training_labels")
#'
#'train_feat <- prep_features(training_occ)
#'labels_train <- prep_labels(training_labels,level = 'detail')
#'
#'train_output <- train_iucnn(x = train_feat,
#'                           labels = labels_train,
#'                           patience = 10)
#'
#'
#'feature_importance_default <- feature_importance(iucnn_model = train_output)
#'feature_importance_custom = feature_importance(iucnn_model = train_output, feature_blocks = list(block1 = c(1,2,3,4),block2 = c(5,6,7,8)), provide_indices = TRUE)
#'}
#'
#' @export
#' @importFrom reticulate import source_python
#' @importFrom checkmate assert_class assert_numeric assert_character assert_logical

feature_importance <- function(iucnn_model,
                               n_permutations=100,
                               predictions_outdir ='',
                               feature_blocks = list(),
                               provide_indices = FALSE,
                               verbose = FALSE,
                               unlink_features_within_block = TRUE){
  # assertions
  assert_class(iucnn_model, "iucnn_model")
  assert_numeric(n_permutations)
  assert_character(predictions_outdir)
  assert_logical(verbose)
  assert_class(feature_blocks, "list")
  assert_logical(provide_indices)
  assert_logical(unlink_features_within_block)

  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
  if (length(feature_blocks) == 0){
    ffb <- list(
      geographic = c("tot_occ","uni_occ","mean_lat","mean_lon","lat_range","lon_range","lat_hemisphere","eoo","aoo"),
      human_footprint = c("humanfootprint_1993_1","humanfootprint_1993_2","humanfootprint_1993_3","humanfootprint_1993_4",
                          "humanfootprint_2009_1","humanfootprint_2009_2","humanfootprint_2009_3","humanfootprint_2009_4"),
      climate = c("bio1","bio4","bio11","bio12","bio15","bio17","range_bio1",
                  "range_bio4","range_bio11","range_bio12","range_bio15","range_bio17"),
      biomes = c("1","2","7","10","13","3","4","5","6","11","98","8","12","9","14","99")
    )

  }else{
    if (provide_indices == TRUE){
      i = 0
      ffb= NULL
      for (block in feature_blocks){
        i = i + 1
        selected_features = iucnn_model$input_data$feature_names[as.integer(block)]
        block_name = paste(selected_features,collapse = ',')
        ffb[[block_name]] = selected_features
      }
    }else{
      ffb= feature_blocks
    }
  }

  all_selected_feature_names = c()
  feature_block_indices = ffb
  for (i in names(ffb)){
    feature_names = ffb[i][[1]]
    feature_indices = c()
    for (fname in feature_names){
      all_selected_feature_names = c(all_selected_feature_names, fname)
      findex = which(iucnn_model$input_data$feature_names == fname)
      feature_indices = c(feature_indices, as.integer(findex-1)) #-1 is necessary because of indexing discrepancy between python and r
    }
    feature_block_indices[i] = list(feature_indices)
  }

  # treat all features that are not part of a defined feature block as an individual block
  remaining_features = setdiff(iucnn_model$input_data$feature_names,
                               all_selected_feature_names)
  for (fname in remaining_features){
    findex = which(iucnn_model$input_data$feature_names == fname)
    feature_block_indices[fname] = as.integer(findex-1)
  }
  if (iucnn_model$model == 'bnn-class'){
    # source python function
    bn <- import("np_bnn")
    feature_importance_out = bn$feature_importance(iucnn_model$input_data$test_data,
                                                   weights_pkl = iucnn_model$trained_model_path,
                                                   true_labels = iucnn_model$input_data$test_labels,
                                                   fname_stem = iucnn_model$input_data$file_name,
                                                   feature_names = iucnn_model$input_data$feature_names,
                                                   n_permutations = as.integer(n_permutations),
                                                   predictions_outdir = predictions_outdir,
                                                   feature_blocks = feature_block_indices,
                                                   unlink_features_within_block = unlink_features_within_block)

  }else{
    reticulate::source_python(system.file("python", "IUCNN_feature_importance.py", package = "IUCNN"))
    feature_importance_out = feature_importance_nn(input_features = iucnn_model$input_data$test_data,
                                                   true_labels = iucnn_model$input_data$test_labels,
                                                   model_dir = iucnn_model$trained_model_path,
                                                   iucnn_mode = iucnn_model$model,
                                                   feature_names = iucnn_model$input_data$feature_names,
                                                   rescale_factor = iucnn_model$label_rescaling_factor,
                                                   min_max_label = iucnn_model$min_max_label,
                                                   stretch_factor_rescaled_labels = iucnn_model$label_stretch_factor,
                                                   verbose=verbose,
                                                   n_permutations = as.integer(n_permutations),
                                                   feature_blocks = feature_block_indices,
                                                   predictions_outdir = predictions_outdir,
                                                   unlink_features_within_block = unlink_features_within_block)
  }
  return(feature_importance_out)
}
