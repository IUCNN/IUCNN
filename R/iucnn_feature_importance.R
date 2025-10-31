#' Evaluate relative importance of training features
#'
#' Uses a model generated with \code{\link{iucnn_train_model}}
#' to evaluate how much each feature or
#' group of features contributes to the accuracy of
#'  the test set predictions. The function
#' implements the concept of permutation feature importance,
#'  in which the values in a given
#' feature column of the test set are shuffled randomly
#'  among all samples. Then the feature
#' data manipulated in this manner are used to predict
#' labels for the test set and the accuracy
#' is compared to that of the original feature data.
#'  The difference (delta accuracy) can be
#' interpreted as a measure of how important a
#' given feature or group of features is for the
#' trained NN to make accurate predictions.
#'
#' By default this function groups the features
#' into geographic, climatic, biome, and human
#' footprint features and determines the importance
#' of each of these blocks of features. The
#' feature blocks can be manually defined using the feature_blocks argument.
#'
#'@param x iucnn_model object, as produced as output
#'when running \code{\link{iucnn_train_model}}
#'@param feature_blocks a list. Default behavior is to
#' group the features into geographic, climatic,
#'biome, and human footprint features. Provide custom
#'list of feature names or indices to define other
#'feature blocks. If feature
#'indices are provided as in this example, turn provide_indices flag to TRUE.
#'@param n_permutations an integer. Defines how many
#' iterations of shuffling feature values and
#'predicting the resulting accuracy are being executed.
#'The mean and standard deviation of the
#'delta accuracy are being summarized from these permutations.
#'@param provide_indices logical. Set to TRUE if custom \code{feature_blocks}
#'are provided as indices. Default is FALSE.
#'@param verbose logical. Set to TRUE to print screen output while calculating
#'feature importance. Default is FALSE.
#'@param unlink_features_within_block logical. If TRUE, the features within each
#'defined block are shuffled independently.
#'If FALSE, each feature column within a block is resorted in the same manner.
#'Default is TRUE.
#'This can also be a vector of TRUE and FALSE with the same length as the
#'feature blocks.
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")}
#'  for a tutorial on how to run IUCNN.
#'
#'@return a data.frame with the relative importance of each feature block (see
#'  delta_acc_mean column).
#'
#' @examples
#'\dontrun{
#'data("training_occ")
#'data("training_labels")
#'
#'train_feat <- iucnn_prepare_features(training_occ, type = "geographic")
#'labels_train <- iucnn_prepare_labels(training_labels, train_feat,
#'                                     level = 'detail')
#'
#'train_output <- iucnn_train_model(x = train_feat,
#'                           lab = labels_train,
#'                           patience = 10,
#'                           overwrite = TRUE)
#'
#'
#'imp_def <- iucnn_feature_importance(x = train_output)
#'imp_cust <- iucnn_feature_importance(x = train_output,
#'                               feature_blocks = list(block1 = c(1,2,3,4),
#'                                                     block2 = c(5,6,7,8)),
#'                               provide_indices = TRUE)
#'}
#'
#' @export
#' @importFrom reticulate import source_python
#' @importFrom checkmate assert_class assert_numeric assert_character
#'   assert_logical

iucnn_feature_importance <- function(x,
                                     feature_blocks = list(),
                                     n_permutations = 100,
                                     provide_indices = FALSE,
                                     verbose = FALSE,
                                     unlink_features_within_block = TRUE){

  if (!any(file.exists(x$trained_model_path))) {
    stop("Model path doesn't exists.
         Please check if you saved it in a temporary directory.")
  }

  # assertions
  assert_class(x, "iucnn_model")
  assert_class(feature_blocks, "list")
  assert_numeric(n_permutations)
  assert_logical(provide_indices)
  assert_logical(verbose)
  assert_logical(unlink_features_within_block)

  fb <- make_feature_block(x = x,
                           feature_blocks = feature_blocks,
                           include_all_features = TRUE,
                           provide_indices = provide_indices,
                           unlink_features_within_block = unlink_features_within_block)
  feature_block_indices <- fb$feature_block_indices
  unlink_features_within_block <- fb$unlink_features_within_block

  if (x$model == 'bnn-class') {
    # source python function
    bn <- import("np_bnn")
    feature_importance_out <- bn$feature_importance(x$input_data$test_data,
                                                   weights_pkl = x$trained_model_path,
                                                   true_labels = x$input_data$test_labels,
                                                   fname_stem = x$input_data$file_name,
                                                   feature_names = x$input_data$feature_names,
                                                   n_permutations = as.integer(n_permutations),
                                                   write_to_file = FALSE,
                                                   feature_blocks = feature_block_indices,
                                                   unlink_features_within_block = unlink_features_within_block)
    selected_cols <- feature_importance_out[,2:4]
  }else{
    if (is.nan(x$input_data$test_data[1])) {
      use_these_features <- x$input_data$data
      use_these_labels <- x$input_data$labels

    }else{
      use_these_features <- x$input_data$test_data
      use_these_labels <- x$input_data$test_labels
    }
    cv_k <- length(x$trained_model_path)
    d_list <- vector(mode = "list", length = cv_k)
    for (k in 1:cv_k) {
      reticulate::source_python(system.file("python", "IUCNN_feature_importance.py", package = "IUCNN"))
      feature_importance_out <- feature_importance_nn(input_features = use_these_features,
                                                      true_labels = use_these_labels,
                                                      model_dir = x$trained_model_path[k],
                                                      iucnn_mode = x$model,
                                                      feature_names = x$input_data$feature_names,
                                                      rescale_factor = x$label_rescaling_factor,
                                                      min_max_label = x$min_max_label,
                                                      stretch_factor_rescaled_labels = x$label_stretch_factor,
                                                      verbose = verbose,
                                                      n_permutations = as.integer(n_permutations),
                                                      feature_blocks = feature_block_indices,
                                                      unlink_features_within_block = unlink_features_within_block)
      d_list[[k]] <- round(data.frame(matrix(unlist(feature_importance_out),
                                             nrow = length(feature_importance_out),
                                             byrow = TRUE)), 3)
    }
    # Mean across cross-validation schemes (cv_k > 1) or a single test_fraction
    d <- Reduce('+', d_list) / cv_k

    d['feature_block'] <- names(feature_importance_out)
    selected_cols <- d[c('feature_block','X1','X2')]
  }
  names(selected_cols) <- c('feature_block',
                            'feat_imp_mean',
                            'feat_imp_std')

  class(selected_cols) <- "iucnn_featureimportance"

  return(selected_cols)
}


make_feature_block <- function(x,
                               feature_blocks = list(),
                               include_all_features = TRUE,
                               provide_indices = FALSE,
                               unlink_features_within_block = TRUE) {
  if (length(feature_blocks) == 0) {
    ffb <- list(
      geographic = c("tot_occ",
                     "uni_occ",
                     "mean_lat",
                     "mean_lon",
                     "lat_range",
                     "lon_range",
                     "lat_hemisphere",
                     "eoo",
                     "aoo"),
      human_footprint = c("humanfootprint_1993_1",
                          "humanfootprint_1993_2",
                          "humanfootprint_1993_3",
                          "humanfootprint_1993_4",
                          "humanfootprint_2009_1",
                          "humanfootprint_2009_2",
                          "humanfootprint_2009_3",
                          "humanfootprint_2009_4"),
      climate = c("bio1",
                  "bio4",
                  "bio11",
                  "bio12",
                  "bio15",
                  "bio17",
                  "range_bio1",
                  "range_bio4",
                  "range_bio11",
                  "range_bio12",
                  "range_bio15",
                  "range_bio17"),
      biomes = c("biome_1",
                 "biome_2",
                 "biome_7",
                 "biome_10",
                 "biome_13",
                 "biome_3",
                 "biome_4",
                 "biome_5",
                 "biome_6",
                 "biome_11",
                 "biome_98",
                 "biome_8",
                 "biome_12",
                 "biome_9",
                 "biome_14",
                 "biome_99")
    )
  }
  else {
    if (provide_indices) {
      i <- 0
      ffb <- NULL
      for (block in feature_blocks) {
        i <- i + 1
        selected_features <- x$input_data$feature_names[as.integer(block)]
        block_name <- paste(selected_features, collapse = ',')
        ffb[[block_name]] <- selected_features
      }
    }
    else {
      if (is.null(names(feature_blocks))) {
        names(feature_blocks) <- feature_blocks
      }
      ffb <- feature_blocks
      if ('species' %in% names(ffb)) {
        ffb['species'] <- NULL
      }
    }
  }

  if (length(unlink_features_within_block) > 1 &&
      (length(unlink_features_within_block) != length(ffb))) {
    stop("Length of feature block and unlink_features_within_block differ")
  }

  all_selected_feature_names <- c()
  feature_block_indices <- ffb
  for (i in names(ffb)) {
    feature_names <- ffb[i][[1]]
    feature_indices <- c()
    for (fname in feature_names) {
      all_selected_feature_names <- c(all_selected_feature_names, fname)
      findex <- which(x$input_data$feature_names == fname)
      feature_indices <- c(feature_indices, as.integer(findex - 1))
      # -1 is necessary because of indexing discrepancy between python and r
    }
    feature_block_indices[i] <- list(feature_indices)
  }

  if (include_all_features) {
    # treat all features that are not part of a defined feature block as an individual block
    remaining_features <- setdiff(x$input_data$feature_names,
                                  all_selected_feature_names)
    for (fname in remaining_features) {
      findex <- which(x$input_data$feature_names == fname)
      feature_block_indices[fname] <- as.integer(findex - 1)
      if (length(unlink_features_within_block) > 1) {
        unlink_features_within_block <- c(unlink_features_within_block, TRUE)
      }
    }
  }

  out <- vector(mode = "list", length = 2)
  names(out) <- c("feature_block_indices", "unlink_features_within_block")
  out[[1]] <- feature_block_indices
  out[[2]] <- unlink_features_within_block

  return(out)
}
