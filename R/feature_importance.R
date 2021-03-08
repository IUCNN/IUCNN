#' Evaluate importance of training features
#'
#' @export
#' @importFrom reticulate import source_python
#' @importFrom checkmate assert_class assert_numeric assert_character assert_logical

feature_importance <- function(x,
                               n_permutations=100,
                               predictions_outdir ='',
                               verbose = FALSE,
                               feature_blocks = "auto",
                               provide_indices = FALSE,
                               unlink_features_within_block = TRUE){
  # assertions
  assert_class(x, "iucnn_model")
  assert_numeric(n_permutations)
  assert_character(predictions_outdir)
  assert_logical(verbose)
  assert_character(feature_blocks)
  assert_logical(provide_indices)
  assert_logical(unlink_features_within_block)

  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
  if (feature_blocks == 'auto'){
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
        selected_features = x$input_data$feature_names[as.integer(block)]
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
      findex = which(x$input_data$feature_names == fname)
      feature_indices = c(feature_indices, as.integer(findex-1)) #-1 is necessary because of indexing discrepancy between python and r
    }
    feature_block_indices[i] = list(feature_indices)
  }

  # treat all features that are not part of a defined feature block as an individual block
  remaining_features = setdiff(x$input_data$feature_names,
                               all_selected_feature_names)
  for (fname in remaining_features){
    findex = which(x$input_data$feature_names == fname)
    feature_block_indices[fname] = as.integer(findex-1)
  }
  if (x$model == 'bnn-class'){
    # source python function
    bn <- import("np_bnn")
    feature_importance_out = bn$feature_importance(x$input_data$test_data,
                                                   weights_pkl = x$trained_model_path,
                                                   true_labels = x$input_data$test_labels,
                                                   fname_stem = x$input_data$file_name,
                                                   feature_names = x$input_data$feature_names,
                                                   n_permutations = as.integer(n_permutations),
                                                   predictions_outdir = predictions_outdir,
                                                   feature_blocks = feature_block_indices,
                                                   unlink_features_within_block = unlink_features_within_block)

  }else{
    reticulate::source_python(system.file("python", "IUCNN_feature_importance.py", package = "IUCNN"))
    feature_importance_out = feature_importance_nn(input_features = x$input_data$test_data,
                                                   true_labels = x$input_data$test_labels,
                                                   model_dir = x$trained_model_path,
                                                   iucnn_mode = x$model,
                                                   feature_names = x$input_data$feature_names,
                                                   rescale_factor = x$label_rescaling_factor,
                                                   min_max_label = x$min_max_label,
                                                   stretch_factor_rescaled_labels = x$label_stretch_factor,
                                                   verbose=verbose,
                                                   n_permutations = as.integer(n_permutations),
                                                   feature_blocks = feature_block_indices,
                                                   predictions_outdir = predictions_outdir,
                                                   unlink_features_within_block = unlink_features_within_block)
  }
  return(feature_importance_out)
}
