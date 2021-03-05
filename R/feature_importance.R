#' Evaluate importance of training features
#'
#' @export
#' @import reticulate

feature_importance <- function(input_model,
                               n_permutations=100,
                               predictions_outdir='',
                               verbose=FALSE,
                               feature_blocks = "auto",
                               provide_indices = FALSE,
                               unlink_features_within_block = TRUE){

  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
  if (feature_blocks == 'auto'){
    formatted_feature_blocks = NULL
    formatted_feature_blocks$geographic = c("tot_occ","uni_occ","mean_lat","mean_lon","lat_range","lon_range","lat_hemisphere","eoo","aoo")
    formatted_feature_blocks$human_footprint = c("humanfootprint_1993_1","humanfootprint_1993_2","humanfootprint_1993_3","humanfootprint_1993_4","humanfootprint_2009_1","humanfootprint_2009_2","humanfootprint_2009_3","humanfootprint_2009_4")
    formatted_feature_blocks$climate = c("bio1","bio4","bio11","bio12","bio15","bio17","range_bio1","range_bio4","range_bio11","range_bio12","range_bio15","range_bio17")
    formatted_feature_blocks$biomes = c("1","2","7","10","13","3","4","5","6","11","98","8","12","9","14","99")
  }else{
    if (provide_indices == TRUE){
      i=0
      formatted_feature_blocks = NULL
      for (block in feature_blocks){
        i = i+1
        selected_features = input_model$input_data$feature_names[as.integer(block)]
        block_name = paste(selected_features,collapse = ',')
        formatted_feature_blocks[[block_name]] = selected_features
      }
    }else{
      formatted_feature_blocks = feature_blocks
    }
  }

  all_selected_feature_names = c()
  feature_block_indices = formatted_feature_blocks
  for (i in names(formatted_feature_blocks)){
    feature_names = formatted_feature_blocks[i][[1]]
    feature_indices = c()
    for (fname in feature_names){
      all_selected_feature_names = c(all_selected_feature_names,fname)
      findex = which(input_model$input_data$feature_names==fname)
      feature_indices = c(feature_indices,as.integer(findex-1)) #-1 is necessary because of indexing discrepancy between python and r
    }
    feature_block_indices[i] = list(feature_indices)
  }

  # treat all features that are not part of a defined feature block as an individual block
  remaining_features = setdiff(input_model$input_data$feature_names, all_selected_feature_names)
  for (fname in remaining_features){
    findex = which(input_model$input_data$feature_names==fname)
    feature_block_indices[fname] = as.integer(findex-1)
  }
  if (input_model$model == 'bnn-class'){
    # source python function
    bn <- import("np_bnn")
    feature_importance_out = bn$feature_importance(input_model$input_data$test_data,
                                                   weights_pkl=input_model$trained_model_path,
                                                   true_labels=input_model$input_data$test_labels,
                                                   fname_stem=input_model$input_data$file_name,
                                                   feature_names=input_model$input_data$feature_names,
                                                   n_permutations=as.integer(n_permutations),
                                                   predictions_outdir=predictions_outdir,
                                                   feature_blocks = feature_block_indices,
                                                   unlink_features_within_block = unlink_features_within_block)

  }else{
    reticulate::source_python(system.file("python", "IUCNN_feature_importance.py", package = "IUCNN"))
    feature_importance_out = feature_importance_nn(input_features = input_model$input_data$test_data,
                                                   true_labels = input_model$input_data$test_labels,
                                                   model_dir = input_model$trained_model_path,
                                                   iucnn_mode = input_model$model,
                                                   feature_names = input_model$input_data$feature_names,
                                                   rescale_factor = input_model$label_rescaling_factor,
                                                   min_max_label = input_model$min_max_label,
                                                   stretch_factor_rescaled_labels = input_model$label_stretch_factor,
                                                   verbose=verbose,
                                                   n_permutations=as.integer(n_permutations),
                                                   feature_blocks=feature_block_indices,
                                                   predictions_outdir=predictions_outdir,
                                                   unlink_features_within_block=unlink_features_within_block)
  }
  return(feature_importance_out)
}
