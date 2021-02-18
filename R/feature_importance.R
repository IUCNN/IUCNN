#' Determine feature importance
#' @export
#' @import reticulate

feature_importance <- function(bnn_data,
                               logger,
                               bnn_model,
                               n_permutations=100,
                               predictions_outdir='feature_importance',
                               feature_blocks = list(c(0,1,2,3,4,5,6,7),c(8,9,10),c(11,12,13,14,15,16,17,18,19,20)),
                               unlink_features_within_block = TRUE
                               ){

  # source python function
  bn <- import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if (length(feature_blocks)>0){
    i=0
    formatted_feature_blocks = list()
    for (block in feature_blocks){
      i = i+1
      formatted_feature_blocks[[i]] = as.integer(block)
    }
  }

  feature_importance_out = bn$feature_importance(bnn_data$test_data,
                                              weights_pkl=py_get_attr(logger,'_w_file'),
                                              true_labels=bnn_data$test_labels,
                                              fname_stem=bnn_data$file_name,
                                              feature_names=bnn_data$feature_names,
                                              n_permutations=as.integer(n_permutations),
                                              predictions_outdir=predictions_outdir,
                                              feature_blocks = formatted_feature_blocks,
                                              unlink_features_within_block = unlink_features_within_block,
                                              actFun = py_get_attr(bnn_model,'_act_fun'))
  return(feature_importance_out)
}



