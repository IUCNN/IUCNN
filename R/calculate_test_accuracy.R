#' Calculate test accuracy
#' @export
#' @import reticulate
#'
#'
calculate_accuracy <- function(bnn_data,
                               logger,
                               bnn_model,
                               data = 'test',
                               post_summary_mode=0){

  # source python function
  bn <- import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
  if (data == 'test'){
    features = bnn_data$test_data
    labels = bnn_data$test_labels
    instance_id = bnn_data$id_test_data
  }else if(data=='train'){
    features = bnn_data$data
    labels = bnn_data$labels
    instance_id = bnn_data$id_data
  }
  post_pr = bn$predictBNN(features,
                          pickle_file=py_get_attr(logger,'_w_file'),
                          test_labels=labels,
                          instance_id=instance_id,
                          fname=bnn_data$file_name,
                          post_summary_mode=as.integer(post_summary_mode),
                          actFun = py_get_attr(bnn_model,'_act_fun'))
  return(post_pr)
}
