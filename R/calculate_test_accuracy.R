#' Calculate test accuracy
#' @export
#' @import reticulate
#'
#'
calculate_test_accuracy <- function(bnn_data,
                                    logger,
                                    post_summary_mode=0){

  # source python function
  bn <- import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
  post_pr_test = bn$predictBNN(bnn_data$test_data,
                               pickle_file=py_get_attr(logger,'_w_file'),
                               test_labels=bnn_data$test_labels,
                               instance_id=bnn_data$id_test_data,
                               fname=bnn_data$file_name,
                               post_summary_mode=as.integer(post_summary_mode))
  return(post_pr_test)
}
