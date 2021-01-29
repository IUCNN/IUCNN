#' Run MCMC
#' @export
#' @import reticulate
#'
run_MCMC <- function(bnn_model,
                     mcmc_object,
                     filename_stem = "BNN",
                     log_all_weights = FALSE){

  # source python function
  reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(log_all_weights==TRUE){
    log_all_weights_switch = as.integer(1)
  }else{
    log_all_weights_switch = as.integer(0)
  }

  # initialize output files
  logger = postLogger(bnn_model, filename=filename_stem, log_all_weights=log_all_weights_switch)

  # run MCMC
  run_mcmc(bnn_model, mcmc_object, logger)
  return(logger)
}
