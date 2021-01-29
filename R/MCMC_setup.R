#' Set up MCMC for BNN
#' @export
#' @import reticulate

MCMC_setup <- function(bnn_model,
                       update_f,
                       update_ws,
                       MCMC_temperature=1,
                       likelihood_tempering=1,
                       n_iteration=5000,
                       sampling_f=10, # how often to write to file (every n iterations)
                       print_f=1000, # how often to print to screen (every n iterations)
                       n_post_samples = 100, # how many samples to keep in log file. If sampling exceeds this threshold it starts overwriting starting from line 1.
                       sample_from_prior = FALSE){

  # source python function
  bn <- import("np_bnn")
  reticulate::py_install("https://github.com/dsilvestro/npBNN/archive/v0.1.4.tar.gz", pip = TRUE)
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(sample_from_prior==TRUE){
    sample_from_prior_switch = as.integer(1)
  }else{
    sample_from_prior_switch = as.integer(0)
  }

  mcmc = bn$MCMC(bnn_model,
              update_f=update_f,
              update_ws=update_ws,
              temperature = MCMC_temperature,
              n_iteration=as.integer(n_iteration),
              sampling_f=as.integer(sampling_f),
              print_f=as.integer(print_f),
              n_post_samples=as.integer(n_post_samples),
              sample_from_prior=sample_from_prior_switch,
              likelihood_tempering=likelihood_tempering)
  return(mcmc)
}
