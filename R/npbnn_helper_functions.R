#' npBNN helper functions (these are run internally and don't require a documentation)
#' @export
#' @import reticulate
#'


bnn_load_data <- function(features,
                          labels,
                          seed=1234,
                          testsize=0.1, # 10% test set
                          all_class_in_testset=TRUE,
                          header=TRUE, # input data has a header
                          instance_id=TRUE, # input data includes column with names of instances
                          from_file=FALSE){
  # source python function
  bn <- import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
  if(all_class_in_testset==TRUE){
    all_class_in_testset_switch = as.integer(1)
  }else{
    all_class_in_testset_switch = as.integer(0)
  }
  if(header==TRUE){
    header_switch = as.integer(1)
  }else{
    header_switch = as.integer(0)
  }
  if(instance_id==TRUE){
    instance_id_switch = as.integer(1)
  }else{
    instance_id_switch = as.integer(0)
  }
  dat = bn$get_data(features,
                    labels,
                    seed=as.integer(seed),
                    testsize=testsize, # 10% test set
                    all_class_in_testset=all_class_in_testset_switch,
                    header=header_switch, # input data has a header
                    instance_id=instance_id_switch, # input data includes column with names of instances
                    from_file=from_file)
  return(dat)
}


create_BNN_model <- function(feature_data,
                             n_nodes_list,
                             seed=1234,
                             use_class_weight=TRUE,
                             use_bias_node=TRUE,
                             actfun = 'swish',
                             prior = 1, # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
                             p_scale = 1, # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
                             init_std=0.1){ # st dev of the initial weights

  # source python function
  bn <- import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(use_class_weight==TRUE){
    use_class_weight_switch = as.integer(1)
  }else{
    use_class_weight_switch = as.integer(0)
  }
  if(use_bias_node==TRUE){
    use_bias_node_switch = as.integer(1)
  }else{
    use_bias_node_switch = as.integer(0)
  }

  alphas = as.integer(c(0, 0))

  bnn_model = bn$npBNN(feature_data,
                       n_nodes = as.integer(as.list(n_nodes_list)),
                       use_class_weights=use_class_weight_switch,
                       actFun=bn$ActFun(fun=actfun),
                       use_bias_node=use_bias_node_switch,
                       prior_f=as.integer(prior),
                       p_scale=as.integer(p_scale),
                       seed=as.integer(seed),
                       init_std=init_std)
  return(bnn_model)
}


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


run_MCMC <- function(bnn_model,
                     mcmc_object,
                     filename_stem = "BNN",
                     log_all_weights = FALSE){

  # source python function
  bn <- import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(log_all_weights==TRUE){
    log_all_weights_switch = as.integer(1)
  }else{
    log_all_weights_switch = as.integer(0)
  }

  # initialize output files
  logger = bn$postLogger(bnn_model, filename=filename_stem, log_all_weights=log_all_weights_switch)

  # run MCMC
  bn$run_mcmc(bnn_model, mcmc_object, logger)
  return(logger)
}


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
                          pickle_file=py_get_attr(logger,'_pklfile'),
                          test_labels=labels,
                          instance_id=instance_id,
                          fname=bnn_data$file_name,
                          post_summary_mode=as.integer(post_summary_mode))
  return(post_pr)
}


bnn_predict <- function(features,
                        instance_id,
                        model_path,
                        filename,
                        post_summary_mode=0){

  # source python function
  bn <- import("np_bnn")


  post_pr = bn$predictBNN(as.matrix(features),
                          pickle_file = model_path,
                          instance_id = instance_id,
                          fname = filename,
                          post_summary_mode = post_summary_mode
  )

  return(post_pr)
}



