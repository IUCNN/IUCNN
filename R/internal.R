#' @importFrom  reticulate import
#' @importFrom utils read.table
#' @importFrom magrittr %>%
#' @importFrom dplyr select left_join mutate


get_footp <- function(x, file_path){
  test <- file.exists(file.path(file_path,
                                paste("HFP", x, ".tif", sep = "")))
  if(!test){
    download.file(paste("https://wcshumanfootprint.org/data/HFP", x, ".zip", sep = ""),
                  destfile = file.path(file_path, paste("HFP", x, ".zip", sep = "")))

    unzip(file.path(file_path, paste("HFP", x, ".zip", sep = "")),
          exdir = file_path)

    file.remove(file.path(file.path(file_path, paste("HFP", x, ".zip", sep = ""))))
    }
}

# BNN helpers
bnn_load_data <- function(features,
                          labels,
                          seed = 1234,
                          testsize = 0.1, # 10% test set
                          all_class_in_testset = TRUE,
                          header = TRUE, # input data has a header
                          instance_id = TRUE,
                          from_file = FALSE){
  # source python function
  bn <- reticulate::import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
  if(all_class_in_testset == TRUE){
    all_class_in_testset_switch <-  as.integer(1)
  }else{
    all_class_in_testset_switch <-  as.integer(0)
  }
  if(header == TRUE){
    header_switch <-  as.integer(1)
  }else{
    header_switch <-  as.integer(0)
  }
  if(instance_id == TRUE){
    instance_id_switch <- as.integer(1)
  }else{
    instance_id_switch <- as.integer(0)
  }
  dat <- bn$get_data(features,
                    labels,
                    seed = as.integer(seed),
                    testsize = testsize, # 10% test set
                    all_class_in_testset = all_class_in_testset_switch,
                    header = header_switch, # input data has a header
                    instance_id = instance_id_switch, # input data includes column with names of instances
                    from_file = from_file)
  return(dat)
}


create_BNN_model <- function(feature_data,
                             n_nodes_list,
                             seed = 1234,
                             use_class_weight = TRUE,
                             use_bias_node = TRUE,
                             actfun = 'swish',
                             prior = 1, # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
                             p_scale = 1, # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
                             init_std = 0.1){ # st dev of the initial weights

  # source python function
  bn <- reticulate::import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(use_class_weight == TRUE){
    use_class_weight_switch = as.integer(1)
  }else{
    use_class_weight_switch = as.integer(0)
  }
  if(use_bias_node==TRUE){
    use_bias_node_switch = as.integer(1)
  }else{
    use_bias_node_switch = as.integer(0)
  }

  alphas <- as.integer(c(0, 0))

  bnn_model <- bn$npBNN(feature_data,
                       n_nodes = as.integer(as.list(n_nodes_list)),
                       use_class_weights = use_class_weight_switch,
                       actFun = bn$ActFun(fun = actfun),
                       use_bias_node = use_bias_node_switch,
                       prior_f = as.integer(prior),
                       p_scale = as.integer(p_scale),
                       seed = as.integer(seed),
                       init_std = init_std)
  return(bnn_model)
}


MCMC_setup <- function(bnn_model,
                       update_f,
                       update_ws,
                       MCMC_temperature = 1,
                       likelihood_tempering = 1,
                       n_iteration = 5000,
                       sampling_f = 10, # how often to write to file (every n iterations)
                       print_f = 1000, # how often to print to screen (every n iterations)
                       n_post_samples = 100, # how many samples to keep in log file. If sampling exceeds this threshold it starts overwriting starting from line 1.
                       sample_from_prior = FALSE){

  # source python function
  bn <- reticulate::import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(sample_from_prior == TRUE){
    sample_from_prior_switch <- as.integer(1)
  }else{
    sample_from_prior_switch <- as.integer(0)
  }

  mcmc <- bn$MCMC(bnn_model,
                 update_f = update_f,
                 update_ws = update_ws,
                 temperature = MCMC_temperature,
                 n_iteration = as.integer(n_iteration),
                 sampling_f = as.integer(sampling_f),
                 print_f = as.integer(print_f),
                 n_post_samples = as.integer(n_post_samples),
                 sample_from_prior = sample_from_prior_switch,
                 likelihood_tempering = likelihood_tempering)
  return(mcmc)
}


run_MCMC <- function(bnn_model,
                     mcmc_object,
                     filename_stem = "BNN",
                     log_all_weights = FALSE){

  # source python function
  bn <- reticulate::import("np_bnn")
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(log_all_weights == TRUE){
    log_all_weights_switch <- as.integer(1)
  }else{
    log_all_weights_switch <- as.integer(0)
  }

  # initialize output files
  logger <- bn$postLogger(bnn_model,
                         filename = filename_stem,
                         log_all_weights = log_all_weights_switch)

  # run MCMC
  bn$run_mcmc(bnn_model, mcmc_object, logger)
  return(logger)
}


calculate_accuracy <- function(bnn_data,
                               logger,
                               bnn_model,
                               data = 'test',
                               post_summary_mode=1){

  # source python function
  bn <- reticulate::import("np_bnn")

  if (data == 'test'){
    features <- bnn_data$test_data
    labels <- bnn_data$test_labels
    instance_id <- bnn_data$id_test_data
  }else if(data =='train'){
    features <- bnn_data$data
    labels <- bnn_data$labels
    instance_id <- bnn_data$id_data
  }
  post_pr <- bn$predictBNN(features,
                          pickle_file = py_get_attr(logger,'_pklfile'),
                          test_labels = labels,
                          instance_id = instance_id,
                          fname = bnn_data$file_name,
                          post_summary_mode = as.integer(post_summary_mode))
  return(post_pr)
}


bnn_predict <- function(features,
                        instance_id,
                        model_path,
                        target_acc,
                        filename,
                        post_summary_mode=1){

  # source python function
  bn <- reticulate::import("np_bnn")


  post_pr = bn$predictBNN(as.matrix(features),
                          pickle_file = model_path,
                          instance_id = instance_id,
                          fname = filename,
                          target_acc = target_acc,
                          post_summary_mode = post_summary_mode
  )

  return(post_pr)
}

subsample_n_per_class <- function(features,
                                  labels,
                                  n_samples){
  # select same of each class
  a <- sample(which(labels$labels$labels == 0),n_samples)
  a <- append(a, sample(which(labels$labels$labels == 1), n_samples))
  a <- append(a, sample(which(labels$labels$labels == 2), n_samples))
  a <- append(a, sample(which(labels$labels$labels == 3), n_samples))
  a <- append(a, sample(which(labels$labels$labels == 4), n_samples))
  labels$labels <- labels$labels[a,]
  target_sp <- labels$labels$species
  features <- features[match(target_sp, features$species),]
  return(list(features, labels))
}

log_results <- function(res,logfile,init_logfile=FALSE){
  if (init_logfile){ # init a new logfile, make sure, you don't overwrite previous results
    header = c("mode","level","dropout_rate","seed","max_epochs","patience","n_layers","use_bias","rescale_features","randomize_instances","mc_dropout","mc_dropout_reps","act_f","act_f_out","cv_fold","validation_split","test_fraction","label_stretch_factor","label_noise_factor","train_acc","val_acc","test_acc","training_loss","validation_loss","confusion_LC","confusion_NT","confusion_VU","confusion_EN","confusion_CR","confusion_0","confusion_1")
    if(file.exists(logfile)){
      overwrite_prompt = readline(prompt="Specified log-file already exists and will be overwritten and all previous contents will be lost. Do you want to proceed? [Y/n]: ")
      if (overwrite_prompt == 'Y'){
        cat(header,file=logfile,sep="\t")
        cat('\n',file=logfile,append=T)
      }else{
        print('Not overwriting existing log-file. Please specify different logfile path or set init_logfile=FALSE')
        break
      }
    }else{
      cat(header,file=logfile,sep="\t")
      cat('\n',file=logfile,append=T)
    }
  }
  if (class(res)=="iucnn_model"){
    if (length(res$input_data$lookup.lab.num.z)==2){
      label_level = 'broad'
      confusion_matrix_lines = c(NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 paste(res$confusion_matrix[1,], collapse = '_'),
                                 paste(res$confusion_matrix[2,], collapse = '_'))
    }else{
      label_level = 'detail'
      confusion_matrix_lines = c(paste(res$confusion_matrix[1,], collapse = '_'),
                                 paste(res$confusion_matrix[2,], collapse = '_'),
                                 paste(res$confusion_matrix[3,], collapse = '_'),
                                 paste(res$confusion_matrix[4,], collapse = '_'),
                                 paste(res$confusion_matrix[5,], collapse = '_'),
                                 NaN,
                                 NaN)
    }
    cat(c(res$model,
          label_level,
          res$dropout_rate,
          res$seed,
          res$max_epochs,
          res$patience,
          paste(res$n_layers, collapse = '_'),
          res$use_bias,
          res$rescale_features,
          res$randomize_instances,
          res$mc_dropout,
          res$mc_dropout_reps,
          res$act_f,
          res$act_f_out,
          res$cv_fold,
          res$validation_split,
          res$test_fraction,
          res$label_stretch_factor,
          res$label_noise_factor,
          round(res$training_accuracy,6),
          round(res$validation_accuracy,6),
          round(res$test_accuracy,6),
          round(res$training_loss,6),
          round(res$validation_loss,6),
          confusion_matrix_lines),sep="\t",file=logfile,append=T)
    cat('\n',file=logfile,append=T)
  message(paste0("Model-testing results written to file: ",logfile))
  }
}


process_iucnn_input <- function(x, lab=NaN, mode=NaN, outpath='.', write_data_files=FALSE, verbose=1){
  if (typeof(lab) == 'double'){ # aka if lab=NaN when running from predict_iucnn
    # complete cases only
    tmp.in <- x[complete.cases(x),]
    if(nrow(tmp.in) != nrow(x)){
      mis <- x[!complete.cases(x),]
      if (verbose ==1){
        warning("Information for species was incomplete, species removed\n", paste(mis$species, "\n"))
      }
    }
    instance_id <- tmp.in$species
    #prepare input data
    tmp <- tmp.in %>%
      dplyr::select(-.data$species)

    dataset <- tmp
    labels <- NaN
    instance_names <- instance_id

  }else{
    ## specific checks
    if(!"species" %in% names(x)){
      stop("species column not found in x.
           The features input need a column named 'species'
           with the species names matching those in labels")
    }

    # merge species and labels to match order
    tmp.in <- left_join(x, lab$labels, by = "species")

    # check if species were lost by the merging
    if(nrow(tmp.in) != nrow(x)){
      mis <- x$species[!x$species %in% tmp$species]
      if (verbose ==1){
        warning("Labels for species not found, species removed.\n", paste(mis, "\n"))
      }
    }

    if(nrow(tmp.in) != nrow(lab$labels)){
      mis <- lab$labels$species[!lab$labels$species %in% tmp$species]
      if (verbose ==1){
        warning("Features for species not found, species removed.\n", paste(mis, "\n"))
      }
    }

    # complete cases only
    tmp <- tmp.in[complete.cases(tmp.in),]

    if(nrow(tmp) != nrow(tmp.in)){
      mis <- tmp.in[!complete.cases(tmp.in),]
      if (verbose ==1){
        warning("Information for species was incomplete, species removed\n", paste(mis$species, "\n"))
      }
    }

    # check that not all species were removed
    if(nrow(tmp) == 0){
      stop("Labels and features do not match or there are no species with complete features.")
    }

    # report the number of species
    t1 <- nrow(tmp)

    if(t1 < 200){
      if (verbose ==1){
        warning("The number of training taxa is low, consider including more species")
      }
    }

    if (verbose ==1){
      message(sprintf("%s species included in model training", t1))
    }

    # check class balance
    t2 <- table(tmp$labels)

    if(max(t2) / min(t2) > 3){
      if (verbose ==1){
        warning("Classes unbalanced")
      }
    }
    if (verbose ==1){
      message(sprintf("Class max/min representation ratio: %s", round(max(t2) / min(t2), 1)))
    }
    # prepare input data for the python function
    dataset <- tmp %>%
      dplyr::select(-.data$species, -.data$labels)

    if (mode=='bnn-class'){
      dataset <- tmp[, seq_along(names(tmp)) - 1]
    }

    instance_names <- tmp %>%
      dplyr::select(.data$species)

    labels <- tmp %>%
      dplyr::select(.data$labels)

    # prepare labels to start at 0
    if(min(labels$labels) != 0){
      if (verbose ==1){
        warning(sprintf("Labels need to start at 0. Labels substracted with %s", min(labels$labels)))
      }

      labels <-  labels %>%
        dplyr::mutate(labels = .data$labels - min(.data$labels))
    }

    if (mode=='bnn-class'){
      # in the current npbnn function we need to add a dummy column of instance names
      labels[['names']] <- replicate(length(labels$labels),'sp.')
      labels <- labels[, c('names','labels')]
    }

  }
  if (write_data_files == TRUE){
    write.table(as.matrix(dataset),paste(outpath,'iucnn_input_features.txt',sep = '/'),sep='\t',quote=FALSE,row.names=FALSE)
    if (typeof(lab) == 'list'){
      write.table(as.matrix(labels),paste(outpath,'iucnn_input_labels.txt',sep = '/'),sep='\t',quote=FALSE,row.names=FALSE)
    }
    write.table(as.matrix(instance_names),paste(outpath,'iucnn_input_instance_names.txt',sep = '/'),sep='\t',quote=FALSE,row.names=FALSE)
    write.table(names(dataset),paste(outpath,'iucnn_input_feature_names.txt',sep = '/'),sep='\t',quote=FALSE,row.names=FALSE)
  }

  return(list(dataset,labels,instance_names))
}






