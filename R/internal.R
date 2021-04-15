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
    header = c("mode","level","dropout_rate","seed","max_epochs","patience","n_layers","use_bias","rescale_features","randomize_instances","mc_dropout","mc_dropout_reps","act_f","act_f_out","cv_fold","validation_fraction","label_stretch_factor","label_noise_factor","final_train_epoch_all","final_train_epoch_mean","train_acc","val_acc","training_loss","validation_loss","confusion_LC","confusion_NT","confusion_VU","confusion_EN","confusion_CR","confusion_0","confusion_1","delta_LC","delta_NT","delta_VU","delta_EN","delta_CR","delta_0","delta_1")
    if(file.exists(logfile)){
      overwrite_prompt = readline(prompt="Specified log-file already exists. Do you want to overwrite? [Y/n]: ")
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
      ratio_prediction_lines = c(NaN,NaN,abs(get_cat_count(res$validation_labels,max_cat = 1)-get_cat_count(res$validation_predictions,max_cat = 1)))
      confusion_matrix_lines = c(NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 paste(res$confusion_matrix[1,], collapse = '_'),
                                 paste(res$confusion_matrix[2,], collapse = '_'))
    }else{
      label_level = 'detail'
      ratio_prediction_lines = c(abs(get_cat_count(res$validation_labels,max_cat = 4)-get_cat_count(res$validation_predictions,max_cat = 4)),NaN,NaN)
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
          res$validation_fraction,
          res$label_stretch_factor,
          res$label_noise_factor,
          paste(res$final_training_epoch, collapse = '_'),
          round(mean(res$final_training_epoch),0),
          round(res$training_accuracy,6),
          round(res$validation_accuracy,6),
          round(res$training_loss,6),
          round(res$validation_loss,6),
          confusion_matrix_lines,
          ratio_prediction_lines),sep="\t",file=logfile,append=T)
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

get_best_model <- function(model_testing_results,rank_mode=1){
  if (rank_mode == 1){ # highest validation accuracy
    best_model = model_testing_results[which((model_testing_results$val_acc == max(model_testing_results$val_acc,na.rm = TRUE))),]
  }else if (rank_mode == 2){ # lowest validation loss
    best_model = model_testing_results[which((model_testing_results$validation_loss == min(model_testing_results$validation_loss,na.rm = TRUE))),]
  }else if (rank_mode == 3){ # smallest weighted misclassification error
    if (typeof(model_testing_results$confusion_LC)=='character'){
      LC_weighted_errors = get_weighted_errors(model_testing_results,'confusion_LC',1)
      NT_weighted_errors = get_weighted_errors(model_testing_results,'confusion_NT',2)
      VU_weighted_errors = get_weighted_errors(model_testing_results,'confusion_VU',3)
      EN_weighted_errors = get_weighted_errors(model_testing_results,'confusion_EN',4)
      CR_weighted_errors = get_weighted_errors(model_testing_results,'confusion_CR',5)
      error_list = list(LC_weighted_errors,NT_weighted_errors,VU_weighted_errors,EN_weighted_errors,CR_weighted_errors)
    }else{
      not_threatened_weighted_errors = get_weighted_errors(model_testing_results,'confusion_0',1)
      threatened_weighted_errors = get_weighted_errors(model_testing_results,'confusion_1',2)
      error_list = list(not_threatened_weighted_errors,threatened_weighted_errors)
    }
    total_error_all_rows = rowSums(data.frame(t(matrix(unlist(error_list), nrow=length(error_list), byrow=TRUE))))
    best_row_index = which.min(total_error_all_rows)
    best_model = model_testing_results[best_row_index,]
  }else if (rank_mode == 4){ # fewest class misclassifications
    if (typeof(model_testing_results$confusion_LC)=='character'){
      sum_false_classes = rowSums(model_testing_results[,c('delta_LC','delta_NT','delta_VU','delta_EN','delta_CR')])
    }else{
      sum_false_classes = rowSums(model_testing_results[,c('delta_0','delta_1')])
    }
    best_row_index = which.min(sum_false_classes)
    best_model = model_testing_results[best_row_index,]
  }else{
    message(paste0('Invalid choice rank_mode = ',rank_mode))
    break
  }
  return(best_model)
}

plot_predictions <- function(predictions){
  counts_detail = table(predictions,useNA = 'ifany')
  bar_names = names(counts_detail)
  bar_names[is.na(bar_names)] = 'NA'
  barplot(counts_detail,names.arg = bar_names)
}

get_weighted_errors <- function(model_testing_results,colname='confusion_LC',true_index=1){
  stat_col = strsplit(model_testing_results[,colname],'_')
  a = data.frame(matrix(unlist(stat_col), nrow=length(stat_col), byrow=TRUE))
  weighted_errors = c()
  for (i in 1:dim(a)[1]){
    row = as.numeric(a[i,])
    weighted_error = sum(abs((1:dim(a)[2]-true_index))*row)
    weighted_errors = c(weighted_errors,weighted_error)
  }
  return(weighted_errors)
}

get_cat_count <- function(target_vector,max_cat = 4){
  cat_counts = c()
  for (i in 0:max_cat){
    cat_counts = c(cat_counts,length(target_vector[target_vector==i]))
  }
  return(cat_counts)
}

get_confusion_matrix <- function(best_model){
  if (typeof(best_model$confusion_LC)=='character'){
    target_cols = as.character(best_model[,c('confusion_LC','confusion_NT','confusion_VU','confusion_EN','confusion_CR')])
    count_strings = strsplit(target_cols,'_')
    confusion_matrix = matrix(as.integer(unlist(count_strings)), nrow=length(count_strings), byrow=TRUE)
    #estimates_per_class = colSums(confusion_matrix)
  }else{
    target_cols = as.character(best_model[,c('confusion_0','confusion_1')])
    count_strings = strsplit(target_cols,'_')
    confusion_matrix = matrix(as.integer(unlist(count_strings)), nrow=length(count_strings), byrow=TRUE)
  }
  return(confusion_matrix)
}


evaluate_model <- function(features,labels,best_model){

  res = train_iucnn(features,
                    labels,
                    path_to_output = "",
                    read_settings = FALSE,
                    mode = best_model$mode,
                    validation_fraction = best_model$validation_fraction,
                    cv_fold = best_model$cv_fold,
                    seed = best_model$seed,
                    max_epochs = best_model$max_epochs,
                    patience = best_model$patience,
                    n_layers = best_model$n_layers,
                    use_bias = best_model$use_bias,
                    act_f = best_model$act_f,
                    act_f_out = best_model$act_f_out,
                    label_stretch_factor = best_model$label_stretch_factor,
                    randomize_instances = best_model$randomize_instances,
                    dropout_rate = best_model$dropout_rate,
                    mc_dropout = best_model$mc_dropout,
                    mc_dropout_reps = best_model$mc_dropout_reps,
                    label_noise_factor = best_model$label_noise_factor,
                    rescale_features = best_model$rescale_features,
                    save_model = FALSE,
                    overwrite = FALSE,
                    verbose = 0)
  summary(res)
  plot(res)
  return(res)
}
