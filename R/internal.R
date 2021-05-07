#' @importFrom reticulate import
#' @importFrom utils read.table
#' @importFrom magrittr %>%
#' @importFrom dplyr select left_join mutate
#' @importFrom utils write.table

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
                          randomize_order = TRUE,
                          header = TRUE, # input data has a header
                          instance_id = TRUE,
                          from_file = FALSE){
  # source python function
  bn <- reticulate::import("np_bnn")

  dat <- bn$get_data(features,
                    labels,
                    seed = as.integer(seed),
                    testsize = testsize, # 10% test set
                    all_class_in_testset = as.integer(all_class_in_testset),
                    randomize_order=randomize_order,
                    header = as.integer(header), # input data has a header
                    instance_id = as.integer(instance_id), # input data includes column with names of instances
                    from_file = from_file)
  return(dat)
}


create_BNN_model <- function(feature_data,
                             n_nodes_list,
                             seed = 1234,
                             use_class_weight = TRUE,
                             use_bias_node = 3,
                             actfun = 'swish',
                             prior = 1, # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
                             p_scale = 1, # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
                             init_std = 0.1){ # st dev of the initial weights

  # source python function
  bn <- reticulate::import("np_bnn")

  alphas <- as.integer(c(0, 0))

  bnn_model <- bn$npBNN(feature_data,
                       n_nodes = as.integer(as.list(n_nodes_list)),
                       use_class_weights = as.integer(use_class_weight),
                       actFun = bn$ActFun(fun = actfun),
                       use_bias_node = as.integer(use_bias_node),
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

  mcmc <- bn$MCMC(bnn_model,
                 update_f = update_f,
                 update_ws = update_ws,
                 temperature = MCMC_temperature,
                 n_iteration = as.integer(n_iteration),
                 sampling_f = as.integer(sampling_f),
                 print_f = as.integer(print_f),
                 n_post_samples = as.integer(n_post_samples),
                 sample_from_prior = as.integer(sample_from_prior),
                 likelihood_tempering = likelihood_tempering)
  return(mcmc)
}


run_MCMC <- function(bnn_model,
                     mcmc_object,
                     filename_stem = "BNN",
                     log_all_weights = FALSE){

  # source python function
  bn <- reticulate::import("np_bnn")

  # initialize output files
  logger <- bn$postLogger(bnn_model,
                         filename = filename_stem,
                         log_all_weights = as.integer(log_all_weights))

  # run MCMC
  bn$run_mcmc(bnn_model, mcmc_object, logger)
  return(logger)
}


calculate_accuracy <- function(bnn_data,
                               logger,
                               bnn_model,
                               data = 'test',
                               post_summary_mode = 0){

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
                        post_cutoff,
                        filename,
                        post_summary_mode=1){

  # source python function
  bn <- reticulate::import("np_bnn")


  post_pr <- bn$predictBNN(as.matrix(features),
                          pickle_file = model_path,
                          instance_id = instance_id,
                          fname = filename,
                          post_cutoff = post_cutoff,
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

log_results <- function(res,logfile,
                        iucnn_model_out,
                        init_logfile = FALSE){
  if (init_logfile){ # init a new logfile, make sure, you don't overwrite previous results
    header <- c("mode",
                "level",
                "dropout_rate",
                "seed",
                "max_epochs",
                "patience",
                "n_layers",
                "use_bias",
                "balance_classes",
                "rescale_features",
                "randomize_instances",
                "mc_dropout",
                "mc_dropout_reps",
                "act_f",
                "act_f_out",
                "cv_fold",
                "validation_fraction",
                "label_stretch_factor",
                "label_noise_factor",
                "final_train_epoch_all",
                "final_train_epoch_mean",
                "train_acc",
                "val_acc",
                "training_loss",
                "validation_loss",
                "confusion_LC",
                "confusion_NT",
                "confusion_VU",
                "confusion_EN",
                "confusion_CR",
                "confusion_0",
                "confusion_1",
                "delta_LC",
                "delta_NT",
                "delta_VU",
                "delta_EN",
                "delta_CR",
                "delta_0",
                "delta_1",
                "model_outpath")
    if(file.exists(logfile)){
      overwrite_prompt <-  readline(prompt="Specified log-file already exists. Do you want to overwrite? [Y/n]: ")
      if (overwrite_prompt == 'Y'){
        cat(header, file = logfile, sep = "\t")
        cat('\n', file = logfile, append = TRUE)
      }else{
        stop('Not overwriting existing log-file. Please specify different logfile path or set init_logfile=FALSE')
      }
    }else{
      cat(header, file = logfile,sep="\t")
      cat('\n', file = logfile, append = TRUE)
    }
  }
  if (class(res) == "iucnn_model"){
    if (length(res$input_data$lookup.lab.num.z) == 2){
      label_level <- 'broad'
      ratio_prediction_lines <- c(NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 abs(get_cat_count(res$validation_labels,max_cat = 1) -
                                       get_cat_count(res$validation_predictions,max_cat = 1)))
      confusion_matrix_lines <- c(NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 NaN,
                                 paste(res$confusion_matrix[1,], collapse = '_'),
                                 paste(res$confusion_matrix[2,], collapse = '_'))
    }else{
      label_level <- 'detail'
      ratio_prediction_lines <- c(abs(get_cat_count(res$validation_labels, max_cat = 4) -
                                       get_cat_count(res$validation_predictions, max_cat = 4)),
                                 NaN,
                                 NaN)
      confusion_matrix_lines <- c(paste(res$confusion_matrix[1,], collapse = '_'),
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
          res$balance_classes,
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
          ratio_prediction_lines,
          iucnn_model_out), sep="\t", file = logfile, append = TRUE)
    cat('\n', file = logfile, append = TRUE)
  message(paste0("Model-testing results written to file: ", logfile))
  }
}


process_iucnn_input <- function(x,
                                lab = NaN,
                                mode = NaN,
                                outpath = '.',
                                write_data_files = FALSE,
                                verbose = 1){
  if (typeof(lab) == 'double'){ # aka if lab=NaN when running from predict_iucnn
    # complete cases only
    tmp.in <- x[complete.cases(x),]
    if(nrow(tmp.in) != nrow(x)){
      mis <- x[!complete.cases(x),]
      if (verbose == 1){
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
      if (verbose == 1){
        warning("Labels for species not found, species removed.\n", paste(mis, "\n"))
      }
    }

    if(nrow(tmp.in) != nrow(lab$labels)){
      mis <- lab$labels$species[!lab$labels$species %in% tmp$species]
      if (verbose == 1){
        warning("Features for species not found, species removed.\n", paste(mis, "\n"))
      }
    }

    # complete cases only
    tmp <- tmp.in[complete.cases(tmp.in),]

    if(nrow(tmp) != nrow(tmp.in)){
      mis <- tmp.in[!complete.cases(tmp.in),]
      if (verbose == 1){
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
      if (verbose == 1){
        warning("The number of training taxa is low, consider including more species")
      }
    }

    if (verbose == 1){
      message(sprintf("%s species included in model training", t1))
    }

    # check class balance
    t2 <- table(tmp$labels)

    if(max(t2) / min(t2) > 3){
      if (verbose ==1){
        warning("Classes unbalanced")
      }
    }
    if (verbose == 1){
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
      if (verbose == 1){
        warning(sprintf("Labels need to start at 0. Labels substracted with %s",
                        min(labels$labels)))
      }

      labels <- labels %>%
        dplyr::mutate(labels = .data$labels - min(.data$labels))
    }

    if (mode =='bnn-class'){
      # in the current npbnn function we need to add a dummy column of instance names
      labels[['names']] <- replicate(length(labels$labels),'sp.')
      labels <- labels[, c('names','labels')]
    }

  }
  if (write_data_files){
    write.table(as.matrix(dataset),
                paste(outpath,'iucnn_input_features.txt' , sep = '/'),
                sep = '\t',
                quote = FALSE,
                row.names = FALSE)
    if (typeof(lab) == 'list'){
      write.table(as.matrix(labels),paste(outpath,'iucnn_input_labels.txt',sep = '/'),
                  sep = '\t',
                  quote = FALSE,
                  row.names = FALSE)
    }
    write.table(as.matrix(instance_names), paste(outpath,'iucnn_input_instance_names.txt', sep = '/'),
                sep = '\t',
                quote = FALSE,
                row.names = FALSE)
    write.table(names(dataset),paste(outpath,'iucnn_input_feature_names.txt',sep = '/'),
                sep = '\t',
                quote = FALSE,
                row.names = FALSE)
  }

  return(list(dataset,labels,instance_names))
}


rank_models <- function(model_testing_results, rank_by = "val_acc") {
  if (rank_by == "val_acc") {
    # highest validation accuracy
    sorted_model_testing_results <-
      model_testing_results[order(model_testing_results$val_acc, decreasing = TRUE), ]
  } else if (rank_by == "val_loss") {
    # lowest validation loss
    sorted_model_testing_results <-
      model_testing_results[order(model_testing_results$validation_loss, decreasing = FALSE), ]
  } else if (rank_by == "weighted_error") {
    # smallest weighted misclassification error
    if (typeof(model_testing_results$confusion_LC) == "character") {
      LC_weighted_errors <- get_weighted_errors(model_testing_results, "confusion_LC", 1)
      NT_weighted_errors <- get_weighted_errors(model_testing_results, "confusion_NT", 2)
      VU_weighted_errors <- get_weighted_errors(model_testing_results, "confusion_VU", 3)
      EN_weighted_errors <- get_weighted_errors(model_testing_results, "confusion_EN", 4)
      CR_weighted_errors <- get_weighted_errors(model_testing_results, "confusion_CR", 5)
      error_list <- list(LC_weighted_errors,
                        NT_weighted_errors,
                        VU_weighted_errors,
                        EN_weighted_errors,
                        CR_weighted_errors)
    } else {
      not_threatened_weighted_errors <- get_weighted_errors(model_testing_results, "confusion_0", 1)
      threatened_weighted_errors <- get_weighted_errors(model_testing_results, "confusion_1", 2)
      error_list <- list(not_threatened_weighted_errors, threatened_weighted_errors)
    }
    total_error_all_rows <- rowSums(data.frame(t(matrix(unlist(error_list),
                                                       nrow = length(error_list),
                                                       byrow = TRUE))))
    model_testing_results["weighted_error"] <-  total_error_all_rows
    sorted_model_testing_results <-
      model_testing_results[order(model_testing_results$weighted_error,
                                  decreasing = FALSE), ]
  } else if (rank_by == "total_class_matches") {

    # fewest class misclassifications
    if (typeof(model_testing_results$confusion_LC) == "character") {
      sum_false_classes <- rowSums(model_testing_results[, c("delta_LC",
                                                             "delta_NT",
                                                             "delta_VU",
                                                             "delta_EN",
                                                             "delta_CR")])
    } else {
      sum_false_classes <- rowSums(model_testing_results[, c("delta_0", "delta_1")])
    }
    model_testing_results["total_class_error"] <- sum_false_classes
    sorted_model_testing_results <-
      model_testing_results[order(model_testing_results$total_class_error,
                                  decreasing = FALSE), ]
  } else {
    stop(paste0("Invalid choice rank_by = '",
                rank_by,
                "'. Choose from 'val_acc','val_loss','weighted_error' ,or 'total_class_matches'"))
  }
  return(sorted_model_testing_results)
}

best_model_iucnn <- function(model_testing_results,
                             criterion = "val_acc",
                             require_dropout = FALSE) {
  ranked_models <- rank_models(model_testing_results, rank_by = criterion)
  if (require_dropout) {
    best_model <- ranked_models[ranked_models$dropout_rate > 0, ][1, ]
  } else {
    best_model <- ranked_models[1, ]
  }

  cat("Best model:\n")
  cat("", sprintf("%s: %s\n", names(best_model), best_model))
  cat("\n")

  iucnn_model <-  readRDS(best_model$model_outpath)
  return(iucnn_model)
}

model_summary <- function(best_model,
                          write_file = FALSE,
                          outfile_name = NULL) {
  cat("Best model:\n")
  cat("", sprintf("%s: %s\n", names(best_model), best_model))
  cat("\n")

  train_acc <- best_model$train_acc
  val_acc <- best_model$val_acc
  label_detail <- best_model$level
  cm <- get_confusion_matrix(best_model)

  if (label_detail == "broad") {
    n_classes <- 2
    maxlab <- 1
  } else if (label_detail == "detail") {
    n_classes <- 5
    maxlab <- 4
  } else {
    stop(paste0("Unknown label level: '",
                label_detail,
                "'. Currently only supporting 'broad' (N=2) or 'detail' (N=5)"))
  }

  cat(sprintf("Training accuracy: %s\n", round(train_acc, 3)))

  cat(sprintf("Accuracy on unseen data: %s\n", round(val_acc, 3)))

  cat(sprintf("Label detail: %s Classes (%s)\n\n", n_classes, label_detail))

  cat("Confusion matrix (Rows: true labels, Columns: predicted labels):\n")
  print(cm)

  if (write_file) {
    if (is.null(outfile_name)) {
      outfile_name <- "evaluation_best_model.txt"
    }
    sink(outfile_name)

    cat("Best model:\n")
    cat("", sprintf("%s: %s\n", names(best_model), c(best_model)))
    cat("\n")
    cat(sprintf("Training accuracy: %s\n", round(train_acc, 3)))

    cat(sprintf("Accuracy on unseen data: %s\n", round(val_acc, 3)))

    cat(sprintf("Label detail: %s Classes (%s)\n\n", n_classes, label_detail))

    cat("Confusion matrix (Rows: true labels, Columns: predicted labels):\n")
    print(cm)
    sink()
    print(paste0("Model evaluation results of best model written to ", outfile_name))
  }
}


evaluate_iucnn <- function(res) {
  if (res$dropout_rate == 0) {
    warning("No acc-thres-tbl and class-freq calculation. Provide model with dropout_rate > 0 to enable these functions.")
  }
  summary(res)
  plot(res)

  get_mc_dropout_cat_counts
  cat_count_out <- get_mc_dropout_cat_counts(res)
  accthres_tbl <- res$accthres_tbl
}

get_weighted_errors <- function(model_testing_results,
                                colname = "confusion_LC",
                                true_index = 1) {
  stat_col <- strsplit(model_testing_results[, colname], "_")
  a <- data.frame(matrix(unlist(stat_col), nrow = length(stat_col), byrow = TRUE))
  weighted_errors <- c()
  for (i in 1:dim(a)[1]) {
    row <- as.numeric(a[i, ])
    weighted_error <- sum(abs((1:dim(a)[2] - true_index)) * row)
    weighted_errors <- c(weighted_errors, weighted_error)
  }
  return(weighted_errors)
}

get_cat_count <- function(target_vector,
                          max_cat = 4) {
  cat_counts <- c()
  for (i in 0:max_cat) {
    cat_counts <- c(cat_counts, length(target_vector[target_vector == i]))
  }
  return(cat_counts)
}

get_confusion_matrix <- function(best_model) {
  if (typeof(best_model$confusion_LC) == "character") {
    target_cols <- as.character(best_model[, c("confusion_LC",
                                               "confusion_NT",
                                               "confusion_VU",
                                               "confusion_EN",
                                               "confusion_CR")])
    count_strings <- strsplit(target_cols, "_")

    confusion_matrix <- matrix(as.integer(unlist(count_strings)),
                               nrow = length(count_strings),
                               byrow = TRUE)
    confusion_matrix <- data.frame(confusion_matrix,
                                   row.names = c("LC", "NT", "VU", "EN", "CR"))
    names(confusion_matrix) <- c("LC", "NT", "VU", "EN", "CR")
  } else {
    target_cols <- as.character(best_model[, c("confusion_0", "confusion_1")])
    count_strings <- strsplit(target_cols, "_")

    confusion_matrix <- matrix(as.integer(unlist(count_strings)),
                               nrow = length(count_strings), byrow = TRUE)
    confusion_matrix <- data.frame(confusion_matrix,
                                   row.names = c("Not Threatened", "Threatened"))
    names(confusion_matrix) <- c("Not threatened", "Threatened")
  }
  return(confusion_matrix)
}


get_mc_dropout_cat_counts <- function(mc_dropout_probs,
                                      label_dict,
                                      mc_dropout,
                                      true_lab=NaN,
                                      nreps = 1000) {

  if (mc_dropout == FALSE){
    warning("This model contains no MC-dropout predictions for unseen data.
            No sampled_cat_freqs can be calculated for this model.")
    cat_count_all_matrix <- NaN
    true_cat_count <- NaN

  }else{
    nlabs <- length(label_dict)
    if (is.nan(true_lab[1])){
      true_cat_count <- NaN
    }else{
      true_cat_count <- get_cat_count(true_lab, max_cat = nlabs - 1)
    }
    n_instances <- dim(mc_dropout_probs)[1]
    cat_mcdropout_sample <- c()
    for (i in 1:n_instances) {
      cat_sample <- replicate(nreps, sample(1:nlabs - 1,
                                            size = 1,
                                            prob = mc_dropout_probs[i, ]))
      cat_mcdropout_sample <- c(cat_mcdropout_sample, c(cat_sample))
    }
    cat_mcdropout_sample_matrix <- matrix(cat_mcdropout_sample, nrow = nreps)
    cat_count_all <- c()
    for (row_id in 1:nreps) {
      row <- cat_mcdropout_sample_matrix[row_id, ]
      cat_count_sample <- get_cat_count(row, max_cat = nlabs - 1)
      cat_count_all <- c(cat_count_all, cat_count_sample)
    }
    cat_count_all_matrix <- t(matrix(cat_count_all, ncol = nreps))
  }

  output <- NULL
  output$predicted_class_count <- cat_count_all_matrix
  output$true_class_count <- true_cat_count
  return(output)
}

rnd <- function(x) trunc(x + sign(x) * 0.5)
