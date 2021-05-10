#' Train a set of IUCNN models and log results to pick best model configuration
#'
#' Uses a data-frame of model-testing results generated with
#' \code{\link{modeltest_iucnn}} as input, and finds the best model
#' based on the chosen criterion.
#'
#'
#'@param x a data.frame of model-testing results as produced
#'by \code{\link{modeltest_iucnn}}
#'@param criterion name the criterion to rank models by (default="val_acc"). Valid options are
#'"val_acc","val_loss","weighted_error", or "total_class_matches"
#'(see details below):
#'- val_acc: highest validation accuracy
#'- val_loss: lowest validation loss
#'- weighted_error: lowest weighted error, e.g. an LC species misclassified as
#'                  CR has a weighted error of 4-0 = 4, while an LC species
#'                  misclassified as NT has a weighted error of 1-0 = 1.
#'                  These error scores are summed across all validation
#'                  predictions
#'- total_class_matches: picks the model that best reproduces the class
#'                       distribution in the validation data. When picking
#'                       this criterion it is not considered whether or not
#'                       individual instances are predicted correctly, but
#'                       instead it only looks at the overall class distribution
#'                       in the predicted data
#'
#'@param require_dropout logical (default=FALSE). If set to TRUE, the best model
#'that contains a dropout rate of > 0 will be picked, even if other non-dropout
#'models scored higher given the chosen criterion. Dropout models are required
#'for certain functionalities within IUCNN, such as e.g. choosing a target
#'accuracy when using predict_iucnn.
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")} for a
#'tutorial on how to run IUCNN.
#'
#'@return outputs an \code{iucnn_model} object containing all information about the best model.
#'
#' @examples
#'\dontrun{
#'# Model-testing
#'logfile = paste0("model_testing_results.txt")
#'model_testing_results = modeltest_iucnn( features,
#'                                         labels,
#'                                         logfile,
#'                                         model_outpath = 'iucnn_modeltest',
#'                                         mode = 'nn-class',
#'                                         seed = 1234,
#'                                         dropout_rate = c(0.0,0.1,0.3),
#'                                         n_layers = c('30','40_20','50_30_10'),
#'                                         cv_fold = 5,
#'                                         init_logfile = TRUE)
#'
#'# Selecting best model based on chosen criterion
#'best_iucnn_model = bestmodel_iucnn(model_testing_results,
#'                                    criterion = 'val_acc',
#'                                    require_dropout = TRUE)
#'}
#'
#'
#' @export
#' @importFrom utils read.csv
#' @importFrom checkmate assert_data_frame assert_character assert_logical assert_numeric assert_list

modeltest_iucnn <- function(x,
                            lab,
                            logfile = "model_testing_logfile.txt",
                            model_outpath = "modeltest",
                            mode = "nn-class",
                            cv_fold = 5,
                            validation_fraction = 0,
                            n_layers = c("50_30_10", "30"),
                            dropout_rate = c(0, 0.1, 0.3),
                            use_bias = TRUE,
                            balance_classes = FALSE,
                            seed = 1234,
                            label_stretch_factor = 1,
                            label_noise_factor = 0,
                            act_f = "relu",
                            act_f_out = "auto",
                            max_epochs = 5000,
                            patience = 200,
                            mc_dropout = TRUE,
                            mc_dropout_reps = 100,
                            randomize_instances = TRUE,
                            rescale_features = FALSE,
                            init_logfile = TRUE,
                            recycle_settings = FALSE) {

  # Check input assertion
  assert_data_frame(x)
  assert_class(lab, classes = "iucnn_labels")
  assert_character(mode)
  assert_character(logfile)
  assert_numeric(cv_fold)
  assert_numeric(dropout_rate, lower = 0, upper = 1)
  assert_character(n_layers)
  assert_logical(use_bias)
  assert_logical(balance_classes)
  assert_numeric(seed)
  assert_numeric(label_stretch_factor, lower = 0, upper = 2)
  assert_numeric(label_noise_factor, lower = 0, upper = 1)
  assert_character(act_f_out)
  assert_character(act_f)
  assert_numeric(max_epochs)
  assert_numeric(patience)
  assert_logical(randomize_instances)
  assert_logical(rescale_features)
  assert_logical(init_logfile)
  assert_logical(recycle_settings)

  if (file.exists(model_outpath)) {
    if (init_logfile == TRUE) {
      # we are starting with a new modeltest-logfile from scratch,
      #so we will also attempt to overwrite the modeltest dir
      overwrite_prompt <-
        readline(prompt = "Specified model_outpath dir already exists. Do you want to overwrite? [Y/n]: ")

      if (overwrite_prompt == "Y") {
        unlink(model_outpath, recursive = TRUE)
      } else {
        stop("Not overwriting existing model_outpath dir. Please specify different model_outpath.")
      }
    } else {
      # in this case we will be adding new models to the same output dir without overwriting anything
      do_nothing <- TRUE
    }
  } else {
    dir.create(model_outpath)
  }

  if (recycle_settings == TRUE) {
    new_logfile <- sub(".txt", "_new.txt", logfile)
    message(paste0("Copying the model-settings from the provided logfile. Model-testing results will be printed to new logfile ",
                   new_logfile,
                   ". Any manual settings provided with this function are ignored. Set recycle_settings=FALSE to disable this behaviour."))

    # load model configurations from logfile and run with modifications
    model_configurations_df <- read.csv(logfile, sep = "\t")

    # init new logfile
    log_results(NaN, new_logfile, NaN, init_logfile = TRUE)

    for (row_id in 1:dim(model_configurations_df)[1]) {
      print(paste0("Running model ", row_id, "/", dim(model_configurations_df)[1]))
      row <- model_configurations_df[row_id, ]
      mode <- row$mode
      dropout_rate <- row$dropout_rate
      seed <- row$seed
      max_epochs <- row$max_epochs
      patience <- row$patience
      n_layers <- row$n_layers
      use_bias <- row$use_bias
      balance_classes <- row$balance_classes
      rescale_features <- row$rescale_features
      randomize_instances <- row$randomize_instances
      mc_dropout <- row$mc_dropout
      mc_dropout_reps <- row$mc_dropout_reps
      act_f <- row$act_f
      act_f_out <- row$act_f_out
      cv_fold <- row$cv_fold
      validation_fraction <- row$validation_fraction
      label_stretch_factor <- row$label_stretch_factor
      label_noise_factor <- row$label_noise_factor

      res <- train_iucnn(x = x, lab = lab,
                         mode = mode,
                         path_to_output = paste0(model_outpath, "/model_", row_id),
                         cv_fold = cv_fold,
                         seed = seed,
                         max_epochs = max_epochs,
                         patience = patience,
                         validation_fraction = validation_fraction,
                         n_layers = n_layers,
                         use_bias = use_bias,
                         balance_classes = balance_classes,
                         act_f = act_f,
                         act_f_out = act_f_out,
                         label_stretch_factor = label_stretch_factor,
                         randomize_instances = randomize_instances,
                         dropout_rate = dropout_rate,
                         mc_dropout = mc_dropout,
                         mc_dropout_reps = mc_dropout_reps,
                         label_noise_factor = label_noise_factor,
                         rescale_features = rescale_features,
                         save_model = TRUE,
                         overwrite = TRUE,
                         verbose = 0)

      iucnn_model_path <- paste0(model_outpath, "/model_",
                                 row_id,
                                 "/iucnn_model.rds")

      #write results to disk
      saveRDS(res, iucnn_model_path)
      log_results(res, new_logfile, iucnn_model_path, init_logfile = FALSE)
    }
    outfile <- new_logfile
  } else {
    if (cv_fold > 1) {
      message("Evaluating models using ",
              cv_fold,
              "-fold cross-validation (validation_fraction setting is ignored).")
      validation_fraction <- 0
    } else {
      if (validation_fraction == 0) {
        stop(paste0("No validation set defined: cv_fold is set to ",
                    cv_fold,
                    " and validation_fraction to ",
                    validation_fraction,
                    ". Change either one of these settings to define a validation set."))
      } else {
        warning(paste0("Running single training round for each model with ",
                       validation_fraction,
                       " validation set."))
      }
    }
    if (init_logfile == TRUE) {
      log_results(NaN, logfile, NaN, init_logfile = init_logfile)
      delta_i <- 0
    } else {
      model_configurations_df <- read.csv(logfile, sep = "\t")
      delta_i <- dim(model_configurations_df)[1]
    }

    permutations <- do.call(expand.grid, list(cv_fold, n_layers,
                                              dropout_rate,
                                              use_bias,
                                              seed,
                                              label_stretch_factor,
                                              label_noise_factor,
                                              act_f,
                                              act_f_out,
                                              max_epochs,
                                              patience,
                                              randomize_instances,
                                              rescale_features,
                                              mc_dropout,
                                              mc_dropout_reps,
                                              mode,
                                              validation_fraction,
                                              balance_classes))
    n_permutations <- dim(permutations)[1]

    message(paste0("Running model test for ",
                   n_permutations,
                   " models. This may take a while..."))

    for (i in 1:n_permutations) {
      print(paste0("Running model ", i, "/", n_permutations))
      settings <- permutations[i, ]
      cv_fold_i <- as.integer(settings[[1]])
      n_layers_i <- as.character(settings[[2]])
      dropout_rate_i <- as.numeric(settings[[3]])
      use_bias_i <- as.logical(settings[[4]])
      seed_i <- as.integer(settings[[5]])
      label_stretch_factor_i <- as.numeric(settings[[6]])
      label_noise_factor_i <- as.numeric(settings[[7]])
      act_f_i <- as.character(settings[[8]])
      act_f_out_i <- as.character(settings[[9]])
      max_epochs_i <- as.integer(settings[[10]])
      patience_i <- as.integer(settings[[11]])
      randomize_instances_i <- as.logical(settings[[12]])
      rescale_features_i <- as.logical(settings[[13]])
      mc_dropout_i <- as.logical(settings[[14]])
      mc_dropout_reps_i <- as.integer(settings[[15]])
      mode_i <- as.character(settings[[16]])
      validation_fraction_i <- as.numeric(settings[[17]])
      balance_classes_i <- as.logical(settings[[18]])

      # set out act fun if chosen auto
      if (act_f_out_i == "auto") {
        if (mode_i == "nn-reg") {
          act_f_out_i <- "tanh"
        } else {
          act_f_out_i <- "softmax"
        }
      }

      # train model
      res <- train_iucnn(x = x,
                         lab = lab,
                         mode = mode_i,
                         path_to_output = paste0(model_outpath,
                                                 "/model_",
                                                 i + delta_i),
                         cv_fold = cv_fold_i,
                         seed = seed_i,
                         max_epochs = max_epochs_i,
                         patience = patience_i,
                         validation_fraction = validation_fraction_i,
                         n_layers = n_layers_i,
                         use_bias = use_bias_i,
                         balance_classes = balance_classes_i,
                         act_f = act_f_i,
                         act_f_out = act_f_out_i,
                         label_stretch_factor = label_stretch_factor_i,
                         randomize_instances = randomize_instances_i,
                         dropout_rate = dropout_rate_i,
                         mc_dropout = mc_dropout_i,
                         mc_dropout_reps = mc_dropout_reps_i,
                         label_noise_factor = label_noise_factor_i,
                         rescale_features = rescale_features_i,
                         save_model = TRUE,
                         overwrite = TRUE,
                         verbose = 0)

      iucnn_model_path <- paste0(model_outpath,
                                 "/model_", i + delta_i,
                                 "/iucnn_model.rds")
      saveRDS(res, iucnn_model_path)
      log_results(res, logfile, iucnn_model_path, init_logfile = FALSE)
    }
    outfile <- logfile
  }

  model_testing_df <- read.csv(outfile, sep = "\t")
  return(model_testing_df)
}
