#' @export
#' @importFrom utils read.csv
#' @importFrom checkmate assert_data_frame assert_character assert_logical assert_numeric assert_list


modeltest_iucnn <- function(x,
                            lab,
                            logfile = 'model_testing_logfile.txt',
                            mode = 'nn-class',
                            cv_fold = 5,
                            n_layers = c('50_30_10','40_20','30'),
                            dropout_rate = c(0.0,0.1,0.3),
                            use_bias = TRUE,
                            seed = 1234,
                            label_stretch_factor = 1.0,
                            label_noise_factor = 0.0,
                            act_f = "relu",
                            act_f_out = "auto",
                            max_epochs = 500,
                            patience = 30,
                            validation_split = 0.1,
                            test_fraction = 0.0,
                            mc_dropout = TRUE,
                            mc_dropout_reps = 10,
                            randomize_instances = TRUE,
                            rescale_features = FALSE,
                            init_logfile = TRUE,
                            recycle_settings = FALSE){

  # Check input
  ## assertion
  assert_data_frame(x)
  assert_class(lab, classes = "iucnn_labels")
  assert_character(mode)
  assert_character(logfile)
  assert_numeric(cv_fold)
  assert_numeric(dropout_rate, lower = 0, upper = 1)
  assert_character(n_layers)
  assert_logical(use_bias)
  assert_numeric(seed)
  assert_numeric(label_stretch_factor, lower = 0, upper = 1)
  assert_numeric(label_noise_factor, lower = 0, upper = 1)
  assert_character(act_f_out)
  assert_character(act_f)
  assert_numeric(max_epochs)
  assert_numeric(patience)
  assert_numeric(validation_split, lower = 0, upper = 1)
  assert_logical(randomize_instances)
  assert_logical(rescale_features)
  assert_logical(init_logfile)
  assert_logical(recycle_settings)

  data_out = process_iucnn_input(x,lab = lab, mode = mode, outpath = '.', write_data_files = FALSE, verbose=0)

  dataset = data_out[[1]]
  labels = data_out[[2]]
  instance_names = data_out[[3]]


  if (recycle_settings == TRUE){
    new_logfile = sub('.txt','_new.txt',logfile)
    message(paste0("Copying the model-settings from the provided logfile. Model-testing results will be printed to new logfile ",new_logfile,". Any manual settings provided with this function are ignored. Set recycle_settings=FALSE to disable this behaviour."))

    # load model configurations from logfile and run with modifications
    model_configurations_df = read.csv(logfile,sep='\t')

    # init new logfile
    log_results(NaN,new_logfile,init_logfile=TRUE)

    for (row_id in 1:dim(model_configurations_df)[1]){
      print(paste0('Running model ',row_id,'/',dim(model_configurations_df)[1]))
      row = model_configurations_df[row_id,]
      mode = row$mode
      dropout_rate = row$dropout_rate
      seed = row$seed
      max_epochs = row$max_epochs
      patience = row$patience
      n_layers = row$n_layers
      use_bias = row$use_bias
      rescale_features = row$rescale_features
      randomize_instances = row$randomize_instances
      mc_dropout = row$mc_dropout
      mc_dropout_reps = row$mc_dropout_reps
      act_f = row$act_f
      act_f_out = row$act_f_out
      cv_fold = row$cv_fold
      validation_split = row$validation_split
      test_fraction = row$test_fraction
      label_stretch_factor = row$label_stretch_factor
      label_noise_factor = row$label_noise_factor

      res = train_iucnn(x = features,
                        lab = labels_train,
                        mode=mode,
                        path_to_output = '',
                        model_name = '',
                        validation_split = validation_split,
                        test_fraction = test_fraction,
                        cv_fold = cv_fold,
                        seed = seed,
                        max_epochs = max_epochs,
                        n_layers = n_layers,
                        use_bias = use_bias,
                        act_f = act_f,
                        act_f_out = act_f_out,
                        label_stretch_factor = label_stretch_factor,
                        patience = patience,
                        randomize_instances = randomize_instances,
                        dropout_rate = dropout_rate,
                        mc_dropout = mc_dropout,
                        mc_dropout_reps = mc_dropout_reps,
                        label_noise_factor = label_noise_factor,
                        rescale_features = rescale_features,
                        save_model = FALSE,
                        overwrite = TRUE,
                        verbose = 0
      )
      log_results(res,new_logfile,init_logfile=FALSE)
    }
    outfile = new_logfile
  }else{

    if (cv_fold > 1){
      message("Evaluating models using cross-validation. User-setting for test_fraction will be ignored. To run a single repitition with the specified test_fraction instead, set cv_fold=1.")
    }
    if (init_logfile == TRUE){
      log_results(NaN,logfile,init_logfile=init_logfile)
    }

    # cv_fold_list = as.list(cv_fold)
    # n_layers_list = as.list(n_layers)
    # dropout_rate_list = as.list(dropout_rate)
    # use_bias_list = as.list(use_bias)
    # seed_list = as.list(seed)
    # label_stretch_factor_list = as.list(label_stretch_factor)
    # label_noise_factor_list = as.list(label_noise_factor)
    # act_f_list = as.list(act_f)
    # act_f_out_list = as.list(act_f_out)
    # max_epochs_list = as.list(max_epochs)
    # patience_list = as.list(patience)
    # validation_split_list = as.list(validation_split)
    # test_fraction_list = as.list(test_fraction)
    # randomize_instances_list = as.list(randomize_instances)
    # rescale_features_list = as.list(rescale_features)

    # create all possible permutations of the values to test
    #reticulate::py_run_string("import itertools")
    #reticulate::py_run_string("import numpy as np")
    #reticulate::py_run_string("permutation = list(itertools.product(*[r.cv_fold_list,r.n_layers_list,r.dropout_rate_list,r.use_bias_list,r.seed_list,r.label_stretch_factor_list,r.label_noise_factor_list,r.act_f_list,r.act_f_out_list,r.max_epochs_list,r.patience_list,r.validation_split_list,r.test_fraction_list,r.randomize_instances_list,r.rescale_features_list]))")
    #permutations = reticulate::py$permutation

    permutations = do.call(expand.grid, list(cv_fold,n_layers,dropout_rate,use_bias,seed,label_stretch_factor,label_noise_factor,act_f,act_f_out,max_epochs,patience,validation_split,test_fraction,randomize_instances,rescale_features,mc_dropout,mc_dropout_reps,mode))
    n_permutations = dim(permutations)[1]

    message(paste0("Running model test for ",n_permutations," models. This can be quite time-intensive."))

    for (i in 1:n_permutations){
      print(paste0('Running model ',i,'/',n_permutations))
      settings = permutations[i,]
      cv_fold_i = as.integer(settings[[1]])
      n_layers_i = as.character(settings[[2]])
      dropout_rate_i = as.numeric(settings[[3]])
      use_bias_i = as.logical(settings[[4]])
      seed_i = as.integer(settings[[5]])
      label_stretch_factor_i = as.numeric(settings[[6]])
      label_noise_factor_i = as.numeric(settings[[7]])
      act_f_i = as.character(settings[[8]])
      act_f_out_i = as.character(settings[[9]])
      max_epochs_i = as.integer(settings[[10]])
      patience_i = as.integer(settings[[11]])
      validation_split_i = as.numeric(settings[[12]])
      test_fraction_i = as.numeric(settings[[13]])
      randomize_instances_i = as.logical(settings[[14]])
      rescale_features_i = as.logical(settings[[15]])
      mc_dropout_i = as.logical(settings[[16]])
      mc_dropout_reps_i = as.integer(settings[[17]])
      mode_i = as.character(settings[[18]])

      # set out act fun if chosen auto
      if (act_f_out_i == 'auto'){
        if (mode_i == 'nn-reg'){
          act_f_out_i  <-  'tanh'
        }else{
          act_f_out_i  <-  'softmax'
        }
      }

      # train model
      res = train_iucnn(x = features,
                        lab = labels_train,
                        mode = mode_i,
                        path_to_output = '',
                        model_name = '',
                        validation_split = validation_split_i,
                        test_fraction = test_fraction_i,
                        cv_fold = cv_fold_i,
                        seed = seed_i,
                        max_epochs = max_epochs_i,
                        n_layers = n_layers_i,
                        use_bias = use_bias_i,
                        act_f = act_f_i,
                        act_f_out = act_f_out_i,
                        label_stretch_factor = label_stretch_factor_i,
                        patience = patience_i,
                        randomize_instances = randomize_instances_i,
                        dropout_rate = dropout_rate_i,
                        mc_dropout = mc_dropout_i,
                        mc_dropout_reps = mc_dropout_reps_i,
                        label_noise_factor = label_noise_factor_i,
                        rescale_features = rescale_features_i,
                        save_model = FALSE,
                        overwrite = TRUE,
                        verbose = 0
      )
      log_results(res,logfile,init_logfile=FALSE)
    }
    outfile = logfile
  }

  model_testing_df = read.csv(outfile,sep='\t')
  return(model_testing_df)
}


