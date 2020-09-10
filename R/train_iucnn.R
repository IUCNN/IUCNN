

train_iucnn <- function(x,
                        labels,
                        path_to_output="",
                        model_name = "iuc_nn_model",
                        validation_split = 0.1,
                        test_fraction = 0.1,
                        seed = 1234,
                        verbose = 0, #can be 0, 1, 2
                        max_epochs = 1000,
                        n_layers = c(60,60,20),
                        use_bias = 1,
                        act_f = "relu",
                        patience = 500){

  # Check input
  if(!"species" %in% names(x)){
    stop("species column not found in x.
         The features input need a column named 'species'
         with the species names matching those in labels")
  }

  if(!"species" %in% names(labels)){
    stop("species column not found in labels.
         The label input need a column named 'species'
         with the species names matching those in x")
  }

  if(!"species" %in% names(labels)){
    stop("labels column not found in labels.
         The label input need a column named 'labels'")
  }

  # merge speces and labels to match order
  tmp.in <- left_join(x, labels, by = "species")

  if(nrow(tmp.in) != nrow(x)){
    mis <- x$species[!x$species %in% tmp$species]
    warning("Labels for species not found, species removed.\n", paste(mis, "\n"))
  }

  if(nrow(tmp.in) != nrow(labels)){
    mis <- labels$species[!labels$species %in% tmp$species]
    warning("Labels for species not found, species removed.\n", paste(mis, "\n"))
  }

  # complete cases only
  tmp <- tmp.in[complete.cases(tmp.in),]

  if(nrow(tmp) != nrow(tmp.in)){
    mis <- tmp.in[!complete.cases(tmp.in),]
    warning("Information for species was incomplete, species removed\n", paste(mis$species, "\n"))
  }

  # prepare input data for the python function
  dataset <- tmp %>%
    dplyr::select(-species, -labels)

  labels <- tmp %>%
    dplyr::select(labels)

  # prepare labels to start at 0
  if(min(labels$labels) != 0){
    warning(sprintf("Labels need to start at 0. Labels substracted with %s", min(labels$labels)))

    labels <-  labels %>%
      dplyr::mutate(labels = labels - min(.data$labels))
  }

  labels <- labels %>%
    mutate(labels = labels - 1)

  # source python function
  reticulate::source_python('inst/python/IUCNN_train.py')

  # run model via python script
  iucnn_train(dataset = as.matrix(dataset),
              labels = as.matrix(labels),
              path_to_output = path_to_output,
              validation_split = validation_split,
              test_fraction = test_fraction,
              seed = as.integer(seed),
              verbose = verbose,
              max_epochs = as.integer(max_epochs),
              n_layers = n_layers,
              use_bias = use_bias,
              act_f = act_f,
              patience = patience
              )

  #   # load python function
  # reticulate::py_install("tensorflow==2.0.0", pip = TRUE)
}
