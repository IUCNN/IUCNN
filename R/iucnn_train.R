

train_iucnn <- function(x,
                        labels,
                        path_to_output="",
                        validation_split=0.1,
                        test_fraction=0.1,
                        seed=1234,
                        verbose=0, #can be 0, 1, 2
                        model_name="iuc_nn_model",
                        max_epochs=1000,
                        n_layers=c(60,60,20),
                        use_bias=1,
                        act_f="relu",
                        patience=500){


  # merge speces and labels to match order
  tmp <- left_join(x, labels, by = "species")

  # complete cases only

  # prepare input data for the python function
  dataset <- tmp %>%
    select(-species, -labels)

  labels <- tmp %>%
    select(labels) %>%
    mutate(labels = labels - 1)

  # source python function
  reticulate::source_python('inst/python/IUCNN_train.py')

  # call python functio
  trained_model <- iucnn_train(as.matrix(dataset),
                               as.matrix(labels))


  # only take complete cases for now

    # load python function
  #reticulate::install_miniconda()
  reticulate::py_install("tensorflow==2.0.0", pip = TRUE)



  # prepare output object

  # set class to output object

  # return
  #return(out)

}
