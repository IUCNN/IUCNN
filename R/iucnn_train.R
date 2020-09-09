

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
  reticulate::source_python('inst/IUCNN_train.py')



  # prepare input data for the python function
  # call python functio
  trained_model <- iucnn_train(dataset,
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
                               patience=500)

  # prepare output object

  # set class to output object

  # return
  #return(out)

}

