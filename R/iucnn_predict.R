

iucnn_predict <- function(x,
                          model_dir,
                          verbose = 0,
                          return_prob = FALSE){

  # complete cases only
  tmp <- x[complete.cases(x),]

  if(nrow(tmp) != nrow(x)){
    mis <- x[!complete.cases(x),]
    warning("Information for species was incomplete, species removed\n", paste(mis$species, "\n"))
  }

  #prepare input data
  tmp <- tmp %>%
    dplyr::select(-species)

  # source python function
  reticulate::source_python('inst/python/IUCNN_predict.py')

  # run predict function
  out <- iucnn_predict(feature_set = as.matrix(tmp),
                       model_dir = model_dir,
                       verbose = verbose,
                       return_prob = return_prob)

  #return output object
  return(out)
}


