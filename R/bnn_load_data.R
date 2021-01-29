#' Prepare feature data for input into Bayesian Neural Network (BNN)
#'
#' Transforms a data frame of features and category labels into the format required for running BNN
#' training or prediction. Optionally the data can be split into a training and test set.
#'
#' -
#'
#'@param features data.frame or filepath. A data.frame containing numeric feature values for a set of instances (rows),
#'e.g. output from geo_features(), biome_features(), and/or clim_features().
#'Alternatively this can be a filepath, in which case set "from_file" to TRUE.
#'@param labels list or filepath. A list of category labels for all instances present in the dataframe provided as features.
#'Alternatively this can be a filepath, in which case set "from_file" to TRUE.
#'@param seed integer. Set seed for random separation of data into training and test set.
#'@param testsize numeric. Determines the fraction of data that should be set aside as test set.
#'@param all_class_in_testset logical. Set to TRUE if all classes should be represented in the test set.
#'@param header logical. If TRUE, the function assumes that the dataframe contains column names (this matters mainly when loading from file).
#'@param instance_id logical. If TRUE, the function assumes that the first column contains the names of the instances.
#'@param from_file logical. If TRUE, the function will try to load the features from a txt file.
#'In that case provide filepaths as "features" and "labels" input.
#'
#'@return a list of features and labels, as well as other information (converted from a python dictionary)
#'
#' @keywords Feature preparation
#' @family Feature preparation
#'
#' @examples
#'
#'bnn_data = bnn_get_data(features,
#'                        labels,
#'                        seed=1234,
#'                        testsize=0.1,
#'                        all_class_in_testset=TRUE,
#'                        header=TRUE, # input data has a header
#'                        instance_id=TRUE, # input data includes names of instances
#'                        from_file=FALSE)
#' @export
#' @importFrom reticulate source_python


bnn_load_data <- function(features,
                          labels,
                          seed=1234,
                          testsize=0.1, # 10% test set
                          all_class_in_testset=TRUE,
                          header=TRUE, # input data has a header
                          instance_id=TRUE, # input data includes column with names of instances
                          from_file=FALSE){
  # source python function
  reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))
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
  dat = get_data(features,
                 labels,
                 seed=as.integer(seed),
                 testsize=testsize, # 10% test set
                 all_class_in_testset=all_class_in_testset_switch,
                 header=header_switch, # input data has a header
                 instance_id=instance_id_switch, # input data includes column with names of instances
                 from_file=from_file)
  return(dat)
}
