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
