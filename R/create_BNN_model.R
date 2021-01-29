#' Set up BNN model structure
#' @export
#' @import reticulate

create_BNN_model <- function(feature_data,
                             n_nodes_list,
                             seed=1234,
                             use_class_weight=TRUE,
                             use_bias_node=TRUE,
                             prior = 1, # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
                             p_scale = 1, # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
                             init_std=0.1){ # st dev of the initial weights

  # source python function
  bn <- import("np_bnn")
  reticulate::py_install("https://github.com/dsilvestro/npBNN/archive/v0.1.4.tar.gz", pip = TRUE)
  #reticulate::source_python(system.file("python", "bnn_library.py", package = "IUCNN"))

  if(use_class_weight==TRUE){
    use_class_weight_switch = as.integer(1)
  }else{
    use_class_weight_switch = as.integer(0)
  }
  if(use_bias_node==TRUE){
    use_bias_node_switch = as.integer(1)
  }else{
    use_bias_node_switch = as.integer(0)
  }

  alphas = as.integer(c(0, 0))

  bnn_model = bn$npBNN(feature_data,
                    n_nodes = as.integer(n_nodes_list),
                    use_class_weights=use_class_weight_switch,
                    actFun=genReLU(prm=alphas, trainable=TRUE),
                    use_bias_node=use_bias_node_switch,
                    prior_f=as.integer(prior),
                    p_scale=as.integer(p_scale),
                    seed=as.integer(seed),
                    init_std=init_std)
  return(bnn_model)
}
