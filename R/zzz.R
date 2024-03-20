
# Load python modules on load of R package
.onLoad <- function(libname, pkgname) {
  # numpy <<- reticulate::import("numpy", delay_load = TRUE)
  # tensorflow <<- reticulate::import("tensorflow", delay_load = TRUE)
  reticulate::configure_environment(pkgname)
  if (isFALSE(reticulate::condaenv_exists())) {
    message(
      paste("Before using the package functions, follow the instrunctions",
            "on Instalation section at:", "https://github.com/IUCNN/IUCNN")
    )
  }
  have_numpy <- reticulate::py_module_available("numpy")
  if (isFALSE(have_numpy)) {
    message("numpy not available. download it using:
          reticulate::py_install('numpy==1.23.5')")
  }
  config <- reticulate::py_config()
  np_version <- as.numeric(gsub("\\.", "", config$numpy$version))
  if (np_version > 1235) {
    message("Incompatible numpy version available. Download it using:
          reticulate::py_install('numpy==1.23.5')")
  }
}

