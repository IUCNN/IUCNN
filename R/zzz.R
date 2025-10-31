
# Load python modules on load of R package
.onLoad <- function(libname, pkgname) {
  # numpy <<- reticulate::import("numpy", delay_load = TRUE)
  # tensorflow <<- reticulate::import("tensorflow", delay_load = TRUE)
  Sys.setenv(TF_USE_LEGACY_KERAS = "False")
  reticulate::configure_environment(pkgname)
}

# .onAttach <- function(libname, pkgname) {
#   if (isFALSE(reticulate::condaenv_exists())) {
#     packageStartupMessage(
#       paste("Before using the package functions, follow the instrunctions",
#             "on Instalation section at:", "https://github.com/IUCNN/IUCNN")
#     )
#   }
#   have_numpy <- reticulate::py_module_available("numpy")
#   if (isFALSE(have_numpy)) {
#     packageStartupMessage("numpy not available. download it using:
#           reticulate::py_install('numpy==1.23.5')")
#   }
#   config <- reticulate::py_config()
#   np_version <- as.numeric(gsub("\\.", "", config$numpy$version))
#   if (np_version > 1235) {
#     packageStartupMessage("Incompatible numpy version available. Download it using:
#           reticulate::py_install('numpy==1.23.5')")
#   }
# }
