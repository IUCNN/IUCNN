
# Load python modules on load of R package
.onLoad <- function(libname, pkgname) {
  # numpy <<- reticulate::import("numpy", delay_load = TRUE)
  # tensorflow <<- reticulate::import("tensorflow", delay_load = TRUE)
  reticulate::configure_environment(pkgname)
}
