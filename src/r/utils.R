install_or_load <- function(packages) {
  if (!is.list(packages) || !is.vector(packages)) {
    packages <- list(packages)
  }
  for (package in packages){
    if (!require(package)){
      install.packages(package, dependencies = TRUE)
      require(package)
    }
  }
}