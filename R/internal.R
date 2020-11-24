
get_footp <- function(x, file_path){
  test <- file.exists(file.path(file_path, paste("HFP", x, ".tif", sep = "")))
  if(!test){
    download.file(paste("https://wcshumanfootprint.org/data/HFP", x, ".zip", sep = ""),
                  destfile = file.path(file_path, paste("HFP", x, ".zip", sep = "")))

    unzip(file.path(file_path, paste("HFP", x, ".zip", sep = "")),
          exdir = file_path)

    file.remove(file.path(file.path(file_path, paste("HFP", x, ".zip", sep = ""))))
    }
}
