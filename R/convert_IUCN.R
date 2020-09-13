#'Format IUCN Red List categories for IUCNN

convert_IUCN <- function(x,
                         species = "species",
                         labels = "labels",
                         accepted_labels = c("CR", "EN", "VU", "NT", "LC"),
                         level = "detail",
                         threatend = c("CR", "EN", "VU"),
                         not_threatened = c("NT", "LC")){

  if(is.list(x)){
    warning("x is list. Assuming input from rredlist")

    dat <- x$result %>%
      select(species = scientific_name, abels = category)
  }else{
    dat <- x %>% select(.data[[species]], .data[[labels]])
  }

  # remove DD and NE
  out <-  dat %>%
    filter(.data[[labels]] %in% accepted_labels)

  # if braod convert to broad categories

  # convert to numerical


  # Print summary to screen (e.g. how amny categories and the category coding)

  # return output
  return(out)

}


