#'Format IUCN Red List categories for IUCNN

prepare_labels <- function(x,
                         species = "species",
                         labels = "labels",
                         accepted_labels = c("CR", "EN", "VU", "NT", "LC"),
                         level = "detail",
                         threatened = c("CR", "EN", "VU")){

  if(is.list(x)){
    warning("x is list. Assuming input from rredlist")

    dat <- bind_rows(x[names(x) == "result"])%>%
      select(species = scientific_name, labels = category)
  }else{
    dat <- x %>% select(.data[[species]], .data[[labels]])
  }

  # remove DD and NE
  out <-  dat %>%
    filter(.data[[labels]] %in% accepted_labels)

  rem <- dat %>%
    filter(!.data[[labels]] %in% accepted_labels)

  if(nrow(rem) > 0){
    mis <- rem %>%
      select(.data[[labels]]) %>%
      distinct() %>%
      unlist()

    warning("Removed species with the follwoing labels: ", paste(mis, "\n"))
  }

  # if broad convert to broad categories
  if(level == "broad"){
    out <- out %>%
      mutate(lab.num.z = ifelse(.data[[labels]] %in% threatened, 0, 1))
  }else{
    lookup <- data.frame(IUCN = accepted_labels,
                         lab.num.z = seq(0, (length(accepted_labels)-1)))

    names(lookup) <- c(labels, "lab.num.z")

    out <- out %>%
      left_join(lookup, by = labels)
    }

  out <- out %>%
    dplyr::select(species = .data[[species]], labels = lab.num.z)

  # return output
  return(out)

}
