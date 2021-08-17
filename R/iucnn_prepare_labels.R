#'Format IUCN Red List categories for IUCNN
#'
#'Converting IUCN category labels into numeric categories required by \code{\link{iucnn_train_model}}.
#'
#'
#'@param x a data.frame or a list. If a data.frame,
#'two columns with the species names and IUCN categories
#'respectively. The column names are defined by the
#'species and labels arguments. If a list, expecting
#'the format as returned by \link[rredlist]{rl_search}.
#'@param y object of class \code{iucnn-features} or \code{iucnn_cnn_features}.
#' Ensures that the species in the return value are in the same order as in y.
#'@param species a character string. The name of the
#' column with the species names in x.
#'@param labels a character string. The name of the
#'column with the labels (assessment categories) in x.
#'@param accepted_labels a character string. The labels
#'to be converted in to numeric values.
#'Entries with labels not mentioned (e.g. "DD") will be removed.
#'The numeric labels returned by the
#'function will correspond to the order in this argument.
#' For instance with default settings, LC -> 0, CR -> 4.
#'@param level a character string. The level of output
#'level detail. If "detail"
#'full IUCN categories, if "broad" then
#' 0 = Not threatened, and 1 = Threatened.
#'@param threatened a character string. Only if level=="broad",
#'Which labels to consider threatened.
#'
#'@note See \code{vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")}
#'for a tutorial on how to run IUCNN.
#'
#'@return a data.frame with species names and numeric labels
#'
#' @examples
#' dat <- data.frame(species = c("A","B"),
#'                   decimallongitude = runif (200,10,15),
#'                   decimallatitude = runif (200,-5,5))
#'
#' labs <- data.frame(species = c("A","B"),
#'                    labels = c("CR", "LC"))
#'
#' features <- iucnn_prepare_features(dat,
#'                                    type = "geographic")
#'
#' iucnn_prepare_labels(x = labs,
#'                      y = features)
#'
#' @export
#' @importFrom magrittr %>%
#' @importFrom dplyr arrange bind_rows distinct filter left_join mutate select
#' @importFrom checkmate assert_character


iucnn_prepare_labels <- function(x,
                                 y,
                                 species = "species",
                                 labels = "labels",
                                 accepted_labels = c('LC','NT','VU','EN','CR'),
                                 level = "detail",
                                 threatened = c("CR", "EN", "VU")){

 # assertions
  assert_character(species)
  assert_character(labels)
  assert_character(accepted_labels)
  assert_character(level)
  assert_character(threatened)

 # check for input from rredlist
  if(is.list(x) & !is.data.frame(x)){
    warning("x is list. Assuming input from rredlist")

    dat <- bind_rows(x[names(x) == "result"])%>%
      dplyr::select(species = .data$scientific_name, labels = .data$category)
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

    warning("Removed species with the following labels: ", paste(mis, "\n"))
  }

  # if broad convert to broad categories
  if(level == "broad"){
    out <- out %>%
      mutate(lab.num.z = ifelse(.data[[labels]] %in% threatened, 1, 0))
    lookup <- data.frame(labels = c("Not Threatened", "Threatened"),
                         lab.num.z = c(0,1))
  }else{
    lookup <- data.frame(IUCN = accepted_labels,
                         lab.num.z = seq(0, (length(accepted_labels)-1)))

    names(lookup) <- c(labels, "lab.num.z")

    out <- out %>%
      left_join(lookup, by = labels)
  }

  names(lookup) <- c("labels", "lab.num.z")

  out <- out %>%
    dplyr::select(species = .data[[species]], labels = .data$lab.num.z)

  # match labels to y if supplied
  if(!is.null(y)){
    # if y are cnn features
    if("cnn_features" %in% class(y)){
      # if not all species are there, crop
      if(!all(names(y) %in% out$species)|
         !all(out$species %in% names(y))){

        out_in <- out
        out <- out %>%
          filter(species %in% names(y))

        y <- y[names(y) %in% out_in$species]
        warning("species mismatch between x and y. Species removed from output.")
      }
      # sort output
      out <- out %>%
        arrange(ordered(species, names(y)))
      #if y are iucnn features
    }else if("iucnn_features" %in% class(y)){
      # if not all species are there, crop
      if(!all(y$species %in% out$species)|
         !all(out$species %in% y$species)){

        out_in <- out
        out <- out %>%
          filter(species %in% y$species)

        y <- y[y$species %in% out_in$species, ]
        warning("species mismatch between x and y. Species removed from output.")
      }
      # sort output
      out <- out %>%
        arrange(ordered(species, y$species))
    }
  }

  # create output object
  out <- list(
    labels = out,
    lookup = lookup
  )

  # set class
  class(out) <- "iucnn_labels"

  # return output
  return(out)
}
