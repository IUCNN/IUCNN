#' @export
#' @method summary iucnn_model
summary.iucnn_model <-  function(object,
                                 ...) {
  cat(sprintf("A model of type %s, trained on %s species and %s features.\n\n",
              object$model,
              length(object$input_data$labels),
              length(object$input_data$feature_names)))

  cat(sprintf("Training accuracy: %s\n",
              round(object$training_accuracy, 3)))
  cat(sprintf("Validation accuracy: %s\n",
              round(object$validation_accuracy, 3)))

  if (object$test_fraction > 0.0) {
    cat(sprintf("Accuracy on unseen data (test set): %s\n\n",
                round(object$test_accuracy, 3)))

    cat(sprintf("Label detail: %s Classes (%s)\n\n",
                length(object$input_data$label_dict),
                ifelse(length(object$input_data$label_dict) > 2,
                       "detailed",
                       "Threatened/Not Threatened")))

    cat("Label representation\n")

    maxlab <- max(object$test_predictions)
    tel <- data.frame(0:maxlab,
                      get_cat_count(object$test_predictions,
                                    max_cat = maxlab))
    names(tel) <- c("Var1","Freq")
    trl <- data.frame(0:maxlab,
                      get_cat_count(object$test_labels,
                                    max_cat = maxlab))
    names(trl) <- c("Var1","Freq")
    lab <- merge(trl,tel, by = "Var1")

    names(lab) <- c("Label", "Input_count", "Estimated_count")

    print(lab)
    cat("\n")

    cat("Confusion matrix (rows test data and columns predicted):\n")

    cm <- data.frame(object$confusion_matrix,
                     row.names = object$input_data$lookup.labels)

    names(cm) <- object$input_data$lookup.labels

    print(cm)
  }

}

#' @export
#' @method plot iucnn_model
#' @importFrom graphics abline legend par points plot text title
plot.iucnn_model <- function(x, ...){

  par_prev <- par()

  if(x$cv_fold > 1){
    par(mfrow = c(rnd(x$cv_fold/2),2),
        mar = c(2, 2, 2, 2))
  }

  for (i in 1:x$cv_fold){
    if (x$model == 'bnn-class'){
      plot(x$training_loss_history[[i]],
           type = "n",
           ylab = "Loss",
           xlab = "MCMC-it",
           ylim = c(min(x$training_loss_history[[i]]),
                    max(x$training_loss_history[[i]])))

      points(x$training_loss_history[[i]],
             type = "b",
             col = "darkblue",
             pch = 1)
      title(paste0('CV-fold ', i))
    }else{
      if (is.nan(x$validation_loss_history[[i]][1])){
        plot(x$training_loss_history[[i]],
             type = "n",
             ylab = "Loss",
             xlab = "Epoch",
             ylim = c(min(x$training_loss_history[[i]]),
                      max(x$training_loss_history[[i]])))

        points(x$training_loss_history[[i]],
               type = "b",
               col = "darkblue",
               pch = 1)
        title(paste0('CV-fold ', i))
      }else{
        plot(x$training_loss_history[[i]],
             type = "n",
             ylab = "Loss",
             xlab = "Epoch",
             ylim = c(min(min(x$training_loss_history[[i]]),
                          min(x$validation_loss_history[[i]])),
                      max(max(x$training_loss_history[[i]]),
                          max(x$validation_loss_history[[i]]))))

        points(x$training_loss_history[[i]],
               type = "b",
               col = "darkblue",
               pch = 1)
        points(x$validation_loss_history[[i]],
               type = "b",
               col = "darkred",
               pch = 2)
        abline(v = x$final_training_epoch[[i]],
               lty = 2)
        title(paste0('CV-fold ', i))
        legend(x = "topright",
               legend = c("Training", "Validation", "Final epoch"),
               col = c("darkblue", "darkred", "black"),
               lty = c(1, 1, 2),
               pch = c(1, 2, NA),
               cex = 0.7)
      }

    }
    }

  par(mfrow = par_prev$mfrow,
      mar = par_prev$mar)
}


#' @export
#' @method plot iucnn_predictions
#' @importFrom graphics abline barplot par
plot.iucnn_predictions <- function(x, ...){

  # count the different categories
  counts <- table(x$class_predictions) # this doens't count NaN
  NA_count <- length(x$class_predictions[is.na(x$class_predictions)])
  counts['NA'] <- NA_count
  # set colors for relevant categories
  if( all(nchar(names(counts)) == 2)){
    cats <- c('LC', 'NT', 'VU', 'EN' ,'CR', 'NA')
    colors <- c('#60C659',
                '#CCE226',
                '#F9E814',
                '#FC7F3F',
                '#D81E05',
                '#C1B5A5')
  }else{
    cats <- c('Not Threatened', 'Threatened')
    colors <-  c('lightgreen', 'orange')
    names(colors) <- c('Not Threatened', 'Threatened')
  }
  # check if any category is missing
  mis <- cats[!cats %in% names(counts)]

  plo <- c(counts, rep(0, length(mis)))
  names(plo) <- c(names(counts), mis)

  # order categories
  plo <- plo[cats]

  # plot
  barplot(plo,
          col = colors,
          main = "Number of species per IUCN category")

  abline(v = 2.5, lty = 2)

}

#' @export
#' @method plot iucnn_featureimportance
#' @importFrom graphics barplot segments
#' @importFrom grDevices cm.colors

plot.iucnn_featureimportance <- function(x, ...){
  bp <- barplot(height = x$feat_imp_mean,
          names.arg = x$feature_block,
          ylim = c(0, max(x$feat_imp_mean) + max(x$feat_imp_std)),
          main = "Delta accuracy",
          col = rev(cm.colors(length(x$feature_block))))

  segments(bp,
           x$feat_imp_mean - x$feat_imp_std,
           bp,
           x$feat_imp_mean + x$feat_imp_std,
           lwd = 1.5)
}



