#' @export
#' @method summary iucnn_model
summary.iucnn_model <-  function(object,
                               ...) {
cat(sprintf("A model of type %s, trained on %s species and %s features.\n\n",
            object$model,
              length(object$input_data$test_labels),
              length(object$input_data$feature_names)))

cat(sprintf("Training accuracy: %s\n",
            round(object$training_accuracy, 3)))

cat(sprintf("Accuracy on unseen data: %s\n",
            round(object$validation_accuracy, 3)))

cat(sprintf("Label detail: %s Classes (%s)\n\n",
            length(object$input_data$label_dict),
            ifelse(length(object$input_data$label_dict) > 2,
                   "detailed",
                   "Threatened/Not Threatened")))

cat("Label representation\n")

maxlab = max(object$validation_predictions)
tel <- data.frame(0:maxlab,get_cat_count(object$validation_predictions,max_cat = maxlab))
names(tel) = c("Var1","Freq")
trl <- data.frame(0:maxlab,get_cat_count(object$validation_labels,max_cat = maxlab))
names(trl) = c("Var1","Freq")
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

#' @export
#' @method plot iucnn_model
#' @importFrom graphics abline legend points text
plot.iucnn_model <- function(x, ...){

  x$validation_accuracy_history
  x$training_accuracy_history
  x$final_training_epoch

  par(mfrow=c(rnd(x$cv_fold/2),2),mar = c(2, 2, 2, 2))
  for (i in 1:x$cv_fold){
    plot(x$training_loss_history[[i]], type = "n", ylab = "Loss", xlab = "Epoch",
         ylim = c(min(min(x$training_loss_history[[i]]), min(x$validation_loss_history[[i]])), max(max(x$training_loss_history[[i]]), max(x$validation_loss_history[[i]]))))
    points(x$training_loss_history[[i]], type = "b", col = "darkblue", pch = 1)
    points(x$validation_loss_history[[i]], type = "b", col = "darkred", pch = 2)
    abline(v = x$final_training_epoch[[i]], lty = 2)
    title(paste0('CV-fold ', i))
    legend(x = "topright",
           legend = c("Training", "Validation", "Final epoch"),
           col = c("darkblue", "darkred", "black"),
           lty = c(1, 1, 2),
           pch = c(1,2,NA),
           cex = 0.7)

  }
  par(mfrow = c(1,1))
}
