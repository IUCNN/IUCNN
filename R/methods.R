#' @export
#' @method summary iucnn_model
summary.iucnn_model <-  function(object,
                               ...) {
cat(sprintf("A model of type %s, trained on %s species and %s features.\n\n",
            object$model,
              length(object$input_data$id_data),
              length(object$input_data$feature_names)))

cat(sprintf("Training accuracy: %s\n",
            round(object$training_accuracy, 3)))

cat(sprintf("Test accuracy: %s\n",
            round(object$test_accuracy, 3)))

cat(sprintf("Validation accuracy: %s\n\n",
            round(object$validation_accuracy, 3)))

cat(sprintf("Label detail: %s Classes (%s)\n\n",
            length(object$input_data$label_dict),
            ifelse(length(object$input_data$label_dict) > 2,
                   "detailed",
                   "Threatened/Not Threatened")))

cat("Label representation\n")

tel <- data.frame(table(object$test_labels))
trl <- data.frame(table(object$input_data$labels))
lab <- merge(trl,tel, by = "Var1")

names(lab) <- c("Label", "Input_freq", "Test_freq")

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

  plot(x$training_loss_history, type = "n", ylab = "Loss", xlab = "Epoch",
       ylim = c(0, max(max(x$training_loss_history), max(x$validation_loss_history))))
  points(x$training_loss_history, type = "b", col = "darkblue", pch = 1)
  points(x$validation_loss_history, type = "b", col = "darkred", pch = 2)
  abline(v = x$final_training_epoch, lty = 2)
  legend(x = "topright",
    legend = c("Training", "Validation", "Final epoch"),
         col = c("darkblue", "darkred", "black"),
         lty = c(1, 1, 2),
         pch = c(1,2,NA),
         cex=1.5)
}
