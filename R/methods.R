#' @export
#' @method summary iucnn_model
summary.iucnn_model <-  function(object,
                               ...) {
cat(sprintf("A model of type %s, trained on %s species and %s features.\n\n",
            object$model,
              length(object$input_data$id_data),
              length(object$input_data$feature_names)))

cat(sprintf("Training accuracy: %s, test-accuracy: %s\n\n",
              round(object$training_accuracy, 3),
              round(object$test_accuracy, 3)))

cat(sprintf("Label detail: %s Classes (%s)\n\n",
              length(object$input_data$label_dict),
              ifelse(length(object$input_data$label_dict) > 2, "detailed", "Threatened/Not Threatened")))


cat("Confusion matrix (rows test data and columns predicted):\n\n")

cm <- data.frame(object$confusion_matrix,
                 row.names = object$input_data$lookup.labels)

names(cm) <- object$input_data$lookup.labels

print(cm)
}

#' @export
#' @method plot iucnn_model
#' @importFrom graphics legend points
plot.iucnn_model <- function(x, ...){

  plot(x$training_loss_history, type = "n", ylab = "Loss", xlab = "Epoch")
  points(x$training_loss_history, type = "b", col = "darkblue", pch = 1)
  points(x$validation_loss_history, type = "b", col = "darkred", pch = 2)
  legend(x = "topright",
    legend = c("Training", "Validation"),
         col=c("darkblue", "darkred"),
         lty=1,
         pch = 1:2,
         cex=1.5)
  # lattice::levelplot(x$confusion_matrix,
  #                    scales=list(x=list(at = x$input_data$lookup.lab.num.z + 1,
  #                                       labels = x$input_data$lookup.labels),
  #                                y=list(at = x$input_data$lookup.lab.num.z + 1,
  #                                       labels = x$input_data$lookup.labels)),
  #                    xlab = "Predicted",
  #                    ylab = "Test data",
  #                    main = "B) Confusion matrix, test data")
}
