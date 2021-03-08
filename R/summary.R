#' @rdname summary
#' @export
summary.iucnn_model <-  function(x,
                               ...) {
cat(sprintf("A model of type %s, trained on %s species and %s features.\n\n",
              x$model,
              length(x$input_data$id_data),
              length(x$input_data$feature_names)))

cat(sprintf("Training accuracy: %s, test-accuracy: %s\n\n",
              round(x$training_accuracy, 3),
              round(x$test_accuracy, 3)))

cat(sprintf("Label detail: %s Classes (%s)\n\n",
              length(x$input_data$label_dict),
              ifelse(length(x$input_data$label_dict) > 2, "detailed", "Threatened/Not Threatened")))


cat("Confusion matrix:\n\n")

cm <- data.frame(x$confusion_matrix,
                 row.names = x$input_data$lookup.labels)

names(cm) <- x$input_data$lookup.labels

print(cm)
}

