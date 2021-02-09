#' Translate numeric labels predicted by trained model into IUCN conservation status
#' @export

translate_numeric_labels <- function(predicted_labels,
                                     label_order = c('LC','NT','VU','EN','CR')
                                     ){
  transformed_labels = predicted_labels
  transformed_labels[[2]][transformed_labels[[2]]==0]<-label_order[1]
  transformed_labels[[2]][transformed_labels[[2]]==1]<-label_order[2]
  transformed_labels[[2]][transformed_labels[[2]]==2]<-label_order[3]
  transformed_labels[[2]][transformed_labels[[2]]==3]<-label_order[4]
  transformed_labels[[2]][transformed_labels[[2]]==4]<-label_order[5]
  return(transformed_labels)
}
