#' Translate numeric labels predicted by trained model into IUCN conservation status
#' @export

translate_numeric_labels <- function(predicted_labels){
  transformed_labels = predicted_labels
  transformed_labels[[2]][transformed_labels[[2]]==0]<-'CR'
  transformed_labels[[2]][transformed_labels[[2]]==1]<-'EN'
  transformed_labels[[2]][transformed_labels[[2]]==2]<-'VU'
  transformed_labels[[2]][transformed_labels[[2]]==3]<-'NT'
  transformed_labels[[2]][transformed_labels[[2]]==4]<-'LC'
  return(transformed_labels)
}
