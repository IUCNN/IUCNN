library(tidyverse)
library(IUCNN)
library(devtools)
document()

run_feature_importance <- function(features_tmp,labels_tmp,modeltest_logfile,n_permutations=100){
  model_testing_results_tmp = read.csv(modeltest_logfile,sep='\t')
  eval_obj_tmp = evaluate_iucnn(model_testing_results_tmp,force_dropout = TRUE)
  iucnn_results_tmp = train_iucnn(features_tmp,
                                  labels_tmp,
                                  path_to_output = 'iucnn_orchid_nn_tmp',
                                  best_model = eval_obj_tmp$best_model
  )
  feature_importance_out = feature_importance(x = iucnn_results_tmp,feature_blocks = as.list(names(features_tmp)),n_permutations=n_permutations)
  return(feature_importance_out)
}

# calculate training features________
# # load occurrence data
# load('orchid_data_iucnn/orchid_original_training_occurrences.rda')
# # convert into table for input into IUCNN
# species = paste(training_occs$genus,training_occs$epitheton,sep = ' ')
# decimallongitude = training_occs$decimallongitude
# decimallatitude = training_occs$decimallatitude
# occurrence_df = tibble(species,decimallongitude,decimallatitude)
# # calculate features
# features = prep_features(occurrence_df)
# features_file = "orchid_data_iucnn/features.RData"
# save(features, file = features_file)

# calculate prediction features________
# # load occurrence data
# load('orchid_data_iucnn/orchid_original_prediction_occurrence.rda')
# # convert into table for input into IUCNN
# species = paste(prediction_occs$genus,prediction_occs$epitheton,sep = ' ')
# decimallongitude = prediction_occs$decimallongitude
# decimallatitude = prediction_occs$decimallatitude
# occurrence_df = tibble(species,decimallongitude,decimallatitude)
# features_pred = prep_features(occurrence_df)
# features_file_pred = "orchid_data_iucnn/features_pred.RData"
# save(features_pred, file = features_file_pred)


# load the compiled features
features_file = "orchid_data_iucnn/features.RData"
features_file_pred = "orchid_data_iucnn/features_pred.RData"
load(features_file)
load(features_file_pred)
# load labels
status_info = read_csv('orchid_data_iucnn/original_orchid_training_labels.csv')
names(status_info) = c("species", "labels")
labels_train_detail <- prep_labels(status_info,level = 'detail')
labels_train_broad <- prep_labels(status_info,level = 'broad')

# define feature indices
species_names = 1
geo = 2:10
biome = 11:26
bioclim = 27:38
hfp = 39:46
all = 2:46

# select feature scenario to run
feature_scenario = 'all'
if (feature_scenario == 'custom'){
  # load modeltest results for model trained on all features
  modeltest_logfile = "model_testing_detailed_nn_class_all.txt"
  labels_tmp = labels_train_detail
  features_tmp = features
  feature_importance_out = run_feature_importance(features_tmp,labels_tmp,modeltest_logfile,n_permutations = 100)
  write.table(feature_importance_out,'feature_importance_results_detailed_class.txt',col.names = TRUE, row.names = FALSE, quote = FALSE, sep='\t')
  # modeltest_logfile = "model_testing_broad_nn_class_all.txt"
  # labels_tmp = labels_train_broad
  # features_tmp = features
  # feature_importance_out = run_feature_importance(features_tmp,labels_tmp,modeltest_logfile,n_permutations = 100)
  best_features = feature_importance_out$feature_block[feature_importance_out$feat_imp_mean>0.0]
  selected_features = features[,c('species',best_features)]
  selected_features_pred = features_pred[,c('species',best_features)]
}else if(feature_scenario == 'all'){
  selected_features = features[,c(species_names,all)]
  selected_features_pred = features_pred[,c(species_names,all)]
}else if(feature_scenario == 'geo'){
  selected_features = features[,c(species_names,geo)]
  selected_features_pred = features_pred[,c(species_names,geo)]
}else if(feature_scenario == 'geo_hfp'){
  selected_features = features[,c(species_names,geo,hfp)]
  selected_features_pred = features_pred[,c(species_names,geo,hfp)]
}



# 1. TRAINING____________________________________________________

# 1.1 DETAILED LABELS_____________________________________________

# 1.1.1 CLASS_______________________________________________________
# define logfile to store info of different runs
logfile_detail_class = paste0("model_testing_detailed_nn_class_",feature_scenario,".txt")
model_testing_results_detail_class = modeltest_iucnn(selected_features,
                                                   labels_train_detail,
                                                   logfile_detail_class,
                                                   seed = 1234,
                                                   dropout_rate = c(0.,0.1,0.3),
                                                   n_layers = c('30','40_20','50_30_10'),
                                                   cv_fold = 5,
                                                   validation_fraction = 0.,
                                                   mode = 'nn-class',
                                                   init_logfile = TRUE,
                                                   recycle_settings = FALSE)

#model_testing_results_detail_class = read.csv(logfile_detail_class,sep='\t')
# evaluate best model
eval_obj_detail_class = evaluate_iucnn(model_testing_results_detail_class,force_dropout = TRUE, write_file = TRUE, outfile = paste0("model_evaluation_best_model_detailed_nn_class_",feature_scenario,".txt"))
# calculate accuracy threshold table for best model
acc_thres_tbl_detail_class = get_accthres_table(selected_features,labels_train_detail,eval_obj_detail_class$best_model)
# train final model on all data
iucnn_results_detail_class = train_iucnn(selected_features,
                                       labels_train_detail,
                                       path_to_output = paste0('iucnn_orchid_nn_class_detail_',feature_scenario),
                                       best_model = eval_obj_detail_class$best_model
)


# 1.1.2 REG_________________________________________________________
# define logfile to store info of different runs
logfile_detail_reg = paste0("model_testing_detailed_nn_reg_",feature_scenario,".txt")
model_testing_results_detail_reg = modeltest_iucnn(selected_features,
                                        labels_train_detail,
                                        logfile_detail_reg,
                                        seed = 1234,
                                        dropout_rate = c(0.3),
                                        n_layers = c('50_50_50_50_50','50_40_30_20_10_5','50_50_50_50_50_50_50'),
                                        cv_fold = 5,
                                        validation_fraction = 0.,
                                        mode = 'nn-reg',
                                        act_f_out = c('tanh'),
                                        label_stretch_factor = c(1.2),
                                        label_noise_factor = 0.0,
                                        init_logfile = FALSE,
                                        recycle_settings = FALSE)

#model_testing_results_detail_reg = read.csv(logfile_detail_reg,sep='\t')
# evaluate best model
eval_obj_detail_reg = evaluate_iucnn(model_testing_results_detail_reg,force_dropout = TRUE, write_file = TRUE, outfile = paste0("model_evaluation_best_model_detailed_nn_reg_",feature_scenario,".txt"))
# calculate accuracy threshold table for best model
acc_thres_tbl_detail_reg = get_accthres_table(selected_features,labels_train_detail,eval_obj_detail_reg$best_model)
# train final model on all data
iucnn_results_detail_reg = train_iucnn(selected_features,
                             labels_train_detail,
                             path_to_output = paste0('iucnn_orchid_nn_reg_detail_',feature_scenario),
                             best_model = eval_obj_detail_reg$best_model
                            )



# 1.2 BROAD LABELS_____________________________________________

# 1.2.1 CLASS_______________________________________________________
# define logfile to store info of different runs
logfile_broad_class = paste0("model_testing_broad_nn_class_",feature_scenario,".txt")
model_testing_results_broad_class = modeltest_iucnn(selected_features,
                                                     labels_train_broad,
                                                     logfile_broad_class,
                                                     seed = 1234,
                                                     dropout_rate = c(0.,0.1,0.3),
                                                     n_layers = c('30','40_20','50_30_10'),
                                                     cv_fold = 5,
                                                     validation_fraction = 0.,
                                                     mode = 'nn-class',
                                                     init_logfile = TRUE,
                                                     recycle_settings = FALSE)

#model_testing_results_broad_class = read.csv(logfile_broad_class,sep='\t')
# evaluate best model
eval_obj_broad_class = evaluate_iucnn(model_testing_results_broad_class,force_dropout = TRUE, write_file = TRUE, outfile = paste0("model_evaluation_best_model_broad_nn_class_",feature_scenario,".txt"))
# calculate accuracy threshold table for best model
acc_thres_tbl_broad_class = get_accthres_table(selected_features,labels_train_broad,eval_obj_broad_class$best_model)
# train final model on all data
iucnn_results_broad_class = train_iucnn(selected_features,
                                         labels_train_broad,
                                         path_to_output = paste0('iucnn_orchid_nn_class_broad_',feature_scenario),
                                         best_model = eval_obj_broad_class$best_model
)


# 1.2.2 REG_________________________________________________________
# define logfile to store info of different runs
logfile_broad_reg = paste0("model_testing_broad_nn_reg_",feature_scenario,".txt")
model_testing_results_broad_reg = modeltest_iucnn(selected_features,
                                                   labels_train_broad,
                                                   logfile_broad_reg,
                                                   seed = 1234,
                                                   dropout_rate = c(0.,0.1,0.3),
                                                   n_layers = c('30','40_20','50_30_10'),
                                                   cv_fold = 5,
                                                   validation_fraction = 0.,
                                                   mode = 'nn-reg',
                                                   act_f_out = c('tanh','sigmoid'),
                                                   label_stretch_factor = c(0.8,1.0,1.2),
                                                   label_noise_factor = 0.0,
                                                   init_logfile = TRUE,
                                                   recycle_settings = FALSE)

#model_testing_results_broad_reg = read.csv(logfile_broad_reg,sep='\t')
# evaluate best model
eval_obj_broad_reg = evaluate_iucnn(model_testing_results_broad_reg,force_dropout = TRUE, write_file = TRUE, outfile = paste0("model_evaluation_best_model_broad_nn_reg_",feature_scenario,".txt"))
# calculate accuracy threshold table for best model
acc_thres_tbl_broad_reg = get_accthres_table(selected_features,labels_train_broad,eval_obj_broad_reg$best_model)
# train final model on all data
iucnn_results_broad_reg = train_iucnn(selected_features,
                                       labels_train_broad,
                                       path_to_output = paste0('iucnn_orchid_nn_reg_broad_',feature_scenario),
                                       best_model = eval_obj_broad_reg$best_model
)


# 2. PREDICTIONS_______________________________________________

# 2.1.1 DETAIL CLASS_________________________________________________
predictions_detail_class = predict_iucnn(selected_features_pred,iucnn_results_detail_class)
predictions_detail_class_60_acc = predict_iucnn(selected_features_pred,iucnn_results_detail_class,acc_thres_tbl_detail_class,target_acc = 0.60)
predictions_detail_class_70_acc = predict_iucnn(selected_features_pred,iucnn_results_detail_class,acc_thres_tbl_detail_class,target_acc = 0.70)
predictions_detail_class_75_acc = predict_iucnn(selected_features_pred,iucnn_results_detail_class,acc_thres_tbl_detail_class,target_acc = 0.75)
pdf(paste0('plots/predictions_detail_class_',feature_scenario,'.pdf'))
plot_predictions(predictions_detail_class$predictions,title="Detailed, iucnn-class")
plot_predictions(predictions_detail_class_60_acc$predictions,title="Detailed, iucnn-class, 60% acc")
plot_predictions(predictions_detail_class_70_acc$predictions,title="Detailed, iucnn-class, 70% acc")
plot_predictions(predictions_detail_class_75_acc$predictions,title="Detailed, iucnn-class, 75% acc")
dev.off()

# 2.1.2 DETAIL REG_________________________________________________
predictions_detail_reg = predict_iucnn(selected_features_pred,iucnn_results_detail_reg)
predictions_detail_reg_60_acc = predict_iucnn(selected_features_pred,iucnn_results_detail_reg,acc_thres_tbl_detail_reg,target_acc = 0.60)
predictions_detail_reg_70_acc = predict_iucnn(selected_features_pred,iucnn_results_detail_reg,acc_thres_tbl_detail_reg,target_acc = 0.70)
predictions_detail_reg_75_acc = predict_iucnn(selected_features_pred,iucnn_results_detail_reg,acc_thres_tbl_detail_reg,target_acc = 0.75)
pdf(paste0('plots/predictions_detail_reg_',feature_scenario,'.pdf'))
plot_predictions(predictions_detail_reg$predictions,title="Detailed, iucnn-reg")
plot_predictions(predictions_detail_reg_60_acc$predictions,title="Detailed, iucnn-reg, 60% acc")
plot_predictions(predictions_detail_reg_70_acc$predictions,title="Detailed, iucnn-reg, 70% acc")
plot_predictions(predictions_detail_reg_75_acc$predictions,title="Detailed, iucnn-reg, 75% acc")
dev.off()

# 2.2.1 BROAD CLASS_________________________________________________
predictions_broad_class = predict_iucnn(selected_features_pred,iucnn_results_broad_class)
predictions_broad_class_70_acc = predict_iucnn(selected_features_pred,iucnn_results_broad_class,acc_thres_tbl_broad_class,target_acc = 0.70)
predictions_broad_class_80_acc = predict_iucnn(selected_features_pred,iucnn_results_broad_class,acc_thres_tbl_broad_class,target_acc = 0.80)
predictions_broad_class_90_acc = predict_iucnn(selected_features_pred,iucnn_results_broad_class,acc_thres_tbl_broad_class,target_acc = 0.90)
pdf(paste0('plots/predictions_broad_class_',feature_scenario,'.pdf'))
plot_predictions(predictions_broad_class$predictions,title="Broad, iucnn-class")
plot_predictions(predictions_broad_class_70_acc$predictions,title="Broad, iucnn-class, 70% acc")
plot_predictions(predictions_broad_class_80_acc$predictions,title="Broad, iucnn-class, 80% acc")
plot_predictions(predictions_broad_class_90_acc$predictions,title="Broad, iucnn-class, 90% acc")
dev.off()

# 2.2.2 BROAD REG_________________________________________________
predictions_broad_reg = predict_iucnn(selected_features_pred,iucnn_results_broad_reg)
predictions_broad_reg_70_acc = predict_iucnn(selected_features_pred,iucnn_results_broad_reg,acc_thres_tbl_broad_reg,target_acc = 0.70)
predictions_broad_reg_80_acc = predict_iucnn(selected_features_pred,iucnn_results_broad_reg,acc_thres_tbl_broad_reg,target_acc = 0.80)
predictions_broad_reg_90_acc = predict_iucnn(selected_features_pred,iucnn_results_broad_reg,acc_thres_tbl_broad_reg,target_acc = 0.90)
pdf(paste0('plots/predictions_broad_reg_',feature_scenario,'.pdf'))
plot_predictions(predictions_broad_reg$predictions,title="Broad, iucnn-reg")
plot_predictions(predictions_broad_reg_70_acc$predictions,title="Broad, iucnn-reg, 70% acc")
plot_predictions(predictions_broad_reg_80_acc$predictions,title="Broad, iucnn-reg, 80% acc")
plot_predictions(predictions_broad_reg_90_acc$predictions,title="Broad, iucnn-reg, 90% acc")
dev.off()

all_pred_detail = data.frame(
                            predictions_detail_class$names,
                            predictions_detail_class$predictions,
                            predictions_detail_class_60_acc$predictions,
                            predictions_detail_class_70_acc$predictions,
                            predictions_detail_class_75_acc$predictions,
                            predictions_detail_reg$predictions,
                            predictions_detail_reg_60_acc$predictions,
                            predictions_detail_reg_70_acc$predictions,
                            predictions_detail_reg_75_acc$predictions
                            )
names(all_pred_detail) = c('species','pred_class','pred_class_acc_70','pred_class_acc_80','pred_class_acc_90','pred_reg','pred_reg_acc_70','pred_reg_acc_80','pred_reg_acc_90')
write.table(all_pred_detail, file = paste0("predictions_detail_",feature_scenario,'.txt'),sep = ',',quote = FALSE)

all_pred_broad = data.frame(
                            predictions_broad_class$names,
                            predictions_broad_class$predictions,
                            predictions_broad_class_70_acc$predictions,
                            predictions_broad_class_80_acc$predictions,
                            predictions_broad_class_90_acc$predictions,
                            predictions_broad_reg$predictions,
                            predictions_broad_reg_70_acc$predictions,
                            predictions_broad_reg_80_acc$predictions,
                            predictions_broad_reg_90_acc$predictions
                            )
names(all_pred_broad) = c('species','pred_class','pred_class_acc_70','pred_class_acc_80','pred_class_acc_90','pred_reg','pred_reg_acc_70','pred_reg_acc_80','pred_reg_acc_90')
write.table(all_pred_broad, file = paste0("predictions_broad_",feature_scenario,'.txt'),sep = ',',quote = FALSE)


#
# res_broad = train_iucnn(features,
#                         labels_train_broad,
#                         cv_fold = 5,
#                         n_layers = '9_9_9',
#                         validation_fraction=0,
#                         save_model=TRUE
#                         )
#
# res_broad$validation_accuracy
#
