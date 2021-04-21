library(tidyverse)
library(IUCNN)
library(devtools)
document()
setwd('~/GitHub/IUCNN/orchid_data')

# load occurrence data
load('orchid_data_iucnn/orchid_original_training_occurrences.rda')
# convert into table for input into IUCNN
species = paste(training_occs$genus,training_occs$epitheton,sep = ' ')
decimallongitude = training_occs$decimallongitude
decimallatitude = training_occs$decimallatitude
occurrence_df = tibble(species,decimallongitude,decimallatitude)

# calculate feature
#features = prep_features(occurrence_df)
features_file = "orchid_data_iucnn/features.RData"
#save(features, file = features_file)
load(features_file)

# load labels
status_info = read_csv('orchid_data_iucnn/original_orchid_training_labels.csv')
names(status_info) = c("species", "labels")


# 1. TRAINING____________________________________________________

# 1.1 DETAILED LABELS_____________________________________________
labels_train_detail <- prep_labels(status_info,level = 'detail')


# 1.1.1 CLASS_______________________________________________________
# define logfile to store info of different runs
logfile_detail_class = "model_testing_detailed_nn_class.txt"
model_testing_results_detail_class = modeltest_iucnn(features,
                                                   labels_train_detail,
                                                   logfile_detail_class,
                                                   seed = 1234,
                                                   dropout_rate = c(0.,0.1,0.3),
                                                   n_layers = c('30','20_40','10_30_50'),
                                                   cv_fold = 5,
                                                   validation_fraction = 0.,
                                                   mode = 'nn-class',
                                                   init_logfile = TRUE,
                                                   recycle_settings = FALSE)

#model_testing_results_detail_class = read.csv('model_testing_detailed_nn_class.txt',sep='\t')
#best_model_detail = model_testing_results_detail[4,]
best_model_detail_class = get_best_model(model_testing_results_detail_class,rank_mode = 1)
res_eval_detail_class = evaluate_model(features,labels_train_detail,best_model_detail_class)
# train final model
iucnn_results_detail_class = train_iucnn(features,
                                       labels_train_detail,
                                       path_to_output = 'iucnn_orchid_nn_class_detail',
                                       best_model = best_model_detail_class
)


# 1.1.2 REG_________________________________________________________
# define logfile to store info of different runs
logfile_detail_reg = "model_testing_detailed_nn_reg.txt"
model_testing_results_detail_reg = modeltest_iucnn(features,
                                        labels_train_detail,
                                        logfile_detail_reg,
                                        seed = 1234,
                                        dropout_rate = c(0.,0.1,0.3),
                                        n_layers = c('30','20_40','10_30_50'),
                                        cv_fold = 5,
                                        validation_fraction = 0.,
                                        mode = 'nn-reg',
                                        act_f_out = c('tanh','sigmoid'),
                                        label_stretch_factor = c(0.8,1.0,1.2),
                                        label_noise_factor = 0.0,
                                        init_logfile = TRUE,
                                        recycle_settings = FALSE)

#model_testing_results_detail = read.csv('model_testing_detailed_nn_reg.txt',sep='\t')
#best_model_detail = model_testing_results_detail[4,]
best_model_detail_reg = get_best_model(model_testing_results_detail_reg,rank_mode = 1)
res_eval_detail_reg = evaluate_model(features,labels_train_detail,best_model_detail_reg)
# train final model
iucnn_results_detail_reg = train_iucnn(features,
                             labels_train_detail,
                             path_to_output = 'iucnn_orchid_nn_reg_detail',
                             best_model = best_model_detail_reg
                            )



# 1.2 BROAD LABELS_____________________________________________
labels_train_broad <- prep_labels(status_info,level = 'broad')


# 1.2.1 CLASS_______________________________________________________
# define logfile to store info of different runs
logfile_broad_class = "model_testing_broad_nn_class.txt"
model_testing_results_broad_class = modeltest_iucnn(features,
                                                     labels_train_broad,
                                                     logfile_broad_class,
                                                     seed = 1234,
                                                     dropout_rate = c(0.,0.1,0.3),
                                                     n_layers = c('30','20_40','10_30_50'),
                                                     cv_fold = 5,
                                                     validation_fraction = 0.,
                                                     mode = 'nn-class',
                                                     init_logfile = TRUE,
                                                     recycle_settings = FALSE)

#model_testing_results_detail = read.csv('model_testing_detailed_nn_reg.txt',sep='\t')
#best_model_detail = model_testing_results_detail[4,]
best_model_broad_class = get_best_model(model_testing_results_broad_class,rank_mode = 1)
res_eval_broad_class = evaluate_model(features,labels_train_broad,best_model_broad_class)
# train final model
iucnn_results_broad_class = train_iucnn(features,
                                         labels_train_broad,
                                         path_to_output = 'iucnn_orchid_nn_class_broad',
                                         best_model = best_model_broad_class
)


# 1.2.2 REG_________________________________________________________
# define logfile to store info of different runs
logfile_broad_reg = "model_testing_broad_nn_reg.txt"
model_testing_results_broad_reg = modeltest_iucnn(features,
                                                   labels_train_broad,
                                                   logfile_broad_reg,
                                                   seed = 1234,
                                                   dropout_rate = c(0.,0.1,0.3),
                                                   n_layers = c('30','20_40','10_30_50'),
                                                   cv_fold = 5,
                                                   validation_fraction = 0.,
                                                   mode = 'nn-reg',
                                                   act_f_out = c('tanh','sigmoid'),
                                                   label_stretch_factor = c(0.8,1.0,1.2),
                                                   label_noise_factor = 0.0,
                                                   init_logfile = TRUE,
                                                   recycle_settings = FALSE)

#model_testing_results_detail = read.csv('model_testing_detailed_nn_reg.txt',sep='\t')
#best_model_detail = model_testing_results_detail[4,]
best_model_broad_reg = get_best_model(model_testing_results_broad_reg,rank_mode = 1)
res_eval_broad_reg = evaluate_model(features,labels_train_broad,best_model_broad_reg)
# train final model
iucnn_results_broad_reg = train_iucnn(features,
                                       labels_train_broad,
                                       path_to_output = 'iucnn_orchid_nn_reg_broad',
                                       best_model = best_model_broad_reg
)





# 2. PREDICTIONS_______________________________________________
# load occurrence data
#load('orchid_data_iucnn/orchid_original_prediction_occurrence.rda')
# convert into table for input into IUCNN
#species = paste(prediction_occs$genus,prediction_occs$epitheton,sep = ' ')
#decimallongitude = prediction_occs$decimallongitude
#decimallatitude = prediction_occs$decimallatitude
#occurrence_df = tibble(species,decimallongitude,decimallatitude)
#features_pred = prep_features(occurrence_df)
features_file_pred = "orchid_data_iucnn/features_pred.RData"
#save(features_pred, file = features_file_pred)
load(features_file_pred)

# 2.1.1 DETAIL CLASS_________________________________________________
predictions_detail_class = predict_iucnn(features_pred,iucnn_results_detail_class)
predictions_detail_class_60_acc = predict_iucnn(features_pred,iucnn_results_detail_class,res_eval_detail_class,target_acc = 0.60)
predictions_detail_class_70_acc = predict_iucnn(features_pred,iucnn_results_detail_class,res_eval_detail_class,target_acc = 0.70)
predictions_detail_class_75_acc = predict_iucnn(features_pred,iucnn_results_detail_class,res_eval_detail_class,target_acc = 0.75)
pdf('plots/predictions_detail_class.pdf')
plot_predictions(predictions_detail_class,title="Detailed, iucnn-class")
plot_predictions(predictions_detail_class_60_acc,title="Detailed, iucnn-class, 60% acc")
plot_predictions(predictions_detail_class_70_acc,title="Detailed, iucnn-class, 70% acc")
plot_predictions(predictions_detail_class_75_acc,title="Detailed, iucnn-class, 75% acc")
dev.off()

# 2.1.2 DETAIL REG_________________________________________________
predictions_detail_reg = predict_iucnn(features_pred,iucnn_results_detail_reg)
predictions_detail_reg_60_acc = predict_iucnn(features_pred,iucnn_results_detail_reg,res_eval_detail_reg,target_acc = 0.60)
predictions_detail_reg_70_acc = predict_iucnn(features_pred,iucnn_results_detail_reg,res_eval_detail_reg,target_acc = 0.70)
predictions_detail_reg_75_acc = predict_iucnn(features_pred,iucnn_results_detail_reg,res_eval_detail_reg,target_acc = 0.75)
pdf('plots/predictions_detail_reg.pdf')
plot_predictions(predictions_detail_reg,title="Detailed, iucnn-reg")
plot_predictions(predictions_detail_reg_60_acc,title="Detailed, iucnn-reg, 60% acc")
plot_predictions(predictions_detail_reg_70_acc,title="Detailed, iucnn-reg, 70% acc")
plot_predictions(predictions_detail_reg_75_acc,title="Detailed, iucnn-reg, 75% acc")
dev.off()

# 2.2.1 BROAD CLASS_________________________________________________
predictions_broad_class = predict_iucnn(features_pred,iucnn_results_broad_class)
predictions_broad_class_70_acc = predict_iucnn(features_pred,iucnn_results_broad_class,res_eval_broad_class,target_acc = 0.70)
predictions_broad_class_80_acc = predict_iucnn(features_pred,iucnn_results_broad_class,res_eval_broad_class,target_acc = 0.80)
predictions_broad_class_90_acc = predict_iucnn(features_pred,iucnn_results_broad_class,res_eval_broad_class,target_acc = 0.90)
pdf('plots/predictions_broad_class.pdf')
plot_predictions(predictions_broad_class,title="Broad, iucnn-class")
plot_predictions(predictions_broad_class_70_acc,title="Broad, iucnn-class, 70% acc")
plot_predictions(predictions_broad_class_80_acc,title="Broad, iucnn-class, 80% acc")
plot_predictions(predictions_broad_class_90_acc,title="Broad, iucnn-class, 90% acc")
dev.off()

# 2.2.2 BROAD REG_________________________________________________
predictions_broad_reg = predict_iucnn(features_pred,iucnn_results_broad_reg)
predictions_broad_reg_70_acc = predict_iucnn(features_pred,iucnn_results_broad_reg,res_eval_broad_reg,target_acc = 0.70)
predictions_broad_reg_80_acc = predict_iucnn(features_pred,iucnn_results_broad_reg,res_eval_broad_reg,target_acc = 0.80)
predictions_broad_reg_90_acc = predict_iucnn(features_pred,iucnn_results_broad_reg,res_eval_broad_reg,target_acc = 0.90)
pdf('plots/predictions_broad_reg.pdf')
plot_predictions(predictions_broad_reg,title="Broad, iucnn-reg")
plot_predictions(predictions_broad_reg_70_acc,title="Broad, iucnn-reg, 70% acc")
plot_predictions(predictions_broad_reg_80_acc,title="Broad, iucnn-reg, 80% acc")
plot_predictions(predictions_broad_reg_90_acc,title="Broad, iucnn-reg, 90% acc")
dev.off()
all_pred_detail = data.frame(
                            predictions_detail_class$names,
                            predictions_detail_class$predictions,
                            predictions_detail_class_60_acc$predictions,
                            predictions_detail_class_70_acc$predictions,
                            predictions_detail_class_75_acc$predictions,
                            predictions_detail_reg,
                            predictions_detail_reg_60_acc$predictions,
                            predictions_detail_reg_70_acc$predictions,
                            predictions_detail_reg_75_acc$predictions
                            )
names(all_pred_detail) = c('species','pred_class','pred_class_acc_70','pred_class_acc_80','pred_class_acc_90','pred_reg','pred_reg_acc_70','pred_reg_acc_80','pred_reg_acc_90')
write.table(all_pred_detail, file = "predictions_detail.txt",sep = ',',quote = FALSE)

all_pred_broad = data.frame(
                            predictions_broad_class$names,
                            predictions_broad_class$predictions,
                            predictions_broad_class_70_acc$predictions,
                            predictions_broad_class_80_acc$predictions,
                            predictions_broad_class_90_acc$predictions,
                            predictions_broad_reg,
                            predictions_broad_reg_70_acc$predictions,
                            predictions_broad_reg_80_acc$predictions,
                            predictions_broad_reg_90_acc$predictions
                            )
names(all_pred_broad) = c('species','pred_class','pred_class_acc_70','pred_class_acc_80','pred_class_acc_90','pred_reg','pred_reg_acc_70','pred_reg_acc_80','pred_reg_acc_90')
write.table(all_pred_broad, file = "predictions_broad.txt",sep = ',',quote = FALSE)



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
# a = feature_importance(x = iucnn_results_broad_class)
# a
