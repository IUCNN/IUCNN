#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:25:09 2021

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import os, sys
# use only one thread
try:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
except:
    pass

import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf compilation warning
except:
    pass

# declare location of the python files make functions of other python files importable
sys.path.append(os.path.dirname(__file__))
from IUCNN_predict import rescale_labels


def get_regression_accuracy(model,features,labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels):
    prm_est = model.predict(features).flatten()
    prm_est_rescaled = rescale_labels(prm_est,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True)
    real_labels = rescale_labels(labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True).astype(int).flatten()
    label_predictions = np.round(prm_est_rescaled, 0).astype(int).flatten()
    cat_acc = np.sum(label_predictions==real_labels)/len(label_predictions)
    return cat_acc, label_predictions, prm_est_rescaled


def feature_importance_nn( input_features,
                           true_labels,
                           model_dir,
                           iucnn_mode,
                           feature_names,
                           rescale_factor,
                           min_max_label,
                           stretch_factor_rescaled_labels,
                           verbose,
                           n_permutations,
                           feature_blocks,
                           unlink_features_within_block):


    feature_indices = np.arange(input_features.shape[1])
    # if no names are provided, name them by index
    if len(feature_names) == 0:
        feature_names = feature_indices.astype(str)
    if len(feature_blocks.keys()) > 0:
        selected_features = []
        feature_block_names = []
        for block_name, block_indices in feature_blocks.items():
            selected_features.append(block_indices)
            feature_block_names.append(block_name)
    else:
        selected_features = [[i] for i in feature_indices]
        feature_block_names = [[i] for i in feature_names]

    # if there is no block-specific information if permutation of the features within the block should be independent
    if isinstance(unlink_features_within_block, bool):
        unlink_features_within_block = [unlink_features_within_block] * len(selected_features)

    model = tf.keras.models.load_model(model_dir)
    if iucnn_mode == 'nn-class':
        true_labels= tf.keras.utils.to_categorical(true_labels)        
        __, ref_accuracy = model.evaluate(input_features,true_labels,verbose=0)

    elif iucnn_mode == 'nn-reg':
        true_labels = rescale_labels(true_labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
        ref_accuracy,__,__ = get_regression_accuracy(model,input_features,true_labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
   
    if verbose:
        print("Reference accuracy:", ref_accuracy,flush=True)
    # go through features and shuffle one at a time
    accuracies_wo_feature = []
    for block_id,feature_block in enumerate(selected_features):
        try: # see if we have an actual list or single element
            feature_block = list(feature_block)
        except TypeError:
            feature_block = [feature_block]
        if verbose:
            print('Processing feature block %i'%(int(block_id)+1),flush=True)
        n_accuracies = []
        for i in np.arange(n_permutations):
            features = input_features.copy()
            if unlink_features_within_block[block_id] and len(feature_block)>1:
                for feature_index in feature_block: # shuffle each column by it's own random indices
                    features[:,feature_index] = np.random.permutation(features[:,feature_index])
            else:
                features[:,feature_block] = np.random.permutation(features[:,feature_block])
            if iucnn_mode == 'nn-class':
                __, accuracy = model.evaluate(features,true_labels,verbose=0)
        
            elif iucnn_mode == 'nn-reg':
                accuracy,__,__ = get_regression_accuracy(model,features,true_labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
            
            n_accuracies.append(accuracy)     
        accuracies_wo_feature.append(n_accuracies)
    accuracies_wo_feature = np.array(accuracies_wo_feature)

    delta_accs = ref_accuracy-np.array(accuracies_wo_feature)    
    delta_accs_means = np.mean(delta_accs,axis=1)
    delta_accs_stds = np.std(delta_accs,axis=1)
    accuracies_wo_feature_means = np.mean(accuracies_wo_feature,axis=1)
    accuracies_wo_feature_stds = np.std(accuracies_wo_feature,axis=1)
    d = dict(zip(feature_block_names,zip(delta_accs_means,delta_accs_stds,accuracies_wo_feature_means,accuracies_wo_feature_stds)))
    feature_importance_sorted = dict(sorted(d.items(), key=lambda item: item[1],reverse=True))
    
    
    # feature_importance_df = pd.DataFrame(np.array([np.arange(0,len(selected_features)),feature_block_names,
    #                                                delta_accs_means,delta_accs_stds,
    #                                                accuracies_wo_feature_means,accuracies_wo_feature_stds]).T,
    #                                      columns=['feature_block_index','feature_name','delta_acc_mean','delta_acc_std',
    #                                               'acc_with_feature_randomized_mean','acc_with_feature_randomized_std'])
    # feature_importance_df.iloc[:,2:] = feature_importance_df.iloc[:,2:].astype(float)
    # feature_importance_df_sorted = feature_importance_df.sort_values('delta_acc_mean',ascending=False)
    # feature_importance_df_sorted['delta_acc_mean'] = pd.to_numeric(feature_importance_df_sorted['delta_acc_mean'])
    # feature_importance_df_sorted['acc_with_feature_randomized_mean'] = pd.to_numeric(feature_importance_df_sorted['acc_with_feature_randomized_mean'])    
    # # define outfile name
    # if predictions_outdir == "":
    #     predictions_outdir = os.path.dirname(model_dir)
    # if not os.path.exists(predictions_outdir) and predictions_outdir != "":
    #     os.makedirs(predictions_outdir)
    # fname_stem = os.path.basename(model_dir)
    # feature_importance_df_filename = os.path.join(predictions_outdir, fname_stem + '_feature_importance.txt')
    # format the last two columns as numeric for applyign float printing formatting options
    # feature_importance_df_sorted.to_csv(feature_importance_df_filename,sep='\t',index=False,header=True,float_format='%.6f')
    # print("Output saved in: %s" % feature_importance_df_filename)

    return feature_importance_sorted
