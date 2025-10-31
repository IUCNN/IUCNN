#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:51:02 2021

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

import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# disable progress bars globally (instead of model.predict(..., verbose=0), which does not supress progress output in R)
tf.keras.utils.disable_interactive_logging()

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf compilation warning
except:
    pass


def rescale_labels(labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=False):
    label_range = max(min_max_label)-min(min_max_label)
    modified_range = stretch_factor_rescaled_labels*label_range
    midpoint_range = np.mean(min_max_label)
    if reverse:
        rescaled_labels_tmp = (labels-midpoint_range)/modified_range
        rescaled_labels = (rescaled_labels_tmp+0.5)*rescale_factor
    else:
        rescaled_labels_tmp = (labels/rescale_factor)-0.5
        rescaled_labels = rescaled_labels_tmp*modified_range+midpoint_range
    return(rescaled_labels)
    
def turn_low_confidence_instances_to_nan(pred,high_pp_indices):
    pred_temp = np.zeros(pred.shape)
    pred_temp[:] = np.nan
    pred_temp[high_pp_indices] = pred[high_pp_indices]
    pred = pred_temp
    return pred

def turn_reg_output_into_softmax(reg_out_rescaled,label_cats):
    predictions = np.round(reg_out_rescaled, 0).astype(int)
    softmax_probs_mean = np.array([[len(np.where(predictions[:,i]==j)[0])/len(predictions[:,i]) for j in label_cats] for  i in np.arange(predictions.shape[1])])
    return softmax_probs_mean


def store_pred_features_as_pkl(input_raw):
    import pickle as pkl
    filehandler = open('orchid_data/cnn_test/data/pred_features_raw.pkl', "wb")
    pkl.dump(input_raw, filehandler)
    filehandler.close()


def load_input_data_manually():
    import pickle as pkl
    file = open("orchid_data/cnn_test/data/pred_features_raw.pkl",'rb')
    input_raw = pkl.load(file)
    file.close()
    model_dir = "iuc_nn_model/production_cnn/cnn_model_0"
    iucnn_mode = 'cnn'
    dropout = True
    dropout_reps = 10
    confidence_threshold = 0.67
    rescale_factor = None
    min_max_label = [0,4]
    stretch_factor_rescaled_labels = 1

def iucnn_predict(input_raw,
                  model_dir,
                  iucnn_mode,
                  dropout,
                  dropout_reps,
                  confidence_threshold,
                  rescale_factor,
                  min_max_label,
                  stretch_factor_rescaled_labels
                  ):
    #print("Loading model...")
    if iucnn_mode == 'nn-class' or iucnn_mode == 'cnn':
        if iucnn_mode == 'cnn':
            instance_names = np.array(list(input_raw.keys()))
            data_matrix = np.array([input_raw[i] for i in instance_names]).astype(int)
            del input_raw
            feature_set = np.array(data_matrix).reshape(list(data_matrix.shape) + [1])
            del data_matrix

        else:
            feature_set = input_raw

        model = tf.keras.models.load_model(model_dir)
        if dropout:
            if iucnn_mode == 'cnn':
                for i in np.arange(dropout_reps):
                    print('Predicting for dropout rep: %i'%i, flush=True)
                    pred = model.predict(feature_set, verbose=0)
                    # this summing below is done to save memory
                    # instead of appending all pred arrays into one giant master array
                    if i == 0:
                        pred_sum = pred
                    else:
                        pred_sum = np.sum([pred_sum,pred],axis=0)
                predictions_raw = pred_sum
                mc_dropout_probs = pred_sum/dropout_reps
            else:
                predictions_raw = np.array([model.predict(feature_set) for i in np.arange(dropout_reps)])
                mc_dropout_probs = np.mean(predictions_raw, axis=0)
            predictions = np.argmax(mc_dropout_probs, axis=1)
        else:
            predictions_raw = model.predict(feature_set, verbose=0)
            mc_dropout_probs = np.nan
            predictions = np.argmax(predictions_raw, axis=1)

    elif iucnn_mode == 'nn-reg':
        feature_set = input_raw
        model = tf.keras.models.load_model(model_dir)
#        model = tf.keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')
        if dropout:
            label_cats = np.arange(rescale_factor+1)
            predictions_raw_unscaled = np.array([model.predict(feature_set, verbose=0).flatten() for i in np.arange(dropout_reps)])
            predictions_raw = np.array([rescale_labels(i,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) for i in predictions_raw_unscaled]).T
            mc_dropout_probs = turn_reg_output_into_softmax(predictions_raw.T,label_cats)
            predictions = np.argmax(mc_dropout_probs, axis=1)
        else:
            predictions_raw_unscaled = model.predict(feature_set, verbose=0).flatten()
            predictions_raw = rescale_labels(predictions_raw_unscaled,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) 
            mc_dropout_probs = np.nan
            predictions = np.round(predictions_raw, 0).astype(int).flatten()


    if confidence_threshold:
        if not dropout:
            sys.exit('target_acc can only be used for models trained with dropout. Retrain your model and specify a dropout rate > 0 to use this option.')
        else:
            #confidence_threshold = get_confidence_threshold(model,iucnn_mode,test_data,test_labels,dropout_reps,rescale_factor,min_max_label,stretch_factor_rescaled_labels,target_acc)
            high_pp_indices = np.where(np.max(mc_dropout_probs, axis=1) > confidence_threshold)[0]
            #mc_dropout_probs = turn_low_confidence_instances_to_nan(mc_dropout_probs,high_pp_indices)
            predictions = turn_low_confidence_instances_to_nan(predictions,high_pp_indices)
            #post_softmax_probs = np.array([turn_low_pp_instances_to_nan(i,high_pp_indices) for i in post_softmax_probs])

    nreps = 1000
    if dropout:
        label_dict = np.arange(mc_dropout_probs.shape[1])
        samples = np.array([np.random.choice(label_dict, nreps, p=i) for i in mc_dropout_probs])
        predicted_class_count = np.array([[list(col).count(i) for i in label_dict] for col in samples.T])
    else:
        predicted_class_count = np.nan


    out_dict = {
        'raw_predictions':predictions_raw,
        'sampled_cat_freqs':predicted_class_count,
        'mc_dropout_probs':mc_dropout_probs,
        'class_predictions':predictions
    }

    return  out_dict



















