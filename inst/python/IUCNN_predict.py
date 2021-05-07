"""
Created on Fri May  6 17:51:02 2021

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import tensorflow as tf
import numpy as np
import os, sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
    

def iucnn_predict(feature_set,
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
    if iucnn_mode == 'nn-class':
        model = tf.keras.models.load_model(model_dir)
        if dropout:
            predictions_raw = np.array([model.predict(feature_set) for i in np.arange(dropout_reps)])
            mc_dropout_probs = np.mean(predictions_raw,axis=0)
            predictions = np.argmax(mc_dropout_probs, axis=1)
        else:
            predictions_raw = model.predict(feature_set, verbose=0)
            mc_dropout_probs = np.nan
            predictions = np.argmax(predictions_raw, axis=1)

        

    elif iucnn_mode == 'nn-reg':
        model = tf.keras.models.load_model(model_dir)
        if dropout:
            label_cats = np.arange(rescale_factor+1)
            predictions_raw_unscaled = np.array([model.predict(feature_set).flatten() for i in np.arange(dropout_reps)])
            predictions_raw = np.array([rescale_labels(i,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) for i in predictions_raw_unscaled]).T
            mc_dropout_probs = turn_reg_output_into_softmax(predictions_raw.T,label_cats)
            predictions = np.argmax(mc_dropout_probs, axis=1)
        else:
            predictions_raw_unscaled = model.predict(feature_set).flatten()
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



















