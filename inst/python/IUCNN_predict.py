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

def get_confidence_threshold(model,iucnn_mode,test_data,test_labels,dropout_reps,rescale_factor,min_max_label,stretch_factor_rescaled_labels,target_acc=0.9):
    if iucnn_mode == 'nn-class':
        prm_est_reps = np.array([model.predict(test_data) for i in np.arange(dropout_reps)])
        prm_est_mean = np.mean(prm_est_reps,axis=0)
    elif iucnn_mode == 'nn-reg':
        label_cats = np.arange(max(min_max_label)+1)
        prm_est_reps_unscaled = np.array([model.predict(test_data).flatten() for i in np.arange(dropout_reps)])
        predictions_raw = np.array([rescale_labels(i,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) for i in prm_est_reps_unscaled])
        prm_est_mean = turn_reg_output_into_softmax(predictions_raw,label_cats)
    # CALC TRADEOFFS
    tbl_results = []
    for i in np.linspace(0.01, 0.99, 99):
        try:
            scores = get_accuracy_threshold(prm_est_mean, test_labels, threshold=i)
            tbl_results.append([i, scores['accuracy'], scores['retained_samples']])
        except:
            pass
    tbl_results = np.array(tbl_results)
    try:
        indx = np.min(np.where(np.round(tbl_results[:,1],2) >= target_acc))
    except ValueError:
        sys.exit('Target accuracy can not be reached. Set a lower threshold and try again.')
    selected_row = tbl_results[indx,:]
    return selected_row[0]
    
def get_accuracy_threshold(probs, labels, threshold=0.75):
    indx = np.where(np.max(probs, axis=1)>threshold)[0]
    res_supported = probs[indx,:]
    labels_supported = labels[indx]
    pred = np.argmax(res_supported, axis=1)
    accuracy = len(pred[pred == labels_supported])/len(pred)
    dropped_frequency = len(pred)/len(labels)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency}

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
                  verbose,
                  iucnn_mode,
                  dropout,
                  dropout_reps,
                  target_acc,
                  test_data,
                  test_labels,
                  rescale_labels_boolean,
                  rescale_factor,
                  min_max_label,
                  stretch_factor_rescaled_labels
                  ):
    #print("Loading model...")
    if iucnn_mode == 'nn-class':
        model = tf.keras.models.load_model(model_dir)
        if dropout:
            predictions_raw = np.array([model.predict(feature_set) for i in np.arange(dropout_reps)])
            predictions_raw_mean = np.mean(predictions_raw,axis=0)
        else:
            predictions_raw_mean = model.predict(feature_set, verbose=verbose)
        softmax_probs = predictions_raw_mean
        #predictions = np.argmax(predictions_raw_mean, axis=1)

    elif iucnn_mode == 'nn-reg':
        model = tf.keras.models.load_model(model_dir)
        if dropout:
            prm_est_reps = np.array([model.predict(feature_set).flatten() for i in np.arange(dropout_reps)])
            prm_est_mean = np.mean(prm_est_reps,axis=0)
            #prm_est_reps_rescaled = np.array([rescale_labels(i,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) for i in prm_est_reps])
        else:
            prm_est_mean = model.predict(feature_set).flatten()

        if rescale_labels_boolean:
            predictions_raw_mean = rescale_labels(prm_est_mean,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True)
        else:
            predictions_raw_mean = prm_est_mean

        if dropout:
            predictions_raw = np.array([rescale_labels(i,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) for i in prm_est_reps])
            label_cats = np.arange(max(min_max_label)+1)
            softmax_probs = turn_reg_output_into_softmax(predictions_raw,label_cats)
        #predictions = np.round(predictions_raw_mean, 0).astype(int).flatten()        
        
    if target_acc > 0 and len(test_labels) > 0:
        if not dropout:
            sys.exit('target_acc can only be used for models trained with dropout. Retrain your model and specify a dropout rate > 0 to use this option.')
        else:
            confidence_threshold = get_confidence_threshold(model,iucnn_mode,test_data,test_labels,dropout_reps,rescale_factor,min_max_label,stretch_factor_rescaled_labels,target_acc)
            high_pp_indices = np.where(np.max(softmax_probs, axis=1) > confidence_threshold)[0]
            predictions_raw_mean = turn_low_confidence_instances_to_nan(predictions_raw_mean,high_pp_indices)
            #post_softmax_probs = np.array([turn_low_pp_instances_to_nan(i,high_pp_indices) for i in post_softmax_probs])

    return predictions_raw_mean



















