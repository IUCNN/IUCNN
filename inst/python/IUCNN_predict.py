import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
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


def iucnn_predict(feature_set,
                  model_dir,
                  verbose,
                  return_raw,
                  iucnn_mode,
                  rescale_labels_boolean,
                  rescale_factor,
                  min_max_label,
                  stretch_factor_rescaled_labels
                  ):
    #print("Loading model...")
    if iucnn_mode == 'nn-class':
        model = tf.keras.models.load_model(model_dir)
        prm_est = model.predict(feature_set, verbose=verbose)
        predictions = np.argmax(prm_est, axis=1)

    elif iucnn_mode == 'nn-reg':
        model = tf.keras.models.load_model(model_dir)
        prm_est_tmp = model.predict(feature_set, verbose=verbose).flatten()
        if rescale_labels:
            prm_est = rescale_labels(prm_est_tmp,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True)
        else:
            prm_est = prm_est_tmp
        predictions = np.round(prm_est, 0).astype(int).flatten()        
        

    if return_raw:
        return [predictions, prm_est]
    else:
        return [predictions]



















