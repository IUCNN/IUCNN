#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Octuber 2025

@author: Torsten hauffe (torsten.hauffe@gmail.com)
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

# declare location of the python files make functions of other python files importable
sys.path.append(os.path.dirname(__file__))
from IUCNN_predict import rescale_labels, turn_reg_output_into_softmax


def iucnn_pdp(input_features,
              focal_features,
              model_dir,
              iucnn_mode,
              cv_fold,
              dropout,
              dropout_reps,
              rescale_factor,
              min_max_label,
              stretch_factor_rescaled_labels
             ):

    uncertainty = dropout
    if iucnn_mode == 'bnn-class':
        import np_bnn as bn
        weight_pickle = model_dir
        bnn_obj, mcmc_obj, logger_obj = bn.load_obj(weight_pickle)
        posterior_weight_samples = logger_obj._post_weight_samples
        actFun = bnn_obj._act_fun
        output_act_fun = bnn_obj._output_act_fun
        uncertainty = True
    else:
        if cv_fold == 1:
            model = [tf.keras.models.load_model(model_dir)]
        else:
            model = [tf.keras.models.load_model(model_dir[i]) for i in range(cv_fold)]

    if not isinstance(focal_features, list):
        focal_features = [focal_features]

    num_taxa = input_features.shape[0]
    pdp_features = make_pdp_features(input_features, focal_features)
    num_pdp_steps = pdp_features.shape[0]

    if iucnn_mode == 'nn-reg':
        num_iucn_cat = int(rescale_factor + 1)
        num_model_output = 1
        label_cats = np.arange(num_iucn_cat)
        depth_idx, row_idx = np.indices((num_taxa, cv_fold * dropout_reps))
    else:
        num_iucn_cat = min_max_label[1] + 1
        num_model_output = num_iucn_cat

    if uncertainty:
        pdp_lwr = np.zeros((num_pdp_steps, num_iucn_cat))
        pdp_upr = np.zeros((num_pdp_steps, num_iucn_cat))
    else:
        dropout_reps = 1

    if iucnn_mode == 'nn-reg':
        idx_axis0, idx_axis1 = np.indices((num_taxa, cv_fold * dropout_reps))

    pdp = np.zeros((num_pdp_steps, num_iucn_cat))

    if iucnn_mode == 'cnn':
        sys.exit('No partial dependence probabilities possible for CNN')

    else:
        for i in range(num_pdp_steps):
            tmp_features = np.copy(input_features)
            tmp_features[:, focal_features] = pdp_features[i, :]

            predictions_raw = np.zeros((num_taxa, cv_fold * dropout_reps, num_model_output))
            counter = 0
            for j in range(cv_fold):
                if iucnn_mode == 'bnn-class':
                    predictions_raw, _ = bn.get_posterior_cat_prob(tmp_features,
                                                                   posterior_weight_samples,
                                                                   actFun=actFun,
                                                                   output_act_fun=output_act_fun)
                    predictions_raw = np.swapaxes(predictions_raw, 0, 1)
                else:
                    for k in range(dropout_reps):
                        predictions_raw[:, counter, :] = model[j].predict(tmp_features, verbose=0)
                        counter += 1

            if iucnn_mode == 'nn-reg':
                predictions_rescaled = rescale_labels(predictions_raw, rescale_factor, min_max_label,
                                                      stretch_factor_rescaled_labels, reverse=True)
                predictions_raw = np.zeros((num_taxa, cv_fold * dropout_reps, num_iucn_cat))
                predictions_rescaled = np.round(predictions_rescaled, 0).astype(int)
                predictions_rescaled = np.squeeze(predictions_rescaled, axis=2)
                predictions_raw[idx_axis0, idx_axis1, predictions_rescaled] = 1.0

            pred_reps = np.cumsum(predictions_raw, axis=2)

            if uncertainty:
                pred_quantiles = np.quantile(np.mean(pred_reps, axis=0), q=(0.025, 0.975), axis=0)
                pdp_lwr[i, :] = pred_quantiles[0, :]
                pdp_upr[i, :] = pred_quantiles[1, :]

            pdp[i, :] = np.mean(pred_reps, axis=(0, 1))

    out_dict = {
        'feature': pdp_features,
        'pdp': pdp
    }
    if uncertainty:
        out_dict.update({'lwr': pdp_lwr, 'upr': pdp_upr})

    return  out_dict


def get_focal_summary(input_features, focal_features):
    """Get whether features are ohe/binary/ordinal/continuous and their min and max values"""
    num_features = len(focal_features)
    focal_summary = np.zeros((3, num_features))

    for i in range(num_features):
        ff = input_features[:, focal_features[i]]
        values, counts = np.unique(ff, return_counts=True)
        focal_summary[1, i] = np.nanmin(values)
        focal_summary[2, i] = np.nanmax(values)
        values_range = np.arange(focal_summary[1, i], focal_summary[2, i] + 1)
        focal_summary[0, i] = np.all(np.isin(values, values_range))

    return focal_summary


def make_pdp_features(input_features, focal_features):
    """Get the features for which we calculate the PDP"""
    focal_summary = get_focal_summary(input_features, focal_features)

    if np.sum(focal_summary[0, :] == 0) and len(focal_features) == 1:
        # Single continuous feature
        pdp_feat = np.linspace(focal_summary[1, 0], focal_summary[2, 0], num=100).reshape(100, 1)
    elif focal_summary[0, 0] == 1 and len(focal_features) == 1:
        # ordinal or binary
        M = focal_summary[2, 0]
        pdp_feat = np.linspace(focal_summary[1, 0], M, num=M + 1).reshape((M + 1, 1))
    else:
        # One-hot-encoded
        pdp_feat = np.eye(focal_summary.shape[1])

    return pdp_feat
