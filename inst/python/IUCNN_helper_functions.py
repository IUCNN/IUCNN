#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:51:02 2021

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import sys,glob,os
import numpy as np
import np_bnn as bn


def get_confidence_threshold(predicted_labels, true_labels, target_acc=0.9):
    # CALC TRADEOFFS
    tbl_results = []
    for i in np.linspace(0.01, 0.99, 99):
        try:
            scores = get_accuracy_threshold(predicted_labels, true_labels, threshold=i)
            tbl_results.append([i, scores['accuracy'], scores['retained_samples']])
        except:
            pass
    tbl_results = np.array(tbl_results)
    if target_acc is None:
        return tbl_results
    else:
        try:
            indx = np.min(np.where(np.round(tbl_results[:, 1], 2) >= target_acc))
        except ValueError:
            sys.exit('Target accuracy can not be reached. Set a lower threshold and try again.')
        selected_row = tbl_results[indx, :]
        return selected_row[0]


def get_accuracy_threshold(probs, labels, threshold=0.75):
    indx = np.where(np.max(probs, axis=1) > threshold)[0]
    res_supported = probs[indx, :]
    labels_supported = labels[indx]
    pred = np.argmax(res_supported, axis=1)
    accuracy = len(pred[pred == labels_supported]) / len(pred)
    dropped_frequency = len(pred) / len(labels)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency}



def get_acctbl_and_catsample_bnn(pkl_path,cv_mode = False):
    if cv_mode:
        cv_folder = pkl_path
        pkl_files = glob.glob(os.path.join(cv_folder,'*.pkl'))
        post_softmax_out = []
        post_predictions = []
        true_labels = []
        pred_labels = []
        for weight_pickle in pkl_files:
            bnn_obj, mcmc_obj, logger_obj = bn.load_obj(weight_pickle)
            test_features = bnn_obj._test_data
            test_labels = bnn_obj._test_labels
            #test_features = test_features[test_labels>0,:]
            #test_labels = test_labels[test_labels>0]
            posterior_weight_samples = logger_obj._post_weight_samples
            actFun = bnn_obj._act_fun
            output_act_fun = bnn_obj._output_act_fun
            post_softmax_probs,post_prob_predictions = bn.get_posterior_cat_prob(test_features,
                                                                                posterior_weight_samples,
                                                                                post_summary_mode=0,
                                                                                actFun=actFun,
                                                                                output_act_fun=output_act_fun)
            post_predictions.append(post_prob_predictions)
            true_labels.append(test_labels)
            post_softmax_out.append(post_softmax_probs)
            #pred_labels.append(np.argmax(post_prob_predictions,axis=1))
        post_predictions = np.concatenate(post_predictions)
        true_labels = np.concatenate(true_labels)
        post_softmax_out = np.concatenate(post_softmax_out,axis=1)
        #pred_labels = np.concatenate(pred_labels)
    else:
        weight_pickle = pkl_path
        bnn_obj, mcmc_obj, logger_obj = bn.load_obj(weight_pickle)
        test_features = bnn_obj._test_data
        true_labels = bnn_obj._test_labels
        posterior_weight_samples = logger_obj._post_weight_samples
        actFun = bnn_obj._act_fun
        output_act_fun = bnn_obj._output_act_fun
        post_softmax_out,post_predictions = bn.get_posterior_cat_prob(test_features,
                                                        posterior_weight_samples,
                                                        post_summary_mode=0,
                                                        actFun=actFun,
                                                        output_act_fun=output_act_fun)
    target_acc_thres_tbl = get_confidence_threshold(post_predictions,true_labels,target_acc=None)
    #acc_thres_tbl = pd.DataFrame(target_acc_thres_tbl)
    #acc_thres_tbl.columns = ['post_cutoff','acc_threshold','retained_samples']
    #acc_thres_tbl.to_csv(os.path.join(cv_folder,'acc_thres_tbl.txt'),sep='\t',index=False,header=False)
    cat_sample_out = bn.sample_from_categorical(posterior_weights=post_softmax_out)
    pred_class_counts = cat_sample_out['class_counts']
    avail_labels = np.arange(post_softmax_out[0].shape[1])
    true_label_distr = [list(true_labels).count(i) for i in avail_labels]
    return (target_acc_thres_tbl,pred_class_counts,true_label_distr)




def predict_bnn(features, model_path, posterior_threshold=0, post_summary_mode = 0):
    weight_pickle = model_path
    bnn_obj, mcmc_obj, logger_obj = bn.load_obj(weight_pickle)
    posterior_weight_samples = logger_obj._post_weight_samples
    actFun = bnn_obj._act_fun
    output_act_fun = bnn_obj._output_act_fun
    post_softmax_probs,post_prob_predictions = bn.get_posterior_cat_prob(features,
                                                    posterior_weight_samples,
                                                    post_summary_mode=post_summary_mode,
                                                    actFun=actFun,
                                                    output_act_fun=output_act_fun)
    cat_sample_out = bn.sample_from_categorical(posterior_weights=post_softmax_probs)
    pred_class_counts = cat_sample_out['class_counts']
    predictions = np.argmax(post_prob_predictions,axis=1)
    if posterior_threshold:
        high_pp_indices = np.where(np.max(post_prob_predictions, axis=1) > posterior_threshold)[0]
        predictions = bn.turn_low_pp_instances_to_nan(predictions,high_pp_indices)
        

    out_dict = {
        'raw_predictions':post_softmax_probs,
        'sampled_cat_freqs':pred_class_counts,
        'posterior_probs':post_prob_predictions,
        'class_predictions':predictions
    }

    return  out_dict
