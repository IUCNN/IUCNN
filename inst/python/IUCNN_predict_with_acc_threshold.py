#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:47:06 2021

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import os
import numpy as np
import np_bnn as bn


# input data
target_acc = 0.9
pkl_file = '/Users/xhofmt/GitHub/IUCNN/manual_tests/iucnn_out_bnn/BNN_p1_h0_l20_5_s1_binf_1234.pkl'
bnn_obj, mcmc_obj, logger_obj = bn.load_obj(pkl_file)
feature_data = bnn_obj._test_data
post_pr = bn.predictBNN(feature_data,
                        pkl_file,
                        target_acc=target_acc,
                        post_summary_mode=1)






















# posterior_threshold = bn.get_posterior_threshold(pkl_file,target_acc)
# bnn_obj, mcmc_obj, logger_obj = bn.load_obj(pkl_file)
# feature_data = bnn_obj._test_data
# # run prediction so it produces the output prob file
# supported_instance_indices = bn.get_supported_instance_indices(feature_data,pkl_file,posterior_threshold)
# supported_instances = feature_data[supported_instance_indices,:]
# post_pr = bn.predictBNN(supported_instances,
#                         pickle_file=pkl_file,
#                         post_summary_mode=1)


# post_summary_mode = 0
# post_softmax_probs = np.load('/Users/xhofmt/GitHub/IUCNN/manual_tests/iucnn_out_bnn/BNN_p1_h0_l20_5_s1_binf_1234_pred_pr.npy')

# post_softmax_probs[0]


# if post_summary_mode == 0: # use argmax for each posterior sample
#     class_call_posterior = np.argmax(post_softmax_probs,axis=2).T
#     n_posterior_samples,n_instances,n_classes = post_softmax_probs.shape
#     posterior_prob_classes = np.zeros([n_instances,n_classes])
#     classes_and_counts = [[np.unique(i,return_counts=True)[0],np.unique(i,return_counts=True)[1]] for i in class_call_posterior]
#     for i,class_count in enumerate(classes_and_counts):
#         for j,class_index in enumerate(class_count[0]):
#             posterior_prob_classes[i,class_index] = class_count[1][j]
#     posterior_prob_classes = posterior_prob_classes/n_posterior_samples
# elif post_summary_mode == 1: # use mean of softmax across posterior samples
#     posterior_prob_classes = np.mean(post_softmax_probs, axis=0)





# np.argmax(post_softmax_probs[0],axis=1)



# a =[1,2,3,4]
# select_instance_indices = a
# if a:
#     print('this')
    
    
    
    
#         if select_instance_indices:
#             pred_temp = np.zeros(pred.shape)
#             pred_temp[:] = np.nan
#             pred_temp[select_instance_indices] = pred[select_instance_indices]
#             pred = pred_temp
            
            
            
# def get_supported_instance_indices(feature_data,pkl_file,posterior_threshold):
#     post_pr = predictBNN(feature_data,
#                          pickle_file=pkl_file,
#                          post_summary_mode=1,
#                          verbose=0)
#     indx_supported = np.where(np.max(post_pr['post_prob_predictions'], axis=1) > posterior_threshold)[0]
#     # get index of supported species
#     return list(indx_supported)

            