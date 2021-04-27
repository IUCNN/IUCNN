#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:42:23 2021

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import np_bnn as bn




# detail_____________
pickle_dir = '/Users/xhofmt/GitHub/IUCNN/orchid_data/BNN_orchid_detail'

#combine MCMCs
combined_pkl = bn.combine_pkls(dir=pickle_dir)

pred_feature_file = "/Users/xhofmt/GitHub/IUCNN/orchid_data/data_files/iucnn_predict_bnn_features.txt"
pred_features = pd.read_csv(pred_feature_file,sep='\t').values

pred_out = bn.predictBNN(pred_features,
                                   combined_pkl,
                                   target_acc = None,
                                   threshold=0.95,
                                   bf=150,
                                   post_summary_mode=0,
                                   fname="",
                                   wd="",
                                   verbose=1)

pred_cat_probs = pred_out['post_prob_predictions']
predicted_labels = np.argmax(pred_cat_probs,axis=1)
plt.hist(predicted_labels)




# broad__________________
pickle_dir_broad = '/Users/xhofmt/GitHub/IUCNN/orchid_data/BNN_orchid_broad'

#combine MCMCs
combined_pkl_broad = bn.combine_pkls(dir=pickle_dir_broad)

pred_out_broad = bn.predictBNN(pred_features,
                                   combined_pkl_broad,
                                   target_acc = None,
                                   threshold=0.95,
                                   bf=150,
                                   post_summary_mode=0,
                                   fname="",
                                   wd="",
                                   verbose=1)

pred_cat_probs_broad = pred_out_broad['post_prob_predictions']
predicted_labels_broad = np.argmax(pred_cat_probs_broad,axis=1)
plt.hist(predicted_labels_broad)


