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


def combine_pkls(files=None, dir=None, tag=""):
    if dir is not None:
        files = glob.glob(os.path.join(dir, "*%s*.pkl" % tag))
        print("Combining files: ", files)
    i = 0
    w_list = []
    out_file = os.path.join(os.path.dirname(files[0]), "combine_pkl%s.pkl" % tag)
    for f in files:
        if f != out_file:
            a, b, w = bn.load_obj(f)
            if i == 0:
                bnn_obj = a
                mcmc_obj = b
                logger_obj = w
            w_list.append(w._post_weight_samples)
    bn.SaveObject([bnn_obj,mcmc_obj,logger_obj],pklfile=out_file)
    return out_file

pickle_dir = '/Users/xhofmt/GitHub/IUCNN/orchid_data/BNN_orchid_detail'

#combine MCMCs
all_pkls = combine_pkls(dir=pickle_dir)
#all_pkls = glob.glob(os.path.join(pickle_dir,'*.pkl'))

pred_feature_file = "/Users/xhofmt/GitHub/IUCNN/orchid_data/data_files/iucnn_predict_bnn_features.txt"
pred_features = pd.read_csv(pred_feature_file,sep='\t')
dat = bn.get_data(f=pred_features,
                    testsize=0,
                    header=1,  # input data has a header
                    instance_id=0,
                    from_file = False)  # input data includes names of instances


for i in all_pkls:
    bnn_obj,mcmc_obj,logger_obj = bn.load_obj(i)
    post_samples = logger_obj._post_weight_samples
    
    
    
    post_samples[1]['weights'][1].shape
    
    
actFun = bnn_obj._act_fun
output_act_fun = bnn_obj._output_act_fun
out_name = os.path.splitext(pickle_file)[0]
out_name = os.path.basename(out_name)
if wd != "":
    predictions_outdir = wd
else:
    predictions_outdir = os.path.dirname(pickle_file)

post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(predict_features,
                                                                  post_samples,
                                                                  post_summary_mode=post_summary_mode,
                                                                  actFun=actFun,
                                                                  output_act_fun=output_act_fun)






post_pr_test = bn.predictBNN(dat['data'],
                                  pickle_file=all_pkls,
                                  post_summary_mode=0)

prob_file = "features201223__pred_mean_pr.txt"
tmp = np.genfromtxt(prob_file, skip_header=False, dtype=str)
probs = tmp[:,1:].astype(float)
sp_names = tmp[:,0].astype(str)

# check how many lost data points with threshold
indx_supported = np.where(np.max(probs, axis=1) > avg_threshold)[0]
print(len(indx_supported)/len(sp_names))
print(len(indx_supported))

pred_status = np.argmax(probs[indx_supported], axis=1)
np.unique(pred_status, return_counts=True)

