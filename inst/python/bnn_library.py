import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import pickle
small_number = 1e-10
import random, sys
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import os
try:
    import tensorflow as tf
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # avoid error about multiple copies of the OpenMP runtime 
    except:
        pass
except:
    print(' ')

# Activation functions
class genReLU():
    def __init__(self, prm=np.zeros(1), trainable=False):
        self._prm = prm
        self._acc_prm = prm
        self._trainable = trainable
        # if alpha < 1 and non trainable: leaky ReLU (https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
        # if trainable: parameteric ReLU (https://arxiv.org/pdf/1502.01852.pdf)
        if prm[0] == 0 and not trainable:
            self._simpleReLU = True
        else:
            self._simpleReLU = False

    def eval(self, z, layer_n):
        if self._simpleReLU:
            z[z < 0] = 0
        else:
            z[z < 0] = self._prm[layer_n] * z[z < 0]
        return z
    def reset_prm(self, prm):
        self._prm = prm

    def reset_accepted_prm(self):
        self._acc_prm = self._prm + 0


# likelihood function (Categorical)
# TODO: refactor this as a class
def calc_likelihood(prediction, labels, sample_id, class_weight=[], lik_temp=1):
    if len(class_weight):
        return lik_temp * np.sum(np.log(prediction[sample_id, labels])*class_weight[labels])
    else:
        # if lik_temp != 1:
        #     tempered_prediction = lik_temp ** prediction
        #     normalized_tempered_prediction = np.einsum('xy,x->xy', tempered_prediction, 1 / np.sum(tempered_prediction,axis=1))
        #     return np.sum(np.log(normalized_tempered_prediction[sample_id, labels]))
        # else:
        return lik_temp * np.sum(np.log(prediction[sample_id, labels]))



def MatrixMultiplication(x1,x2):
    z1 = np.einsum('nj,ij->ni', x1, x2, optimize=False)
    # same as:
    # for i in range(n_samples):
    # 	print(np.einsum('j,ij->i', x[i], w_in_l1))
    return z1

# SoftMax function
def SoftMax(z):
    # return ((np.exp(z).T)/np.sum(np.exp(z),axis=1)).T
    return scipy.special.softmax(z, axis=1)


def RunHiddenLayer(z0, w01, actFun, layer_n):
    z1 = MatrixMultiplication(z0, w01)
    if actFun:
        return actFun.eval(z1, layer_n)
    else:
        return z1


def CalcAccuracy(y,lab):
    if len(y.shape) == 3: # if the posterior softmax array is used, return array of accuracies
        acc = np.array([np.sum(i==lab)/len(i) for i in np.argmax(y,axis=2)])
    else:
        prediction = np.argmax(y, axis=1)
        acc = np.sum(prediction==lab)/len(prediction)
    return acc

def CalcLabelAccuracy(y,lab):
    prediction = np.argmax(y, axis=1)
    label_accs = []
    for label in np.unique(lab):
        cat_lab = lab[lab==label]
        cat_prediction = prediction[lab==label]
        acc = np.sum(cat_prediction==cat_lab)/len(cat_prediction)
        label_accs.append(acc)
    return np.array(label_accs)

def CalcConfusionMatrix(y,lab):
    prediction = np.argmax(y, axis=1)
    y_actu = pd.Series(lab, name='True')
    y_pred = pd.Series(prediction, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, margins=True, rownames=['True'], colnames=['Predicted'])
    return df_confusion

def CalcLabelFreq(y):
    prediction = np.argmax(y, axis=1)
    f = np.zeros(y.shape[1])
    tmp = np.unique(prediction, return_counts = True)
    f[tmp[0]] = tmp[1]
    return f/len(prediction)


def SaveObject(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def RunPredict(data, weights, actFun):
    # weights: list of 2D arrays
    tmp = data+0
    for i in range(len(weights)-1):
        tmp = RunHiddenLayer(tmp,weights[i],actFun, i)
    tmp = RunHiddenLayer(tmp, weights[i+1], False, i+1)
    # output
    y_predict = SoftMax(tmp)
    return y_predict

def RunPredictInd(data, weights, ind, actFun):
    # weights: list of 2D arrays
    tmp = data+0
    for i in range(len(weights)-1):
        if i ==0:
            tmp = RunHiddenLayer(tmp,weights[i]*ind,actFun, i)
        elif i < len(weights)-1:
            tmp = RunHiddenLayer(tmp,weights[i],actFun, i)
    tmp = RunHiddenLayer(tmp, weights[i+1], False, i+1)
    # output
    y_predict = SoftMax(tmp)
    return y_predict


def RecurMeanVar(it, list_mu_var_old, list_curr_param, indx):
    min_var = 1
    [Ix, Iy] = indx
    [mu_it_old, var_it_old] = list_mu_var_old
    mu_it, var_it = mu_it_old+0, var_it_old+0
    curr_param = list_curr_param[Ix, Iy]
    it = it + 1
    mu_it[Ix, Iy] = (it - 1)/it * mu_it_old[Ix, Iy] + 1/it * curr_param
    var_it[Ix, Iy] = (it - 1)/it * var_it_old[Ix, Iy] + 1/(it - 1) * (curr_param - mu_it[Ix, Iy])**2
    return [mu_it, var_it]

def calcHPD(data, level):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nToo little data to calculate marginal rates.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
            rk = d[k+nIn-1] - d[k]
            if rk < r :
                r = rk
                i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])


def CalcTP(y,lab, threshold=0.95):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)),prediction]
    z = np.zeros(len(prediction))
    z[max_p > threshold] = 1
    return np.sum(z[prediction == lab])/len(prediction)

def CalcFP(y,lab, threshold=0.95):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)),prediction]
    z = np.zeros(len(prediction))
    z[max_p > threshold] = 1
    return np.sum(z[prediction != lab])/len(prediction)


def CalcTP_BF(y, y_p, lab, threshold=150):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)), prediction]
    prior = y_p[range(len(prediction)), prediction]
    bf = (max_p / (small_number + 1 - max_p)) / (prior / (small_number + 1 - prior))
    z = np.zeros(len(prediction))
    z[bf > threshold] = 1
    return np.sum(z[prediction == lab]) / len(prediction)


def CalcFP_BF(y, y_p, lab, threshold=150):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)), prediction]
    prior = y_p[range(len(prediction)), prediction]
    bf = (max_p / (small_number + 1 - max_p)) / (prior / (small_number + 1 - prior))
    z = np.zeros(len(prediction))
    z[bf > threshold] = 1
    return np.sum(z[prediction != lab]) / len(prediction)

def CalcAccAboveThreshold(y,lab, threshold=0.95):
    max_prob = np.max(y, axis=1)
    supported_estimate = np.where(max_prob > threshold)

    prediction = np.argmax(y, axis=1)[supported_estimate]
    max_p = y[range(len(prediction)),prediction]
    z = np.zeros(len(prediction))
    z[max_p > threshold] = 1
    res= np.sum(z[prediction == lab[supported_estimate]])/len(prediction)

    print(res)


def get_posterior_cat_prob(pred_features,
                           pickle_file=None,
                           post_samples=None,
                           feature_index_to_shuffle=None,
                           post_summary_mode=0, # mode 0 is argmax, mode 1 is mean softmax
                           unlink_features_within_block=False):
    if len(pred_features) ==0:
        print("Data not found.")
        return 0
    else:
        n_features = pred_features.shape[1]
    predict_features = pred_features.copy()
    # shuffle features if index is provided
    if feature_index_to_shuffle: # shuffle the feature values for the given feature between all instances
        if unlink_features_within_block and type(feature_index_to_shuffle)==list:
            for feature_index in feature_index_to_shuffle: # shuffle each column by it's own random indices
                predict_features[:,feature_index] = np.random.permutation(predict_features[:,feature_index])
        else:
            predict_features[:,feature_index_to_shuffle] = np.random.permutation(predict_features[:,feature_index_to_shuffle])
    # load posterior weights
    if pickle_file is not None:
        post_samples = load_obj(pickle_file)
    post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
    post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
    if n_features < post_weights[0][0].shape[1]:
        "add bias node"
        predict_features = np.c_[np.ones(predict_features.shape[0]), predict_features]
    post_cat_probs = []
    for i in range(len(post_weights)):
        actFun = genReLU(prm=post_alphas[i])
        pred = RunPredict(predict_features, post_weights[i], actFun=actFun)
        post_cat_probs.append(pred)
    post_softmax_probs = np.array(post_cat_probs)
    if post_summary_mode == 0: # use argmax for each posterior sample
        class_call_posterior = np.argmax(post_softmax_probs,axis=2).T
        n_posterior_samples,n_instances,n_classes = post_softmax_probs.shape
        posterior_prob_classes = np.zeros([n_instances,n_classes])
        classes_and_counts = [[np.unique(i,return_counts=True)[0],np.unique(i,return_counts=True)[1]] for i in class_call_posterior]
        for i,class_count in enumerate(classes_and_counts):
            for j,class_index in enumerate(class_count[0]):
                posterior_prob_classes[i,class_index] = class_count[1][j]
        posterior_prob_classes = posterior_prob_classes/n_posterior_samples
    elif post_summary_mode == 1: # use mean of softmax across posterior samples
        posterior_prob_classes = np.mean(post_softmax_probs, axis=0)
    elif post_summary_mode == 2: # resample classification based on softmax/categorical probabilities (posterior predictive)
        res = sample_from_categorical(posterior_weights=post_softmax_probs)
        posterior_prob_classes = res['predictions']
    
    return(post_softmax_probs,posterior_prob_classes)

        # if summary_mode == 0: # use argmax for each posterior sample
        #     pred = np.argmax(pred, axis=1)



def predictBNN(predict_features, pickle_file=None, post_samples=None, test_labels=[], instance_id=[],
               pickle_file_prior=0, threshold=0.95, bf=150, fname="",post_summary_mode=0,
               wd=""):

    post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(predict_features,
                                                                      pickle_file,
                                                                      post_samples,
                                                                      post_summary_mode=post_summary_mode)
    
    if pickle_file:
        predictions_outdir = os.path.dirname(pickle_file)
        out_name = os.path.splitext(pickle_file)[0]
        out_name = os.path.basename(out_name)
    else:
        predictions_outdir = wd
        out_name = ""
    if fname != "":
        fname = fname + "_"
    out_file_post_pr = os.path.join(predictions_outdir, fname + out_name + '_pred_pr.npy')
    out_file_mean_pr = os.path.join(predictions_outdir, fname + out_name + '_pred_mean_pr.txt')

    if len(test_labels) > 0:
        CalcAccAboveThreshold(post_prob_predictions, test_labels, threshold=0.95)
        accuracy = CalcAccuracy(post_prob_predictions, test_labels)
        TPrate = CalcTP(post_prob_predictions, test_labels, threshold=threshold)
        FPrate = CalcFP(post_prob_predictions, test_labels, threshold=threshold)
        mean_accuracy = np.mean(accuracy)
        print("Accuracy:", mean_accuracy)
        print("True positive rate:", np.mean(TPrate))
        print("False positive rate:", np.mean(FPrate))
        print("Confusion matrix:\n", CalcConfusionMatrix(post_prob_predictions, test_labels))
        out_file_acc = os.path.join(predictions_outdir, fname + out_name + '_accuracy.txt')
        with open(out_file_acc,'w') as outf:
            outf.writelines("Mean accuracy: %s (TP: %s; FP: %s)" % (mean_accuracy, TPrate, FPrate))
    else:
        mean_accuracy = np.nan
    if pickle_file_prior:
        prior_samples = load_obj(pickle_file_prior)
        prior_weights = [prior_samples[i]['weights'] for i in range(len(prior_samples))]
        prior_alphas = [prior_samples[i]['alphas'] for i in range(len(prior_samples))]
        prior_predictions = []
        for i in range(len(prior_weights)):
            actFun = genReLU(prm=prior_alphas[i])
            pred = RunPredict(predict_features, prior_weights[i], actFun=actFun)
            prior_predictions.append(pred)
    
        prior_predictions = np.array(prior_predictions)
        prior_prob_predictions = np.mean(prior_predictions, axis=0)
    
        TPrate = CalcTP_BF(post_prob_predictions, prior_prob_predictions, test_labels, threshold=bf)
        FPrate = CalcFP_BF(post_prob_predictions, prior_prob_predictions, test_labels, threshold=bf)
    
        print("True positive rate (BF):", np.mean(TPrate))
        print("False positive rate (BF):", np.mean(FPrate))

    if len(instance_id):
        post_prob_predictions_id = np.hstack((instance_id.reshape(len(instance_id), 1),
                                              np.round(post_prob_predictions,4).astype(str)))
        np.savetxt(out_file_mean_pr, post_prob_predictions_id, fmt='%s',delimiter='\t')
    else:
        np.savetxt(out_file_mean_pr, post_prob_predictions, fmt='%.3f')
    # print the arrays to file
    np.save(out_file_post_pr, post_softmax_probs)
    print("Predictions saved in files:")
    print('   ', out_file_post_pr)
    print('   ', out_file_mean_pr,"\n")
    return {'post_prob_predictions': post_prob_predictions, 'mean_accuracy': mean_accuracy}


def feature_importance(input_features,
                       weights_pkl=None,
                       weights_posterior=None,
                       true_labels=[],
                       fname_stem='',
                       feature_names=[],
                       verbose=False,
                       post_summary_mode=0,
                       n_permutations=100,
                       feature_blocks=[],
                       predictions_outdir='',
                       unlink_features_within_block=False):
    features = input_features.copy()
    feature_indices = np.arange(features.shape[1])
    # if no names are provided, name them by index
    if len(feature_names) == 0:
        feature_names = feature_indices.astype(str)
    if len(feature_blocks) > 0:
        selected_features = []
        selected_feature_names = []
        for block_indices in feature_blocks:
            selected_features.append(list(np.array(feature_indices)[block_indices]))
            selected_feature_names.append(list(np.array(feature_names)[block_indices]))
    else:
        selected_features = [[i] for i in feature_indices]
        selected_feature_names = [[i] for i in feature_names]
    feature_block_names = [','.join(np.array(i).astype(str)) for i in selected_feature_names] #join the feature names into one string for each block for output df
    # get accuracy with all features
    post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(input_features, weights_pkl, weights_posterior,
                                                                      post_summary_mode=post_summary_mode)
    ref_accuracy = CalcAccuracy(post_prob_predictions, true_labels)
    if verbose:
        print("Reference accuracy (mean):", np.mean(ref_accuracy))
    # go through features and shuffle one at a time
    accuracies_wo_feature = []
    for block_id,feature_block in enumerate(selected_features):
        if verbose:
            print('Processing feature block %i',block_id+1)
        n_accuracies = []
        for rep in np.arange(n_permutations):
            post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(input_features, weights_pkl, weights_posterior,
                                                                              feature_index_to_shuffle=feature_block,
                                                                              post_summary_mode=post_summary_mode,
                                                                              unlink_features_within_block=unlink_features_within_block)
            accuracy = CalcAccuracy(post_prob_predictions, true_labels)
            n_accuracies.append(accuracy)
        accuracies_wo_feature.append(n_accuracies)
    accuracies_wo_feature = np.array(accuracies_wo_feature)
    delta_accs = ref_accuracy-np.array(accuracies_wo_feature)    
    delta_accs_means = np.mean(delta_accs,axis=1)
    delta_accs_stds = np.std(delta_accs,axis=1)
    accuracies_wo_feature_means = np.mean(accuracies_wo_feature,axis=1)
    accuracies_wo_feature_stds = np.std(accuracies_wo_feature,axis=1)
    feature_importance_df = pd.DataFrame(np.array([np.arange(0,len(selected_features)),feature_block_names,
                                                   delta_accs_means,delta_accs_stds,
                                                   accuracies_wo_feature_means,accuracies_wo_feature_stds]).T,
                                         columns=['feature_block_index','feature_name','delta_acc_mean','delta_acc_std',
                                                  'acc_with_feature_randomized_mean','acc_with_feature_randomized_std'])
    feature_importance_df.iloc[:,2:] = feature_importance_df.iloc[:,2:].astype(float)
    feature_importance_df_sorted = feature_importance_df.sort_values('delta_acc_mean',ascending=False)
    # define outfile name
    if predictions_outdir == "":
        predictions_outdir = os.path.dirname(weights_pkl)
    if not os.path.exists(predictions_outdir) and predictions_outdir != "":
        os.makedirs(predictions_outdir)
    if fname_stem != "":
        fname_stem = fname_stem + "_"
    feature_importance_df_filename = os.path.join(predictions_outdir, fname_stem + '_feature_importance.txt')
    # format the last two columns as numeric for applyign float printing formatting options
    feature_importance_df_sorted['delta_acc_mean'] = pd.to_numeric(feature_importance_df_sorted['delta_acc_mean'])
    feature_importance_df_sorted['acc_with_feature_randomized_mean'] = pd.to_numeric(feature_importance_df_sorted['acc_with_feature_randomized_mean'])
    feature_importance_df_sorted.to_csv(feature_importance_df_filename,sep='\t',index=False,header=True,float_format='%.6f')
    print("Output saved in: %s" % feature_importance_df_filename)
    return feature_importance_df_sorted


def get_weights_from_tensorflow_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    n_nodes_list = []
    init_weights = []
    bias_node_weights = []
    for layer in model.layers:
        #layer_name = layer.weights[0].name 
        layer_shape = np.array(layer.weights[0].shape)
        weights = layer.weights[0].numpy().T
        n_nodes_list.append(layer_shape[1])
        init_weights.append(weights)
        if len(layer.weights) == 2: #bias node layer
            bias_node = layer.weights[1].numpy()
            bias_node_weights.append(bias_node)
    return([n_nodes_list[:-1],init_weights,bias_node_weights])


def get_accuracy_threshold(probs, labels, threshold=0.75):
    indx = np.where(np.max(probs, axis=1)>threshold)[0]
    res_supported = probs[indx,:]
    labels_supported = labels[indx]
    pred = np.argmax(res_supported, axis=1)
    accuracy = len(pred[pred == labels_supported])/len(pred)
    dropped_frequency = len(pred)/len(labels)
    cm = CalcConfusionMatrix(res_supported, labels_supported)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency, 'confusion_matrix': cm}

def sample_from_categorical(posterior_weights=None, post_prob_file=None, verbose=False):
    if posterior_weights is not None:
        pass
    elif post_prob_file:
        posterior_weights = np.load(post_prob_file)
    else:
        print("Input pickle file or posterior weights required.")
    n_post_samples = posterior_weights.shape[0]
    n_instances = posterior_weights.shape[1]
    n_classes = posterior_weights.shape[2]

    res = np.zeros((n_instances, n_post_samples))
    point_estimates = np.zeros((n_instances, n_classes))
    for instance_j in range(posterior_weights.shape[1]):
        if instance_j % 1000 == 0 and verbose is True:
            print(instance_j)
        post_sample = posterior_weights[:, instance_j, :]
        p = np.cumsum(post_sample, axis=1)
        r = np.random.random(len(p))
        q = p - r.reshape(len(r), 1)
        q[q < 0] = 1  # arbitrarily large number
        classification = np.argmin(q, axis=1)
        res[instance_j, :] = classification
        # mode (point estimate)
        counts = np.bincount(classification, minlength=n_classes)
        point_estimates[instance_j, :] = counts / np.sum(counts)
    
    class_counts = np.zeros((n_post_samples, n_classes))
    for i in range(res.shape[1]):
        class_counts[i] = np.bincount(res[:, i].astype(int), minlength=n_classes)
    
    return {'predictions': point_estimates, 'class_counts': class_counts, 'post_predictions': res}
import scipy.stats
from numpy.random import MT19937



class npBNN():
    def __init__(self, dat, n_nodes=[50, 5],
                 use_bias_node=1, init_std=0.1, p_scale=1, prior_ind1=0.5,
                 prior_f=1, hyper_p=0, freq_indicator=0, w_bound=np.infty,
                 pickle_file="", seed=1234, use_class_weights=0, actFun=genReLU(),init_weights=None):
        # prior_f: 0) uniform 1) normal 2) cauchy
        # to change the boundaries of a uniform prior use -p_scale
        # hyper_p: 0) no hyperpriors, 1) 1 per layer, 2) 1 per input node, 3) 1 per node
        # freq_indicator -> update freq indicators: 0) no indicators, 0-1
        data, labels, test_data, test_labels = dat['data'], dat['labels'], dat['test_data'], dat['test_labels']
        self._seed = seed
        np.random.seed(self._seed)
        self._data = data
        self._labels = labels.astype(int)
        self._test_data = test_data
        if len(test_labels) > 0:
            self._test_labels = test_labels.astype(int)
        else:
            self._test_labels = []
        self._size_output = len(np.unique(self._labels))
        self._init_std = init_std
        self._n_layers = len(n_nodes) + 1
        self._n_nodes = n_nodes
        self._use_bias_node = use_bias_node
        if use_bias_node:
            self._data = np.c_[np.ones(self._data.shape[0]), self._data]
            if len(test_labels) > 0:
                self._test_data = np.c_[np.ones(self._test_data.shape[0]), self._test_data]
            else:
                self._test_data = []

        self._n_samples = self._data.shape[0]
        self._n_features = self._data.shape[1]
        self._sample_id = np.arange(self._n_samples)
        self._w_bound = w_bound
        self._freq_indicator = freq_indicator
        self._hyper_p = hyper_p
        self._sample_id = np.arange(self._n_samples)
        self._prior = prior_f
        self._p_scale = p_scale
        self._prior_ind1 = prior_ind1

        if use_class_weights:
            class_counts = np.unique(self._labels, return_counts=True)[1]
            self._class_w = 1 / (class_counts / np.max(class_counts))
            self._class_w = self._class_w / np.mean(self._class_w)
            print("Using class weights:", self._class_w)
        else:
            self._class_w = []

        # init weights
        if init_weights is None:
            if pickle_file == "":
                # 1st layer
                w_layers = [np.random.normal(0, self._init_std, (self._n_nodes[0], self._n_features))]
                # add hidden layers
                for i in range(1, self._n_layers - 1):
                    w_layers.append(np.random.normal(0, self._init_std, (self._n_nodes[i], self._n_nodes[i - 1])))
                # last layer
                w_layers.append(np.random.normal(0, self._init_std, (self._size_output, self._n_nodes[-1])))
            else:
                post_samples = load_obj(pickle_file)
                post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
                w_layers = post_weights[-1]
        else:
            w_layers = init_weights
            #self._n_layers -= 1
        self._w_layers = w_layers

        self._indicators = np.ones(self._w_layers[0].shape)
        if pickle_file == "":
            self._act_fun = actFun
        else:
            post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
            self._act_fun = genReLU(prm=post_alphas[-1])


        # init prior function
        if self._prior == 0:
            'Uniform'
            self._w_bound = self._p_scale
        else:
            if self._prior == 1:
                'Normal'
                self._prior_f = scipy.stats.norm.logpdf
            if self._prior == 2:
                'Cauchy'
                self._prior_f = scipy.stats.cauchy.logpdf
            elif self._prior == 3:
                'Laplace'
                self._prior_f = scipy.stats.laplace.logpdf
            else:
                print('Using default prior N(0,s)')
                self._prior_f = scipy.stats.norm.logpdf
        # init prior scales: will be updated if hyper-priors
        self._prior_scale = np.ones(self._n_layers) * self._p_scale

        if len(self._test_data) > 0:
            print("\nTraining set:", self._n_samples, "test set:", self._test_data.shape[0])
        else:
            print("\nTraining set:", self._n_samples, "test set:", None)
        print("Number of features:", self._n_features)
        
        n_params = np.sum(np.array([np.size(i) for i in self._w_layers]))
        if self._act_fun._trainable:
            n_params += self._n_layers
        print("N. of parameters:", n_params)
        for w in self._w_layers: print(w.shape)

    # init prior functions
    def calc_prior(self, w=0, ind=[]):
        if w == 0:
            w = self._w_layers
        if len(ind) == 0:
            ind = self._indicators
        if self._prior == 0:
            logPrior = 0
        else:
            logPrior = 0
            for i in range(self._n_layers):
                logPrior += np.sum(self._prior_f(w[i], 0, scale=self._prior_scale[i]))
        if self._freq_indicator:
            logPrior += np.sum(ind) * np.log(self._prior_ind1) + \
                        (self._indicators.size - np.sum(ind)) * np.log(1 - self._prior_ind1)
        return logPrior

    def sample_prior_scale(self):
        if self._prior != 1:
            print("Hyper-priors available only for Normal priors.")
            quit()
        if self._hyper_p == 1:
            '1 Hyp / layer'
            prior_scale = list()
            for x in self._w_layers:
                prior_scale.append(GibbsSampleNormStdGammaVector(x.flatten()))
            self._prior_scale = prior_scale
        elif self._hyper_p == 2:
            '1 Hyp / input node / layer'
            self._prior_scale = [np.ones(w.shape[1]) for w in self._w_layers]
            prior_scale = list()
            for x in self._w_layers:
                prior_scale.append(GibbsSampleNormStdGamma2D(x))
            self._prior_scale = prior_scale
        elif self._hyper_p == 3:
            '1 Hyp / weight / layer'
            self._prior_scale = [np.ones(w.shape) for w in self._w_layers]
            prior_scale = list()
            for x in self._w_layers:
                prior_scale.append(GibbsSampleNormStdGammaONE(x))
            self._prior_scale = prior_scale
        else:
            pass

    def reset_weights(self, w):
        self._w_layers = w

    def reset_indicators(self, ind):
        self._indicators = ind

    def update_data(self, data_dict):
        self._data = data_dict['data']
        self._labels = data_dict['labels']
        self._test_data = data_dict['test_data']
        self._test_labels = data_dict['test_labels']


class MCMC():
    def __init__(self, bnn_obj, update_f=None, update_ws=None,
                 temperature=1, n_iteration=100000, sampling_f=100, print_f=1000, n_post_samples=1000,
                 update_function=UpdateNormal, sample_from_prior=0, run_ID="", init_additional_prob=0,
                 likelihood_tempering=1, mcmc_id=0, randomize_seed=False):
        if update_ws is None:
            update_ws = [0.075] * bnn_obj._n_layers
        if update_f is None:
            update_f = [0.05] * bnn_obj._n_layers
        if run_ID == "":
            self._runID = bnn_obj._seed
        else:
            self._runID = run_ID
        self._update_f = update_f
        self._update_ws = [np.ones(bnn_obj._w_layers[i].shape) * update_ws[i] for i in range(bnn_obj._n_layers)]
        self._update_n = [np.max([1, np.round(bnn_obj._w_layers[i].size * update_f[i]).astype(int)]) for i in
                          range(bnn_obj._n_layers)]
        self._temperature = temperature
        self._n_iterations = n_iteration
        self._sampling_f = sampling_f
        self._print_f = print_f
        self._current_iteration = 0
        self._y = RunPredict(bnn_obj._data, bnn_obj._w_layers, bnn_obj._act_fun)
        if sample_from_prior:
            self._logLik = 0
        else:
            self._logLik = calc_likelihood(self._y,
                                           bnn_obj._labels,
                                           bnn_obj._sample_id,
                                           bnn_obj._class_w,
                                           likelihood_tempering)
        self._logPrior = bnn_obj.calc_prior() + init_additional_prob
        self._logPost = self._logLik + self._logPrior
        self._accuracy = CalcAccuracy(self._y, bnn_obj._labels)
        self._label_acc = CalcLabelAccuracy(self._y, bnn_obj._labels)
        if len(bnn_obj._test_data) > 0:
            self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers, bnn_obj._indicators,bnn_obj._act_fun)
            self._test_accuracy = CalcAccuracy(self._y_test, bnn_obj._test_labels)
        else:
            self._y_test = []
            self._test_accuracy = 0
        self._label_freq = CalcLabelFreq(self._y)
        self.update_function = update_function
        self._sample_from_prior = sample_from_prior
        self._last_accepted = 1
        self._lik_temp = likelihood_tempering
        self._mcmc_id = mcmc_id
        self._randomize_seed = randomize_seed
        self._rs = RandomState(MT19937(SeedSequence(1234)))
        self._counter = 0
        self._n_post_samples = n_post_samples
        self._accepted_states = 0

    def mh_step(self, bnn_obj, additional_prob=0, return_bnn=False):
        if self._randomize_seed:
            self._rs = RandomState(MT19937(SeedSequence(self._current_iteration + self._mcmc_id)))

        hastings = 0
        w_layers_prime = []
        tmp = bnn_obj._data + 0
        indicators_prime = bnn_obj._indicators + 0
        
        # if trainable prm in activation function
        if bnn_obj._act_fun._trainable:
            prm_tmp, _, h = UpdateNormal1D(bnn_obj._act_fun._acc_prm, d=0.05, n=1, Mb=1, mb=0, rs=self._rs)
            r = 10
            additional_prob += np.log(r) * -np.sum(prm_tmp)*r # aka exponential Exp(r)
            hastings += h
            bnn_obj._act_fun.reset_prm(prm_tmp)
            
        for i in range(bnn_obj._n_layers):
            if np.random.random() > bnn_obj._freq_indicator or i > 0:
                update, indx, h = self.update_function(bnn_obj._w_layers[i], d=self._update_ws[i], n=self._update_n[i],
                                                    Mb=bnn_obj._w_bound, mb=-bnn_obj._w_bound, rs=self._rs)
                w_layers_prime.append(update)
                hastings += h
            else:
                w_layers_prime.append(bnn_obj._w_layers[i] + 0)
                indicators_prime = UpdateBinomial(bnn_obj._indicators, self._update_f[3], bnn_obj._indicators.shape)
            if i == 0:
                w_layers_prime_temp = w_layers_prime[i] * indicators_prime
            else:
                w_layers_prime_temp = w_layers_prime[i]
            if i < bnn_obj._n_layers-1:
                tmp = RunHiddenLayer(tmp, w_layers_prime_temp,bnn_obj._act_fun, i)
            else:
                tmp = RunHiddenLayer(tmp, w_layers_prime_temp, False, i)
        y_prime = SoftMax(tmp)

        logPrior_prime = bnn_obj.calc_prior(w=w_layers_prime, ind=indicators_prime) + additional_prob
        if self._sample_from_prior:
            logLik_prime = 0
        else:
            # TODO: expose self._lik_temp as parameter that can be estimated by MCMC
            logLik_prime = calc_likelihood(y_prime,
                                           bnn_obj._labels,
                                           bnn_obj._sample_id,
                                           bnn_obj._class_w,
                                           self._lik_temp)
        logPost_prime = logLik_prime + logPrior_prime
        rrr = np.log(self._rs.random())
        if (logPost_prime - self._logPost) * self._temperature + hastings >= rrr:
            # print(logPost_prime, self._logPost)
            bnn_obj.reset_weights(w_layers_prime)
            bnn_obj.reset_indicators(indicators_prime)
            bnn_obj._act_fun.reset_accepted_prm()
            self._logPost = logPost_prime
            self._logLik = logLik_prime
            self._logPrior = logPrior_prime
            self._y = y_prime
            self._accuracy = CalcAccuracy(self._y, bnn_obj._labels)
            self._label_acc = CalcLabelAccuracy(self._y, bnn_obj._labels)
            self._label_freq = CalcLabelFreq(self._y)
            if len(bnn_obj._test_data) > 0:
                self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers,
                                             bnn_obj._indicators, bnn_obj._act_fun)
                self._test_accuracy = CalcAccuracy(self._y_test, bnn_obj._test_labels)
            else:
                self._y_test = []
                self._test_accuracy = 0
            self._last_accepted = 1
            self._accepted_states += 1
        else:
            self._last_accepted = 0

        self._current_iteration += 1
        if return_bnn:
            return bnn_obj, self

    def gibbs_step(self, bnn_obj):
        bnn_obj.sample_prior_scale()
        self._logPrior = bnn_obj.calc_prior()
        self._logPost = self._logLik + self._logPrior
        self._current_iteration += 1

    def reset_update_n(self, n):
        self._update_n = n
    
    def reset_temperature(self,temp):
        self._temperature = temp
    

class postLogger():
    def __init__(self,
                 bnn_obj,
                 filename="BNN",
                 wdir="",
                 sample_from_prior=0,
                 add_prms=None,
                 continue_logfile=False,
                 log_all_weights=0):
        
        wlog, logfile, w_file, wweight = init_output_files(bnn_obj, filename, sample_from_prior,
                                                                            outpath=wdir, add_prms=add_prms,
                                                                            continue_logfile=continue_logfile,
                                                                            log_all_weights=log_all_weights)
        self._wlog = wlog
        self._logfile = logfile
        self._w_file = w_file
        self._wweight = wweight
        self.log_all_weights = log_all_weights
        self._counter = 0
        self._list_post_weights = list()

    def reset_counter(self):
        self._counter = 0

    def update_counter(self):
        self._counter += 1

    def log_sample(self, bnn_obj, mcmc_obj, add_prms=None):
        row = [mcmc_obj._current_iteration, mcmc_obj._logPost, mcmc_obj._logLik, mcmc_obj._logPrior,
               mcmc_obj._accuracy, mcmc_obj._test_accuracy] + list(mcmc_obj._label_acc)#list(mcmc_obj._label_freq)
        for i in range(bnn_obj._n_layers):
            row = row + [np.mean(bnn_obj._w_layers[i]), np.std(bnn_obj._w_layers[i])]
            if bnn_obj._hyper_p:
                if bnn_obj._hyper_p == 1:
                    row.append(bnn_obj._prior_scale[i])
                else:
                    row.append(np.mean(bnn_obj._prior_scale[i]))
        if bnn_obj._freq_indicator > 0:
            row.append(np.mean(bnn_obj._indicators))
        if add_prms:
            row = row + add_prms
        if bnn_obj._act_fun._trainable:
            row = row + list(bnn_obj._act_fun._acc_prm)
        row.append(mcmc_obj._accepted_states / mcmc_obj._current_iteration)
        row.append(mcmc_obj._mcmc_id)
        self._wlog.writerow(row)
        self._logfile.flush()

    def log_weights(self, bnn_obj, mcmc_obj, add_prms=None):
        # print(mcmc_obj._current_iteration, self._counter, len(self._list_post_weights))
        if not self.log_all_weights:
            if bnn_obj._freq_indicator:
                tmp = list()
                tmp.append(bnn_obj._w_layers[0] * bnn_obj._indicators)
                for i in range(1, bnn_obj._n_layers):
                    tmp.append(bnn_obj._w_layers[i])
            else:
                tmp = bnn_obj._w_layers
            post_prm = {'weights': tmp}
            # a ReLU prms
            post_prm['alphas'] = list(bnn_obj._act_fun._acc_prm)

            if add_prms:
                post_prm['additional_prm'] = list(add_prms)

            # print(mcmc_obj._current_iteration, self._counter, len(self._list_post_weights))
            if len(self._list_post_weights) < mcmc_obj._n_post_samples:
                self._list_post_weights.append(post_prm)
            else:
                self._list_post_weights[self._counter] = post_prm
            self.update_counter()
            if self._counter == mcmc_obj._n_post_samples:
                self.reset_counter()
            SaveObject(self._list_post_weights, self._w_file)
        else:
            row = [mcmc_obj._current_iteration]
            tmp = bnn_obj._w_layers[0] * bnn_obj._indicators[0]
            row = row + [j for j in list(tmp.flatten())]
            for i in range(1, bnn_obj._n_layers):
                row = row + [j for j in list(bnn_obj._w_layers[i].flatten())]
            self._wweight.writerow(row)
            self._w_file.flush()

import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import os


def plotResults(bnn_predictions_file, bnn_lower_file, bnn_upper_file, nn_predictions_file, predictions_outdir, filename_str):

	filename_str = os.path.basename(filename_str)
	try:
		plot_nn = 1
		nn_predictions = np.loadtxt(nn_predictions_file)
	except:
		plot_nn = 0
		nn_predictions = 0
	bnn_predictions = np.loadtxt(bnn_predictions_file)
	bnn_lower = np.loadtxt(bnn_lower_file)
	bnn_upper = np.loadtxt(bnn_upper_file)

	delta_lower = np.abs(bnn_predictions[:,0]-bnn_lower)
	delta_upper = np.abs(bnn_upper-bnn_predictions[:,0])

	fig=plt.figure()
	if plot_nn:
		plt.plot(nn_predictions[:,0],'rx',label='NN predictions')
	plt.plot(bnn_predictions[:,0],'gx',label='BNN predictions')
	plt.axhline(0.5,color='black')
	plt.ylabel('Probability of cat 0')
	plt.xlabel('Prediction instances')
	plt.legend()
	fig.savefig(os.path.join(predictions_outdir, '%s_cat_0_probs_plot.pdf'%filename_str))

	fig=plt.figure()
	if plot_nn:
		plt.plot(nn_predictions[:,0],'rx',label='NN predictions')
	#plt.plot(bnn_predictions[:,0],'gx',label='BNN predictions')
	plt.errorbar(np.arange(bnn_predictions[:,0].shape[0]),bnn_predictions[:,0], yerr=np.array([delta_lower,delta_upper]),
	             fmt='gx',ecolor='black',elinewidth=1,capsize=2,label='BNN predictions')
	plt.axhline(0.5,color='black')
	plt.ylabel('Probability of cat 0')
	plt.xlabel('Prediction instances')
	plt.legend()
	fig.savefig(os.path.join(predictions_outdir, '%s_cat_0_probs_hpd_bars_plot.pdf'%filename_str))

def summarizeOutput(predictions_outdir, pred_features, w_file, nn_predictions_file, use_bias_node):
	if not os.path.exists(predictions_outdir):
		os.makedirs(predictions_outdir)
	loaded_weights = np.array(load_obj(w_file))
	predict_features = np.load(pred_features)
	if use_bias_node:
		predict_features = np.c_[np.ones(predict_features.shape[0]), predict_features]
	# run prediction with these weights
	post_predictions = []
	for weights in loaded_weights:
		pred =  RunPredict(predict_features, weights)
		post_predictions.append(pred)
	post_predictions = np.array(post_predictions)
	out_name = os.path.splitext(w_file)[0]
	out_name = os.path.basename(out_name)
	
	out_file_post_pr = os.path.join(predictions_outdir, out_name + '_pred_pr.npy')
	out_file_mean_pr = os.path.join(predictions_outdir, out_name + '_pred_mean_pr.txt')
	out_file_upper_pr = os.path.join(predictions_outdir, out_name + '_pred_upper_pr.txt')
	out_file_lower_pr = os.path.join(predictions_outdir, out_name + '_pred_lower_pr.txt')

	# print the arrays to file
	np.save(out_file_post_pr, post_predictions)
	np.savetxt(out_file_mean_pr, np.mean(post_predictions, axis=0), fmt='%.3f')

	# just for plotting reasons, calculate the hpd interval of the first category predictions
	lower = [calcHPD(point, 0.95)[0] for point in post_predictions[:, :, 0].T]
	upper = [calcHPD(point, 0.95)[1] for point in post_predictions[:, :, 0].T]
	np.savetxt(out_file_upper_pr, upper, fmt='%.3f')
	np.savetxt(out_file_lower_pr, lower, fmt='%.3f')

	plotResults(out_file_mean_pr, out_file_lower_pr, out_file_upper_pr, nn_predictions_file, predictions_outdir, out_name)
import csv
import glob
import os
import numpy as np


# get data
def get_data(f,l=None,testsize=0.1, batch_training=0,seed=1234, all_class_in_testset=1,
             instance_id=0, header=0,feature_indx=None,randomize_order=True,from_file=True):
    np.random.seed(seed)
    inst_id = []
    if from_file:
        fname = os.path.splitext(os.path.basename(f))[0]
        try:
            tot_x = np.load(f)
        except(ValueError):
            if not instance_id:
                tot_x = np.loadtxt(f)
            else:
                tmp = np.genfromtxt(f, skip_header=header, dtype=str)
                tot_x = tmp[:,1:].astype(float)
                inst_id = tmp[:,0].astype(str)
            
        if header:
            feature_names = np.array(next(open(f)).split()[1:])
        else:
            feature_names = np.array(["feature_%s" % i for i in range(tot_x.shape[1])])
    else:
        f = pd.DataFrame(f)
        fname = 'bnn'
        if not instance_id:
            feature_names = np.array(f.columns)
            tot_x = f.values
        else:
            feature_names = np.array(f.columns[1:])
            tot_x = f.values[:,1:]
            inst_id = f.values[:,0].astype(str)

    if feature_indx is not None:
        feature_indx = np.array(feature_indx)
        tot_x = tot_x[:,feature_indx]
        feature_names = feature_names[feature_indx]

    
    try:
        if not l.empty:
            l = pd.DataFrame(l)
            tot_labels = l.values.astype(str) # if l already is a dataframe
    except:
        if not l:
            return {'data': np.array(tot_x).astype(float), 'labels': [], 'label_dict': [],
                    'test_data': [], 'test_labels': [],
                    'id_data': inst_id, 'id_test_data': [],
                    'file_name': fname, 'feature_names': feature_names}
        else:
            tot_labels = np.loadtxt(l,skiprows=header,dtype=str)
    
    if instance_id:
        tot_labels = tot_labels[:,1]
    tot_labels_numeric = turn_labels_to_numeric(tot_labels, l)
    x, labels, x_test, labels_test, inst_id_x, inst_id_x_test = randomize_data(tot_x, tot_labels_numeric,
                                                                               testsize=testsize,
                                                                               all_class_in_testset=all_class_in_testset,
                                                                               inst_id=inst_id,
                                                                               randomize=randomize_order)

    if batch_training:
        indx = np.random.randint(0,len(labels),batch_training)
        x = x[indx]
        labels = labels[indx]

    return {'data': np.array(x).astype(float), 'labels': labels, 'label_dict': np.unique(tot_labels),
            'test_data': np.array(x_test).astype(float), 'test_labels': labels_test,
            'id_data': inst_id_x, 'id_test_data': inst_id_x_test,
            'file_name': fname, 'feature_names': feature_names}


def save_data(dat, lab, outname="data", test_dat=[], test_lab=[]):
    np.savetxt(outname+"_features.txt", dat, delimiter="\t")
    np.savetxt(outname+"_labeles.txt", lab.astype(int), delimiter="\t")
    if len(test_dat) > 0:
        np.savetxt(outname + "_test_features.txt", test_dat, delimiter="\t")
        np.savetxt(outname + "_test_labeles.txt", test_lab.astype(int), delimiter="\t")

def init_output_files(bnn_obj, filename="BNN", sample_from_prior=0, outpath="",add_prms=None,
                      continue_logfile=False, log_all_weights=0):
    'prior_f = 0, p_scale = 1, hyper_p = 0, freq_indicator = 0'
    if bnn_obj._freq_indicator ==0:
        ind = ""
    else:
        ind = "_ind"
    if sample_from_prior:
        ind = ind + "_prior"
    ind =  ind + "_%s" % bnn_obj._seed
    outname = "%s_p%s_h%s_l%s_s%s_b%s%s" % (filename, bnn_obj._prior,bnn_obj._hyper_p, "_".join(map(str,
                                            bnn_obj._n_nodes)), bnn_obj._p_scale, bnn_obj._w_bound, ind)

    logfile_name = os.path.join(outpath, outname + ".log")
    if not log_all_weights:
        w_file_name = os.path.join(outpath, outname + ".pkl")
        wweights = None
    else:
        w_file_name = os.path.join(outpath, outname + "_W.log")
        w_file_name = open(w_file_name, "w")
        head_w = ["it"]

    head = ["it", "posterior", "likelihood", "prior", "accuracy", "test_accuracy"]
    for i in range(bnn_obj._size_output):
        head.append("acc_C%s" % i)
    for i in range(bnn_obj._n_layers):
        head.append("mean_w%s" % i)
        head.append("std_w%s" % i)
        if bnn_obj._hyper_p:
            if bnn_obj._hyper_p == 1:
                head.append("prior_std_w%s" % i)
            else:
                head.append("mean_prior_std_w%s" % i)
        if log_all_weights:
            head_w = head_w + ["w_%s_%s" % (i, j) for j in range(bnn_obj._w_layers[i].size)]
    if bnn_obj._freq_indicator:
        head.append("mean_ind")
    if add_prms:
        head = head + add_prms

    if bnn_obj._act_fun._trainable:
        head = head + ["alpha_%s" % (i) for i in range(bnn_obj._n_layers-1)]
    
    head.append("acc_prob")
    head.append("mcmc_id")
    
    if not continue_logfile:
        logfile = open(logfile_name, "w")
        wlog = csv.writer(logfile, delimiter='\t')
        wlog.writerow(head)
    else:
        logfile = open(logfile_name, "a")
        wlog = csv.writer(logfile, delimiter='\t')

    if log_all_weights:
        wweights = csv.writer(w_file_name, delimiter='\t')
        wweights.writerow(head_w)

    return wlog, logfile, w_file_name, wweights


def randomize_data(tot_x, tot_labels, testsize=0.1, all_class_in_testset=1, inst_id=[], randomize=True):
    if randomize:
        if testsize:
            rnd_order = np.random.choice(range(len(tot_labels)), len(tot_labels), replace=False)
        else:
            rnd_order = np.arange(len(tot_labels))
    else:
        rnd_order = np.arange(len(tot_labels))
        all_class_in_testset=0
    tot_x = tot_x[rnd_order]
    tot_labels = tot_labels[rnd_order]
    test_set_ind = int(testsize * len(tot_labels))
    inst_id_x = []
    inst_id_test = []
    tot_inst_id = []
    if len(inst_id):
        tot_inst_id = inst_id[rnd_order]

    if all_class_in_testset and testsize:
        test_set_ind = []

        for i in np.unique(tot_labels):
            ind = np.where(tot_labels == i)[0]
            test_set_ind = test_set_ind + list(np.random.choice(ind, np.max([1, int(testsize*len(ind))])))

        test_set_ind = np.array(test_set_ind)
        x_test = tot_x[test_set_ind]
        labels_test = tot_labels[test_set_ind]
        train_ind = np.array([z for z in range(tot_labels.size) if not z in test_set_ind])

        x = tot_x[train_ind]
        labels = tot_labels[train_ind]
        if len(inst_id):
            inst_id_x = tot_inst_id[train_ind]
            inst_id_test = tot_inst_id[test_set_ind]
    elif test_set_ind == 0:
        x_test = []
        labels_test = []
        x = tot_x
        labels = tot_labels
        if len(inst_id):
            inst_id_x = tot_inst_id
            inst_id_test = []
    else:
        x_test = tot_x[-test_set_ind:, :]
        labels_test = tot_labels[-test_set_ind:]
        x = tot_x[:-test_set_ind, :]
        labels = tot_labels[:-test_set_ind]
        if len(inst_id):
            inst_id_test = tot_inst_id[-test_set_ind:]
            inst_id_x = tot_inst_id[:-test_set_ind]
         
    return x, labels, x_test, labels_test, inst_id_x, inst_id_test


def load_obj(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except:
        import pickle5
        with open(file_name, 'rb') as f:
            return pickle5.load(f)

def merge_dict(d1, d2):
    from collections import defaultdict
    d = defaultdict(list)
    for a, b in d1.items() + d2.items():
        d[a].append(b)
    return d


def combine_pkls(files=None, dir=None, tag=""):
    if dir is not None:
        files = glob.glob(os.path.join(dir, "*%s*.pkl" % tag))
        print("Combining files: ", files)
    comb_pkl = list()
    for f in files:
        w = load_obj(f)
        comb_pkl = comb_pkl + w
    return comb_pkl

def turn_labels_to_numeric(labels,label_file,save_to_file=False):
    numerical_labels = np.zeros(len(labels)).astype(int)
    c = 0
    for i in np.unique(labels):
        numerical_labels[labels == i] = c
        c += 1

    if save_to_file:
        outfile = label_file.replace('.txt','_numerical.txt')
        np.savetxt(outfile,numerical_labels,fmt='%i')
    return numerical_labels






import numpy as np
import scipy.special
import scipy.stats

np.set_printoptions(suppress=True, precision=3)
small_number = 1e-10
import random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


def UpdateFixedNormal(i, d=1, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = rs.randint(0, i.shape[1],n)
    current_prm = i[Ix,Iy]
    new_prm = rs.normal(0, d[Ix,Iy], n)
    hastings = np.sum(scipy.stats.norm.logpdf(current_prm, 0, d[Ix,Iy]) - \
               scipy.stats.norm.logpdf(new_prm, 0, d[Ix,Iy]))
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = new_prm
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    return z, (Ix, Iy), hastings

def UpdateNormal1D(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, len(i),n) # faster than np.random.choice
    z = np.zeros(i.shape) + i
    z[Ix] = z[Ix] + rs.normal(0, d, n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    hastings = 0
    return z, Ix, hastings

def UpdateNormal(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = rs.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + rs.normal(0, d[Ix,Iy], n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    hastings = 0
    return z, (Ix, Iy), hastings

def UpdateNormalNormalized(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = rs.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + rs.normal(0, d[Ix,Iy], n)
    z = z/np.sum(z)
    hastings = 0
    return z, (Ix, Iy), hastings



def UpdateUniform(i, d=0.1, n=1, Mb=100, mb= -100):
    Ix = np.random.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = np.random.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + np.random.uniform(-d[Ix,Iy], d[Ix,Iy], n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    hastings = 0
    return z, (Ix, Iy), hastings


def UpdateBinomial(ind,update_f,shape_out):
    return np.abs(ind - np.random.binomial(1, np.random.random() * update_f, shape_out))


def GibbsSampleNormStdGammaVector(x,a=2,b=0.1,mu=0):
    Gamma_a = a + len(x)/2.
    Gamma_b = b + np.sum((x-mu)**2)/2.
    tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
    return 1/np.sqrt(tau)


def GibbsSampleNormStdGamma2D(x,a=1,b=0.1,mu=0):
    Gamma_a = a + (x.shape[0])/2. #
    Gamma_b = b + np.sum((x-mu)**2,axis=0)/2.
    tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
    return 1/np.sqrt(tau)

def GibbsSampleNormStdGammaONE(x,a=1.5,b=0.1,mu=0):
    Gamma_a = a + 1/2. # one observation for each value (1 Y for 1 s2)
    Gamma_b = b + ((x-mu)**2)/2.
    tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
    return 1/np.sqrt(tau)

def GibbsSampleGammaRateExp(sd,a,alpha_0=1.,beta_0=1.):
    # prior is on precision tau
    tau = 1./(sd**2) #np.array(tau_list)
    conjugate_a = alpha_0 + len(tau)*a
    conjugate_b = beta_0 + np.sum(tau)
    return np.random.gamma(conjugate_a,scale=1./conjugate_b)


def run_mcmc(bnn, mcmc, logger):
    while True:
        mcmc.mh_step(bnn)
        # print some stats (iteration number, likelihood, training accuracy, test accuracy
        if mcmc._current_iteration % mcmc._print_f == 0 or mcmc._current_iteration == 1:
            print(mcmc._current_iteration, np.round([mcmc._logLik, mcmc._accuracy, mcmc._test_accuracy],3))
        # save to file
        if mcmc._current_iteration % mcmc._sampling_f == 0:
            logger.log_sample(bnn,mcmc)
            logger.log_weights(bnn,mcmc)
        # stop MCMC after running desired number of iterations
        if mcmc._current_iteration == mcmc._n_iterations:
            break
