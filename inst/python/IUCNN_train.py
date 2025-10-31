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

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import keras

from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, Optimizer, RMSprop, SGD

# disable progress bars globally (instead of model.predict(..., verbose=0), which does not supress progress output in R)
tf.keras.utils.disable_interactive_logging()

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf compilation warning
except:
    pass

# declare location of the python files make functions of other python files importable
sys.path.append(os.path.dirname(__file__))
from IUCNN_predict import rescale_labels, turn_reg_output_into_softmax

def iucnn_train(dataset,
                labels,
                mode,
                path_to_output,
                test_fraction,
                cv_k,
                seed,
                instance_names,
                feature_names,
                verbose,
                max_epochs,
                patience,
                batch_size,
                n_layers,
                use_bias,
                balance_classes,
                act_f,
                act_f_out,
                stretch_factor_rescaled_labels,
                randomize_instances,
                rescale_features,
                dropout_rate,
                dropout_reps,
                mc_dropout,
                label_noise_factor,
                no_validation,
                save_model,
                optimizer,
                optimizer_kwargs,
                l2_regularizer,
                test_label_balance_factor = 1.0):
    
    keras.utils.set_random_seed(seed)
    
    @keras.saving.register_keras_serializable()
    class MCDropout(tf.keras.layers.Dropout):
        def call(self, inputs):
            return super().call(inputs, training=True)

    def get_optimizer(optimizer, optimizer_kwargs):
        opt = optimizer
        if optimizer == 'adadelta' and not optimizer_kwargs is None:
            opt = Adadelta(**optimizer_kwargs)
        elif optimizer == 'adafactor' and not optimizer_kwargs is None:
            opt = Adafactor(**optimizer_kwargs)
        elif optimizer == 'adagrad' and not optimizer_kwargs is None:
            opt = Adagrad(**optimizer_kwargs)
        elif optimizer == 'adam' and not optimizer_kwargs is None:
            opt = Adam(**optimizer_kwargs)
        elif optimizer == 'adamw' and not optimizer_kwargs is None:
            opt = AdamW(**optimizer_kwargs)
        elif optimizer == 'adamax' and not optimizer_kwargs is None:
            opt = Adamax(**optimizer_kwargs)
        elif optimizer == 'ftrl' and not optimizer_kwargs is None:
            opt = Ftrl(**optimizer_kwargs)
        elif optimizer == 'nadam' and not optimizer_kwargs is None:
            opt = Nadam(**optimizer_kwargs)
        elif optimizer == 'optimizer' and not optimizer_kwargs is None:
            opt = Optimizer(**optimizer_kwargs)
        elif optimizer == 'rmsprop' and not optimizer_kwargs is None:
            opt = RMSprop(**optimizer_kwargs)
        elif optimizer == 'sgd' and not optimizer_kwargs is None:
            opt = SGD(**optimizer_kwargs)
        return opt

    def get_l2_regularizer(l2_regularizer):
        l2_reg = [None, None, None]
        if not l2_regularizer is None:
            if 'kernel_regularizer' in l2_regularizer.keys():
                l2_reg[0] = tf.keras.regularizers.L2(l2_regularizer['kernel_regularizer'])
            if 'bias_regularizer' in l2_regularizer.keys():
                l2_reg[1] = tf.keras.regularizers.L2(l2_regularizer['bias_regularizer'])
            if 'activity_regularizer' in l2_regularizer.keys():
                l2_reg[2] = tf.keras.regularizers.L2(l2_regularizer['activity_regularizer'])
        return l2_reg

    def build_classification_model(dropout, dropout_rate, use_bias, l2_regularizer):
        l2_regularizer_settings = get_l2_regularizer(l2_regularizer)
        architecture = [tf.keras.layers.Flatten(input_shape=[train_set.shape[1]])]
        architecture.append(tf.keras.layers.Dense(n_layers[0],
                                                  activation=act_f,
                                                  use_bias=use_bias,
                                                  kernel_regularizer=l2_regularizer_settings[0],
                                                  bias_regularizer=l2_regularizer_settings[1],
                                                  activity_regularizer=l2_regularizer_settings[2]))
        for i in n_layers[1:]:
            architecture.append(tf.keras.layers.Dense(i, activation=act_f))
        if dropout:
            dropout_layers = [MCDropout(dropout_rate) for i in architecture[1:]]
            architecture = [architecture[0]] + [j for i in zip(architecture[1:],dropout_layers) for j in i]

        architecture.append(tf.keras.layers.Dense(n_class,
                                                  activation=act_f_out))
        model = tf.keras.Sequential(architecture)
        opt = get_optimizer(optimizer, optimizer_kwargs)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return model

    def build_regression_model(dropout, dropout_rate, use_bias, l2_regularizer):
        l2_regularizer_settings = get_l2_regularizer(l2_regularizer)
        architecture = [tf.keras.layers.Flatten(input_shape=[train_set.shape[1]])]
        architecture.append(tf.keras.layers.Dense(n_layers[0],
                                                  activation=act_f,
                                                  use_bias=use_bias,
                                                  kernel_regularizer=l2_regularizer_settings[0],
                                                  bias_regularizer=l2_regularizer_settings[1],
                                                  activity_regularizer=l2_regularizer_settings[2]))
        for i in n_layers[1:]:
            architecture.append(tf.keras.layers.Dense(i, activation=act_f))

        if dropout:
            dropout_layers = [MCDropout(dropout_rate) for i in architecture[1:]]
            architecture = [architecture[0]] + [j for i in zip(architecture[1:], dropout_layers) for j in i]

        if act_f_out:
            # sigmoid or tanh
            architecture.append(tf.keras.layers.Dense(1,
                                                      activation=act_f_out))
        else:
            architecture.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(architecture)
#        optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
        
        opt = get_optimizer(optimizer, optimizer_kwargs)
        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['mae', 'mse'])
        return model

    def determine_optim_rounding_boundary(regressed_labels,true_labels):
        accs = []
        values= np.linspace(-1,1,100)
        for x in values:
            label_predictions = np.round(regressed_labels + x, 0).astype(int).flatten()
            cat_acc = np.sum(label_predictions==true_labels)/len(label_predictions)
            accs.append(cat_acc)
        #plt.plot(values,accs)
        add_value = values[np.where(accs==np.max(accs))[0][0]]
        return(add_value)

    def get_classification_accuracy(model,features,true_labels,mc_dropout,dropout_reps,loss=False):
        if features.shape[0] == 0:
            return np.array([]), np.array([]), np.nan, np.nan
        else:
            if mc_dropout:
                predictions_raw = np.array([model.predict(features) for i in np.arange(dropout_reps)])
                predictions_raw_mean = np.mean(predictions_raw,axis=0)
            else:
                predictions_raw_mean = model.predict(features)
            label_predictions = np.argmax(predictions_raw_mean, axis=1)
            true_label_cats = np.argmax(true_labels,axis=1)
            mean_acc_man = np.sum(label_predictions==true_label_cats)/len(label_predictions)
            if loss:
                if mc_dropout:
                    preds = np.array([model.evaluate(features,true_labels,verbose=0) for i in np.arange(dropout_reps)])
                    test_loss, test_acc = np.hsplit(preds,2)
                    mean_loss = np.mean(test_loss)
                    #mean_acc = np.mean(test_acc)
                else:
                    mean_loss,mean_acc = model.evaluate(features,true_labels,verbose=0) 
            else:
                mean_loss = np.nan
                #mean_acc = np.nan
            return label_predictions, predictions_raw_mean, mean_loss, mean_acc_man


    def get_regression_accuracy(model,features,labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,mc_dropout,dropout_reps=1):
        if features.shape[0] == 0:
            return np.nan, np.array([]), np.array([])
        else:   
            if mc_dropout:
                label_cats = np.arange(rescale_factor+1)
                prm_est_reps_unscaled = np.array([model.predict(features).flatten() for i in np.arange(dropout_reps)])
                predictions_raw = np.array([rescale_labels(i,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) for i in prm_est_reps_unscaled])
                prm_est_mean = turn_reg_output_into_softmax(predictions_raw,label_cats)
                label_predictions = np.argmax(prm_est_mean, axis=1)
            else:
                prm_est_mean_unscaled = model.predict(features).flatten()
                prm_est_mean = rescale_labels(prm_est_mean_unscaled,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True)
                label_predictions = np.round(prm_est_mean, 0).astype(int).flatten()
            real_labels = rescale_labels(labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True).astype(int).flatten()

            cat_acc = np.sum(label_predictions==real_labels)/len(label_predictions)
            return cat_acc, label_predictions, prm_est_mean

    def model_init(mode,dropout, dropout_rate, use_bias, l2_regularizer):
        if mode == 'nn-reg':
            model = build_regression_model(dropout, dropout_rate, use_bias, l2_regularizer)
        elif mode == 'nn-class':
            model = build_classification_model(dropout, dropout_rate, use_bias, l2_regularizer)
        return model 

    def iter_test_indices(features, n_splits = 5, shuffle = True, seed = None):
        n_samples = features.shape[0]
        indices = np.arange(n_samples)
        if seed:
            np.random.seed(seed)
        if shuffle:
            indices = np.random.choice(indices, len(indices), replace=False)
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int_)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        index_output = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            index_output.append(indices[start:stop])
            current = stop
        return index_output

    def get_confidence_threshold(predicted_labels,true_labels,target_acc=0.9):
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

    def supersample_classes(train_ids,labels):
        train_ids = np.array(train_ids)
        labels = labels.flatten()
        train_labels = labels[train_ids]
        max_class_count = max(np.unique(train_labels,return_counts=True)[1])
        final_train_ids = []
        for i in np.unique(train_labels):
            train_ids_class = train_ids[train_labels == i]
            n_in_class = len(train_labels[train_labels == i])
            delta_n_inst = max_class_count-n_in_class
            drawn_inst_ind = np.random.choice(train_ids_class,delta_n_inst,replace=True)
            train_ids_final = np.concatenate([train_ids_class,drawn_inst_ind])
            final_train_ids.append(train_ids_final)
        ids_class_balance = np.concatenate(final_train_ids)
        np.random.shuffle(ids_class_balance)
        return(ids_class_balance)

    def manipulate_instance_distribution(features, labels, factor=1., multi_col_labels = False):
        if multi_col_labels:
            tmp_labels = np.argmax(labels, axis=1)
        else:
            tmp_labels = labels
        classes, counts = np.unique(tmp_labels, return_counts=True)
        min_count = min(counts)
        min_class = int(classes[counts == min_count])
        final_ids = []
        for i in classes:
            label_ids = np.where(tmp_labels == i)[0]
            delta_n = len(label_ids) - min_count
            sample_n = int(np.round(delta_n * factor))
            final_ids += sorted(np.random.choice(label_ids, min_count + sample_n, replace=False))
        final_ids = sorted(final_ids)
        out_features = features[final_ids, :]
        if multi_col_labels:
            out_labels = labels[final_ids,]
        else:
            out_labels = labels[final_ids]
        return (out_features, out_labels)

    # randomize data
    if seed > 0:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    if rescale_features:
        print('rescale_features flag is being ignored, this function is currently not supported',flush=True)
        #dataset = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))
    if dropout_rate>0:
        dropout = True
    else:
        dropout = False
        mc_dropout = False
    if use_bias == 1:
        use_bias = True
    elif use_bias == 0:
        use_bias = False
    if mode == 'nn-reg':
        rescale_labels_boolean = True
        if act_f_out == 'tanh':
            min_max_label = [-1,1]
        elif act_f_out == 'sigmoid':
            min_max_label = [0,1]
        else:
            min_max_label = [min(labels),max(labels)]
            act_f_out = None
            #quit('Invalid activation function choice for output layer. Currently IUCNN only supports "tanh" or "sigmoid" as output layer activation functions for the regression model (set with act_f_out flag).')
    elif mode == 'nn-class':
        rescale_labels_boolean = False
        min_max_label = [min(labels),max(labels)]
    else:
        quit('Invalid mode provided. Choose from "nn-class", "nn-reg", or "bnn-class"')
    if balance_classes:
        print('Super-sampling minority classes to reach class balance for training.',flush=True)
    
    
    rescale_factor = max(labels)
    rescaled_labels = rescale_labels(labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
    
    if cv_k > 1:
        train_index_blocks = iter_test_indices(dataset,n_splits = cv_k,shuffle=randomize_instances)
        cv = True
        test_fraction = 0.0
    else:
        if randomize_instances:
            rnd_indx = np.random.choice(range(len(labels)), len(labels), replace=False)
        else:
            rnd_indx = np.arange(len(labels))
        
        if test_fraction == 0:
            train_index_blocks = [rnd_indx]
            test_indices = [[]]
        else:
            test_size = int(len(labels)*test_fraction)
            train_index_blocks = [rnd_indx[:-test_size]]
            test_indices = [rnd_indx[-test_size:]]
                
        cv = False
        
    train_acc_per_fold = []
    train_loss_per_fold = []
    validation_acc_per_fold = []
    validation_loss_per_fold = []
    test_acc_per_fold = []
    test_loss_per_fold = []

    all_test_labels = []
    all_test_instance_names = []
    all_test_predictions = []
    all_test_predictions_raw = []
    
    stopping_points = []
    training_histories = {}
    validation_histories = {}
    train_acc_histories = {}
    val_acc_histories = {}
    train_mae_histories = {}
    val_mae_histories = {}

    all_model_outpaths = []
    
    orig_dataset = dataset
    orig_labels = labels

    test_count_cv_folds = []

    for it, __ in enumerate(train_index_blocks):
        if cv:
            test_ids = train_index_blocks[it] # in case of cv, choose one of the k chunks as test set
            train_ids = np.array([])
            for i in list(np.delete(np.arange(len(train_index_blocks)), it)):
                train_ids = np.concatenate((train_ids, train_index_blocks[i]), axis = None)
            train_ids = train_ids.astype(int)
            #train_ids = np.concatenate(np.array([train_index_blocks[i] for i in list(np.delete(np.arange(len(train_index_blocks)),it))])).astype(int)
            print("Training CV fold %i/%i on %i training instances (%i test instances)..." % (it+1, cv_k, len(train_ids), len(test_ids)), flush=True)
        else:
            test_ids = list(test_indices[it])
            train_ids = list(train_index_blocks[it])
            print("Training model on %i training instances (%i test instances)..."%(len(train_ids),len(test_ids)),flush=True)
        
        # these are just to keep track of the true, unaltered arrays for output
        orig_train_set = dataset[train_ids,:]
        orig_train_labels = labels[train_ids]
        orig_test_set = dataset[test_ids,:]
        orig_test_labels = labels[test_ids]
        orig_test_instance_names = instance_names[test_ids]
        all_test_labels.append(orig_test_labels)
        all_test_instance_names.append(orig_test_instance_names)

        # supersample train_ids if balance_class mode is active
        if balance_classes:
            train_ids = supersample_classes(train_ids,labels)

        # define train and test set
        train_set = dataset[train_ids,:]
        test_set = orig_test_set

        # fix labels depending on whether it's regression or classification model
        labels_cat = tf.keras.utils.to_categorical(labels)
        n_class = labels_cat.shape[1]
        if mode == 'nn-class':
            labels_for_training = labels_cat[train_ids,:]
            labels_for_testing = labels_cat[test_ids,:]
            optimize_for_this = "val_loss"
        elif mode == 'nn-reg':
            labels_for_training = rescaled_labels[train_ids]
            noise_radius = (((np.max(rescaled_labels)-np.min(rescaled_labels))/(n_class-1))/2)*label_noise_factor
            #labels_for_training = np.array([np.random.normal(i,noise_radius/3) for i in labels_for_training])
            labels_for_training = np.array([np.random.uniform(i-noise_radius,i+noise_radius) for i in labels_for_training])
            labels_for_testing = rescaled_labels[test_ids]
            optimize_for_this = "val_mae"

        # run for set number of iterations, no early stopping
        if no_validation:
            print('Running training for set number of epochs: %i'%max_epochs,flush=True)
            tf.random.set_seed(seed)
            # determining optimal number of epochs
            model = model_init(mode,dropout, dropout_rate, use_bias, l2_regularizer)
            #model.build((train_set.shape[1],))
            #model.summary()
            
            history = model.fit(train_set,
                                labels_for_training,
                                epochs=max_epochs,
                                batch_size=batch_size,
                                verbose=verbose)
            stopping_point = max_epochs-1
        else:
            tf.random.set_seed(seed)
            # train model
            model = model_init(mode,dropout, dropout_rate, use_bias, l2_regularizer)
            #model.build((train_set.shape[1],))
            if verbose:
                model.summary()
            # The patience parameter is the amount of epochs to check for improvement
            early_stop = tf.keras.callbacks.EarlyStopping(monitor=optimize_for_this, patience=patience, restore_best_weights=True)
            if cv: # when using CV use test set to determine stopping point
                history = model.fit(train_set,
                                    labels_for_training,
                                    epochs=max_epochs,
                                    batch_size=batch_size,
                                    validation_data=(test_set,labels_for_testing),
                                    verbose=verbose,
                                    callbacks=[early_stop])
            else:
                history = model.fit(train_set,
                                    labels_for_training,
                                    epochs=max_epochs,
                                    batch_size=batch_size,
                                    validation_split=0.2,
                                    verbose=verbose,
                                    callbacks=[early_stop])
            if 'accuracy' in optimize_for_this:
                stopping_point = np.argmax(history.history[optimize_for_this])
            else:
                stopping_point = np.argmin(history.history[optimize_for_this])
            print('Best training epoch: ',stopping_point+1,flush=True)
    
        if mode == 'nn-class':
            train_predictions, train_predictions_raw, train_loss, train_acc = get_classification_accuracy(model,train_set,labels_for_training,mc_dropout,dropout_reps,loss=True)
            if no_validation:  # if no validation set exists
                val_acc = np.nan
                val_loss = np.nan
                val_acc_history = np.nan
                val_loss_history = np.nan
            else:
                val_acc = history.history['val_accuracy'][stopping_point]
                val_loss = history.history['val_loss'][stopping_point]
                val_acc_history = np.array(history.history['val_accuracy'])
                val_loss_history = np.array(history.history['val_loss'])
            if len(labels_for_testing)>0:
                if test_label_balance_factor < 1:
                    test_set, labels_for_testing = manipulate_instance_distribution(test_set, labels_for_testing, factor=test_label_balance_factor, multi_col_labels=True)
                    test_count_cv_fold = np.unique(np.argmax(labels_for_testing, axis=1), return_counts=True)[1]
                test_predictions, test_predictions_raw, test_loss, test_acc = get_classification_accuracy(model,test_set,labels_for_testing,mc_dropout,dropout_reps,loss=True)
            else:
                test_loss = np.nan
                test_acc = np.nan
                test_predictions = np.nan
                test_predictions_raw = np.nan
            train_acc_history = np.array(history.history['accuracy'])
            train_mae_history = np.nan
            val_mae_history = np.nan
    
        elif mode == 'nn-reg':
            train_loss = history.history['loss'][stopping_point]
            train_acc, train_predictions, train_predictions_raw = get_regression_accuracy(model,train_set,labels_for_training,rescale_factor,min_max_label,stretch_factor_rescaled_labels,mc_dropout,dropout_reps)
            if no_validation:
                val_loss = np.nan
                val_loss_history = np.nan
                val_mae_history = np.nan
            else:
                val_loss = history.history['val_loss'][stopping_point]
                val_loss_history = np.array(history.history['val_loss'])
                val_mae_history = np.array(history.history['val_mae'])
            val_acc = np.nan
            if len(labels_for_testing)>0:
                if test_label_balance_factor < 1:
                    test_set, labels_for_testing = manipulate_instance_distribution(test_set, labels_for_testing, factor=test_label_balance_factor, multi_col_labels=False)
                    test_count_cv_fold = np.unique(np.argmax(labels_for_testing, axis=1), return_counts=True)[1]
                test_acc, test_predictions, test_predictions_raw = get_regression_accuracy(model,test_set,labels_for_testing,rescale_factor,min_max_label,stretch_factor_rescaled_labels,mc_dropout,dropout_reps)
                if cv:
                    val_acc = test_acc
                #test_loss = np.nan#np.mean(tf.keras.losses.sparse_categorical_crossentropy(orig_test_labels, test_predictions_raw))
                test_loss = np.mean([model.evaluate(test_set, labels_for_testing, verbose=0,return_dict=True)['loss'] for i in np.arange(dropout_reps)])

            else:
                test_acc = np.nan
                test_predictions = np.nan
                test_predictions_raw = np.nan
                test_loss = np.nan
            train_acc_history = np.nan
            val_acc_history = np.nan
            train_mae_history = np.array(history.history['mae'])

        train_acc_per_fold.append(train_acc)
        train_loss_per_fold.append(train_loss)
        validation_acc_per_fold.append(val_acc)
        validation_loss_per_fold.append(val_loss)
        test_acc_per_fold.append(test_acc)
        test_loss_per_fold.append(test_loss)
        
        all_test_predictions.append(test_predictions)
        all_test_predictions_raw.append(test_predictions_raw)
        
        stopping_points.append(stopping_point+1) # add +1 because R does different indexing than python

        training_histories.setdefault('train_rep_%i'%it,np.array(history.history['loss']))
        validation_histories.setdefault('train_rep_%i'%it,val_loss_history)
        train_acc_histories.setdefault('train_rep_%i'%it,train_acc_history)
        val_acc_histories.setdefault('train_rep_%i'%it,val_acc_history)
        train_mae_histories.setdefault('train_rep_%i'%it,train_mae_history)
        val_mae_histories.setdefault('train_rep_%i'%it,val_mae_history)

        if test_label_balance_factor < 1:
            test_count_cv_folds.append(test_count_cv_fold)

        if save_model:
            if not os.path.exists(path_to_output):
                os.makedirs(path_to_output)
            model_outpath = os.path.join(path_to_output, 'nn_model_%s.keras' % it)
            model.save( model_outpath )
            print("\nIUC-NN model saved at: ", model_outpath, flush=True)
        else:
            model_outpath = ''
        all_model_outpaths.append(model_outpath)

    # export model outpath as string instead of list for a non-CV model
    if len(all_model_outpaths) == 1:
        all_model_outpaths = all_model_outpaths[0]
    
    # print stats to screen
    avg_train_acc = np.mean(train_acc_per_fold)
    avg_validation_acc = np.mean(validation_acc_per_fold)
    avg_test_acc = np.mean(test_acc_per_fold)
    avg_train_loss = np.mean(train_loss_per_fold)
    avg_validation_loss = np.mean(validation_loss_per_fold)
    avg_test_loss = np.mean(test_loss_per_fold)

    if verbose:
        print('Average scores for all folds:')
        print('> Test accuracy: %.5f (+- %.5f (std))'%(avg_test_acc,np.std(test_acc_per_fold)))
        print('> Test loss: %.5f'%avg_test_loss)

    if len(labels_for_testing)>0:
        all_test_labels = np.concatenate(all_test_labels).flatten()
        all_test_instance_names = np.concatenate(all_test_instance_names).flatten()
        all_test_predictions = np.concatenate(all_test_predictions)
        all_test_predictions_raw = np.concatenate(all_test_predictions_raw)
    else:
        all_test_labels = np.nan
        all_test_instance_names = np.nan
        all_test_predictions: np.nan
        all_test_predictions_raw = np.nan
      
    if len(labels_for_testing) == 0:
        confusion_matrix = np.zeros([n_class,n_class])
        accthres_tbl = np.nan
    elif test_label_balance_factor < 1:
        confusion_matrix = np.zeros([n_class, n_class])
        accthres_tbl = np.nan
    else:
        if mc_dropout:
            accthres_tbl = get_confidence_threshold(all_test_predictions_raw,all_test_labels,target_acc=None)
        else:
            accthres_tbl = np.nan
        confusion_matrix = np.array(tf.math.confusion_matrix(all_test_labels,all_test_predictions))

    no_test = False
    if test_fraction == 0 and cv_k < 2:
        data_train = orig_dataset
        labels_train = orig_labels
        data_test = np.nan
        labels_test = np.nan
        no_test = True
        train_instance_names = instance_names
        test_instance_names = instance_names
    elif test_fraction == 0 and cv_k > 1:
        avg_test_loss = avg_validation_loss
        avg_test_acc = avg_validation_acc
        data_train = orig_dataset
        labels_train = orig_labels
        data_test = orig_dataset
        labels_test = orig_labels
        train_instance_names = instance_names
        test_instance_names = instance_names
    else:
        data_train = orig_train_set
        labels_train = orig_train_labels.flatten()
        data_test = test_set
        labels_test = all_test_labels.flatten()
        instance_names_test = all_test_instance_names.flatten()
        train_instance_names = instance_names[train_ids]
        test_instance_names = instance_names[test_ids]

    # sample from categorical
    nreps = 1000
    if mc_dropout:
        if no_test:
            predicted_class_count = np.nan
        else:
            mc_dropout_probs = all_test_predictions_raw
            label_dict = np.arange(mc_dropout_probs.shape[1])
            samples = np.array([np.random.choice(label_dict, nreps, p=i) for i in mc_dropout_probs])
            predicted_class_count = np.array([[list(col).count(i) for i in label_dict] for col in samples.T])
    else:
        predicted_class_count = np.nan

    if no_test:
        true_class_count = np.nan
    else:
        true_class_count = [list(all_test_labels).count(i) for i in np.unique(orig_labels)]
    if test_label_balance_factor < 1:
        true_class_count = np.sum(np.array(test_count_cv_folds), axis=0)



    output = {
                'test_labels': all_test_labels,
                'test_predictions': all_test_predictions,
                'test_instance_names': all_test_instance_names,
                'test_predictions_raw': all_test_predictions_raw,
                
                'training_accuracy': avg_train_acc,
                'validation_accuracy': avg_validation_acc,
                'test_accuracy': avg_test_acc,

                'training_loss': avg_train_loss,
                'validation_loss': avg_validation_loss,
                'test_loss': avg_test_loss,
                
                'training_loss_history': training_histories,
                'validation_loss_history': validation_histories,

                'training_accuracy_history': train_acc_histories,
                'validation_accuracy_history': val_acc_histories,
                
                'training_mae_history': train_mae_histories,
                'validation_mae_history': val_mae_histories,
                
                'rescale_labels_boolean': rescale_labels_boolean,
                'label_rescaling_factor': rescale_factor,
                'min_max_label': np.array(min_max_label),
                'label_stretch_factor': stretch_factor_rescaled_labels,
                
                'activation_function': act_f_out,
                'trained_model_path': all_model_outpaths,
                
                'confusion_matrix': confusion_matrix,
                'mc_dropout': mc_dropout,
                'accthres_tbl': accthres_tbl,
                'true_class_count': true_class_count,
                'predicted_class_count': predicted_class_count,
                'stopping_point': np.array(stopping_points),
                
                'input_data':   {"data": data_train,
                                "labels": labels_train,
                                "label_dict": np.unique(orig_labels).astype(str),
                                "test_data": data_test,
                                "test_labels": labels_test,
                                "id_data": train_instance_names,
                                "id_test_data": test_instance_names,
                                "file_name": os.path.basename(path_to_output),
                                "feature_names": feature_names
                                }
    }
    
    return output
