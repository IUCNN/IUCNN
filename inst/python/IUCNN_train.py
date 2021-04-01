import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf compilation warning
except:
    pass

def iucnn_train(dataset, 
                labels,
                mode,
                path_to_output,
                validation_split,
                test_fraction,
                cv_k,
                seed,
                instance_names,
                feature_names,
                verbose,
                model_name,
                max_epochs,
                n_layers,
                use_bias,
                act_f,
                act_f_out,
                stretch_factor_rescaled_labels,
                patience,
                randomize_instances,
                rescale_features,
                dropout_rate,
                dropout_reps,
                mc_dropout,
                label_noise_factor,
                save_model):    
    
    
    class MCDropout(tf.keras.layers.Dropout):
        def call(self, inputs):
            return super().call(inputs, training=True)    
    
    def build_classification_model(dropout,dropout_rate,use_bias):
        architecture = [tf.keras.layers.Flatten(input_shape=[train_set.shape[1]])]
        architecture.append(tf.keras.layers.Dense(n_layers[0],
                                      activation=act_f,
                                      use_bias=use_bias))
        for i in n_layers[1:]:
            architecture.append(tf.keras.layers.Dense(i, activation=act_f))
        if dropout:
            dropout_layers = [MCDropout(dropout_rate) for i in architecture[1:]]
            architecture = [architecture[0]] + [j for i in zip(architecture[1:],dropout_layers) for j in i]

        architecture.append(tf.keras.layers.Dense(n_class, 
                                         activation=act_f_out))
        model = tf.keras.Sequential(architecture)
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model

    def build_regression_model(dropout,dropout_rate,use_bias):
        architecture = [tf.keras.layers.Flatten(input_shape=[train_set.shape[1]])]
        architecture.append(tf.keras.layers.Dense(n_layers[0],
                                      activation=act_f,
                                      use_bias=use_bias))
        for i in n_layers[1:]:
            architecture.append(tf.keras.layers.Dense(i, activation=act_f))

        if dropout:
            dropout_layers = [MCDropout(dropout_rate) for i in architecture[1:]]
            architecture = [architecture[0]] + [j for i in zip(architecture[1:],dropout_layers) for j in i]

        if act_f_out:
            architecture.append(tf.keras.layers.Dense(1, activation=act_f_out))    #sigmoid or tanh
        else:
            architecture.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(architecture)
        optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mae','mse'])
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
                mean_loss == np.nan
                #mean_acc = np.nan
            return label_predictions, predictions_raw_mean, mean_loss, mean_acc_man

    def get_regression_accuracy(model,features,labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,mc_dropout,dropout_reps=1,return_std = False):
        if features.shape[0] == 0:
            return np.nan, np.array([]), np.array([])
        else:   
            if mc_dropout:
                prm_est_reps = np.array([model.predict(features).flatten() for i in np.arange(dropout_reps)])
                prm_est_mean = np.mean(prm_est_reps,axis=0)
            else:
                prm_est_mean = model.predict(features).flatten()
            prm_est_rescaled = rescale_labels(prm_est_mean,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True)
            real_labels = rescale_labels(labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True).astype(int).flatten()
            label_predictions = np.round(prm_est_rescaled, 0).astype(int).flatten()
            cat_acc = np.sum(label_predictions==real_labels)/len(label_predictions)
            if return_std:
                prm_est_reps_rescaled = np.array([rescale_labels(i,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True) for i in prm_est_reps])
                stds_rescaled = np.std(prm_est_reps_rescaled,axis=0)
                min_rescaled = np.min(prm_est_reps_rescaled,axis=0)
                max_rescaled = np.max(prm_est_reps_rescaled,axis=0)
                return cat_acc, label_predictions, prm_est_rescaled, np.array([min_rescaled,max_rescaled])
            else:
                return cat_acc, label_predictions, prm_est_rescaled

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
    
    # def rescale_labels(labels,n_labels,lab_range):
    #     if n_labels == 0:
    #         rescaled_labels = labels
    #     else:
    #         rescaled_labels = ((labels/lab_range) +0.5) * (n_labels-1)
    #     return(rescaled_labels)

    def model_init(mode,dropout,dropout_rate,use_bias):
        if mode == 'nn-reg':
            model = build_regression_model(dropout,dropout_rate,use_bias)
        elif mode == 'nn-class':    
            model = build_classification_model(dropout,dropout_rate,use_bias)
        return model 

    def iter_test_indices(features, n_splits = 5, shuffle = True, seed = None):
        n_samples = features.shape[0]
        indices = np.arange(n_samples)
        if seed:
            np.random.seed(seed)
        if shuffle:
            indices = np.random.choice(indices, len(indices), replace=False)
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        index_output = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            index_output.append(indices[start:stop])
            current = stop
        return index_output

    # randomize data
    if seed > 0:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    if rescale_features:
        dataset = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))
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
            act_f_out == None
            #quit('Invalid activation function choice for output layer. Currently IUCNN only supports "tanh" or "sigmoid" as output layer activation functions for the regression model (set with act_f_out flag).')
    elif mode == 'nn-class':
        rescale_labels_boolean = False
        min_max_label = [min(labels),max(labels)]
    else:
        quit('Invalid mode provided. Choose from "nn-class", "nn-reg", or "bnn-class"')
    
    rescale_factor = max(labels)
    rescaled_labels = rescale_labels(labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
    
    if cv_k > 1:
        train_index_blocks = iter_test_indices(dataset,n_splits = cv_k,shuffle=randomize_instances)
        test_indices = []
        cv = True
    else:
        if randomize_instances:
            rnd_indx = np.random.choice(range(len(labels)), len(labels), replace=False)
        else:
            rnd_indx = np.arange(len(labels))
        test_size = int(len(labels)*test_fraction)
        if test_size == 0:
            train_index_blocks = [rnd_indx]
            test_indices = [[]]
        else:
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
    all_test_predictions = []
    all_test_predictions_raw = []
    
    for it, __ in enumerate(train_index_blocks):
        if cv:
            print("Training CV fold %i/%i..."%(it+1,cv_k),flush=True)
            test_ids = train_index_blocks[it] # in case of cv, choose one of the k chunks as test set
            train_ids = np.concatenate(np.array([train_index_blocks[i] for i in list(np.delete(np.arange(len(train_index_blocks)),it))])).astype(int)
        else:
            test_ids = list(test_indices[it])
            train_ids = list(train_index_blocks[it])
        
        train_set = dataset[train_ids,:]
        test_set = dataset[test_ids,:]
        
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

        # these are just to keep track of the true, unaltered (not-rescaled) labels for output
        output_train_labels = labels[train_ids]
        all_test_labels.append(labels[test_ids])
        
        train_instance_names = instance_names[train_ids]
        test_instance_names = instance_names[test_ids]
        
        # define validation set manually as the last chunck of the training data (for manually compiling validation accuracy of some altered models)
        val_set_cutoff = int(np.round(train_set.shape[0]*validation_split))
        validation_set = train_set[-val_set_cutoff:,:]
        validation_labels = labels_for_training[-val_set_cutoff:]
        
        # determine stopping point
        if patience == 0:
            tf.random.set_seed(seed)
            # determining optimal number of epochs
            model = model_init(mode,dropout,dropout_rate,use_bias)
            #model.summary()
            history_fit = model.fit(train_set, 
                                    labels_for_training, 
                                    epochs=max_epochs,
                                    validation_split=validation_split, 
                                    verbose=verbose)    
        else:
            tf.random.set_seed(seed)
            # train model
            model = model_init(mode,dropout,dropout_rate,use_bias)
            model.build((train_set.shape[1],))
            if verbose:
                model.summary()
            # The patience parameter is the amount of epochs to check for improvement
            early_stop = tf.keras.callbacks.EarlyStopping(monitor=optimize_for_this, patience=patience)
            history_fit = model.fit(train_set, 
                                labels_for_training, 
                                epochs=max_epochs,
                                validation_split=validation_split, 
                                verbose=verbose,
                                callbacks=[early_stop])
        
        if 'accuracy' in optimize_for_this:
            stopping_point = np.argmax(history_fit.history[optimize_for_this])+1
        else:
            stopping_point = np.argmin(history_fit.history[optimize_for_this])+1
        print('Best training epoch: ',stopping_point,flush=True)
        # train model
        tf.random.set_seed(seed)
        model = model_init(mode,dropout,dropout_rate,use_bias)
        if verbose:
            model.summary()
        history = model.fit(train_set, 
                            labels_for_training, 
                            epochs=stopping_point,
                            validation_split=validation_split, 
                            verbose=verbose)   
    
        if mode == 'nn-class':
            train_predictions, train_predictions_raw, train_loss, train_acc = get_classification_accuracy(model,train_set,labels_for_training,mc_dropout,dropout_reps,loss=True)
            test_predictions, test_predictions_raw, test_loss, test_acc = get_classification_accuracy(model,test_set,labels_for_testing,mc_dropout,dropout_reps,loss=True)        
            if mc_dropout:
                val_predictions, val_predictions_raw, val_loss, val_acc = get_classification_accuracy(model,validation_set,validation_labels,mc_dropout,dropout_reps,loss=True)  
            else:
                train_acc = history.history['accuracy'][-1]
                train_loss = history.history['loss'][-1]
                val_acc = history.history['val_accuracy'][-1]
                val_loss = history.history['val_loss'][-1]
            train_acc_history = np.array(history.history['accuracy'])
            val_acc_history = np.array(history.history['val_accuracy'])
            train_mae_history = np.nan
            val_mae_history = np.nan
    
        elif mode == 'nn-reg':
            test_acc,test_predictions,test_predictions_raw = get_regression_accuracy(model,test_set,labels_for_testing,rescale_factor,min_max_label,stretch_factor_rescaled_labels,mc_dropout,dropout_reps)
            train_loss = history.history['loss'][-1]
            test_loss = np.nan
            train_acc, train_predictions, train_predictions_raw = get_regression_accuracy(model,train_set,labels_for_training,rescale_factor,min_max_label,stretch_factor_rescaled_labels,mc_dropout,dropout_reps)
            val_acc, __, __ = get_regression_accuracy(model,validation_set,validation_labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,mc_dropout,dropout_reps)
            val_loss = history.history['val_loss'][-1]
            train_acc_history = np.nan
            val_acc_history = np.nan
            train_mae_history = np.array(history.history['mae'])
            val_mae_history = np.array(history.history['val_mae'])

        train_acc_per_fold.append(train_acc)
        train_loss_per_fold.append(train_loss)
        validation_acc_per_fold.append(val_acc)
        validation_loss_per_fold.append(val_loss)
        test_acc_per_fold.append(test_acc)
        test_loss_per_fold.append(test_loss) 
        
        all_test_predictions.append(test_predictions)
        all_test_predictions_raw.append(test_predictions_raw)

        if save_model:
            if not os.path.exists(path_to_output):
                os.makedirs(path_to_output)
            model_outpath = os.path.join(path_to_output, model_name+ '_%i'%it)
            model.save( model_outpath )
            print("IUC-NN model saved as:", model_name, "in", path_to_output)
        else:
            model_outpath = ''
    
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

    all_test_labels = np.concatenate(all_test_labels)
    all_test_predictions = np.concatenate(all_test_predictions)
    all_test_predictions_raw = np.concatenate(all_test_predictions_raw)
    
    if len(all_test_labels) > 0:
        confusion_matrix = np.array(tf.math.confusion_matrix(all_test_labels,all_test_predictions))
    else:
        confusion_matrix = np.zeros([n_class,n_class])
        
    output = [
                all_test_labels,
                all_test_predictions,
                all_test_predictions_raw,
                
                avg_train_acc,
                avg_validation_acc,
                avg_test_acc,

                avg_train_loss,
                avg_validation_loss,
                avg_test_loss,
                
                np.array(history.history['loss']),
                np.array(history.history['val_loss']),

                train_acc_history,
                val_acc_history,
                
                train_mae_history,
                val_mae_history,
                
                rescale_labels_boolean,
                rescale_factor,
                np.array(min_max_label),
                stretch_factor_rescaled_labels,
                
                act_f_out,
                model_outpath,
                
                confusion_matrix,
                
                {"data":train_set,
                 "labels":output_train_labels.flatten(),
                 "label_dict":np.unique(labels).astype(str),
                 "test_data":test_set,
                 "test_labels":all_test_labels.flatten(),
                 "id_data":train_instance_names,
                 "id_test_data":test_instance_names,
                 "file_name":model_name,
                 "feature_names":feature_names
                 }
                ]
    
    return output


#error_min_max = [test_predictions_raw-test_predictions_raw_std[0],test_predictions_raw_std[1]-test_predictions_raw]
#plt.errorbar(test_predictions_raw, test_labels, xerr=error_min_max, fmt='o')

    # print("\nVStopped after:", len(history.history['val_loss']), "epochs")
    # print("\nTraining loss: {:5.3f}".format(history.history['loss'][-1]))
    # print("Training accuracy: {:5.3f}".format(history.history['accuracy'][-1]))
    # print("\nValidation loss: {:5.3f}".format(history.history['val_loss'][-1]))
    # print("Validation accuracy: {:5.3f}".format(history.history['val_accuracy'][-1]))

    # print("\nTest accuracy: {:5.3f}".format(test_acc))


#plt.scatter(train_labels_scaled,model.predict(train_set).flatten())

#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

    # if plot_labels_against_features:    
    #     # define figure dimensions
    #     n_features = train_set.shape[1]
    #     n_cols = 4
    #     n_rows = int(n_features/n_cols)+1
    #     fig = plt.figure(figsize=(n_cols*2, n_rows*2))
    #     # specify grid
    #     grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)
    #     for i,feature in enumerate(train_set.T):
    #         fig.add_subplot(grid[i])
    #         plt.scatter(feature,train_labels,c='black',ec=None,s=10)
    #         plt.scatter(feature,train_predictions,c='red',ec=None,s=10,alpha=0.5)
    #         if rescale_features:
    #             plt.xlim(-0.04,1.04)
    #     fig.savefig(os.path.join(path_to_output,'predicted_labels_by_feature.png'),bbox_inches='tight', dpi = 200)
    #     plt.close()
    
    # if plot_training_stats:
    #     fig = plt.figure()
    #     if mode == 'nn-class':            
    #         grid = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.4)
    #         fig.add_subplot(grid[0])
    #         if patience == 0:
    #             acc_at_best_epoch = history_fit.history["val_accuracy"][stopping_point]
    #             plt.plot(history_fit.history['accuracy'],label='train')
    #             plt.plot(history_fit.history['val_accuracy'], label='val')
    #             plt.axvline(stopping_point,linestyle='--',color='red')
    #             plt.axhline(acc_at_best_epoch,linestyle='--',color='red')
    #         else:
    #             plt.plot(history.history['accuracy'],label='train')
    #             plt.plot(history.history['val_accuracy'], label='val')
    #         plt.title('Accuracy')
    #         fig.add_subplot(grid[1])
    #         if patience == 0:
    #             loss_at_best_epoch = history_fit.history["val_loss"][stopping_point]
    #             plt.plot(history_fit.history['loss'],label='train')
    #             plt.plot(history_fit.history['val_loss'], label='val')
    #             plt.axvline(stopping_point,linestyle='--',color='red')
    #             plt.axhline(loss_at_best_epoch,linestyle='--',color='red')
    #         else:
    #             plt.plot(history.history['loss'],label='train')
    #             plt.plot(history.history['val_loss'], label='val')                
    #         plt.title('Loss')                
    #     elif mode == 'nn-reg':
    #         if patience == 0:
    #             mae_at_best_epoch = history_fit.history["val_mae"][stopping_point]
    #             plt.plot(history_fit.history['mae'],label='train')
    #             plt.plot(history_fit.history['val_mae'], label='val')
    #             plt.axvline(stopping_point,linestyle='--',color='red')
    #             plt.axhline(mae_at_best_epoch,linestyle='--',color='red')
    #         else:
    #             plt.plot(history.history['mae'],label='train')
    #             plt.plot(history.history['val_mae'], label='val')
    #         plt.title('Mean Average Error') 
    #     plt.legend(loc='lower right')
    #     fig.savefig(os.path.join(path_to_output,'training_stats.pdf'),bbox_inches='tight')

    # def build_regression_model():
    #     architecture = [tf.keras.layers.Dense(n_layers[0],
    #                                  activation=act_f,
    #                                  input_shape=[train_set.shape[1]], 
    #                                  use_bias=use_bias)]
    #     for i in n_layers[1:]:
    #         architecture.append(tf.keras.layers.Dense(i, activation=act_f))
        
    #     if act_f_out:
    #         architecture.append(tf.keras.layers.Dense(1, activation=act_f_out))    #sigmoid or tanh
    #     else:
    #         architecture.append(tf.keras.layers.Dense(1))
    #     model = tf.keras.Sequential(architecture)
    #     optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=optimizer,
    #                   metrics=['mae','mse'])
    #     return model

    # def build_regression_model():
    #     input_layer = tf.keras.layers.Input(shape=(train_set.shape[1],))
    #     if dropout:
    #         input_layer = tf.keras.layers.Dropout(dropout_rate)(input_layer,training=True)
    #     hidden_1 = tf.keras.layers.Dense(n_layers[0],activation=act_f,use_bias=use_bias)(input_layer)
    #     if dropout:
    #         hidden_1 = tf.keras.layers.Dropout(dropout_rate)(hidden_1,training=True)
    #     for i in n_layers[1:]:
    #         hidden_n = tf.keras.layers.Dense(i, activation=act_f)(hidden_1)
    #         if dropout:
    #             hidden_n = tf.keras.layers.Dropout(dropout_rate)(hidden_n,training=True)
    #     if act_f_out:
    #         outputs = tf.keras.layers.Dense(1, activation=act_f_out)(hidden_n)    #sigmoid or tanh
    #     else:
    #         outputs = tf.keras.layers.Dense(1)(hidden_n)
    #     #my_loss = keras.losses.mean_squared_error
    #     #my_metric = [keras.metrics.mean_absolute_error,keras.metrics.mean_squared_error]
    #     model = tf.keras.Model(outputs)
    #     optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
    #     model.compile(loss='mae',
    #                   optimizer=optimizer,
    #                   metrics=['mae','mse'])
    #     return model

    # class RegressionModel(tf.keras.Model):
    #     def __init__(self, **kwargs):
    #         super().__init__(**kwargs)
    #         self.input_layer = tf.keras.layers.Flatten(input_shape=(train_set.shape[1],))
    #         self.hidden1 = tf.keras.layers.Dense(n_layers[0],activation=act_f,use_bias=use_bias)
    #         self.hidden2 = tf.keras.layers.Dense(n_layers[1],activation=act_f,use_bias=use_bias)
    #         self.hidden3 = tf.keras.layers.Dense(n_layers[2],activation=act_f,use_bias=use_bias)
    #         self.output_layer = tf.keras.layers.Dense(1, activation=act_f_out)
    #         self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        
    #     def call(self, input, training=None):
    #         input_layer = self.input_layer(input)
    #         input_layer = self.dropout_layer(input_layer)
    #         hidden1 = self.hidden1(input_layer)
    #         hidden1 = self.dropout_layer(hidden1, training=True)
    #         hidden2 = self.hidden2(hidden1)
    #         hidden2 = self.dropout_layer(hidden2, training=True)
    #         hidden3 = self.hidden3(hidden2)
    #         hidden3 = self.dropout_layer(hidden3, training=True)
    #         output_layer = self.output_layer(hidden3)
    #         return output_layer


    # model = RegressionModel()
    # optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
    # model.compile(loss='mean_squared_error',
    #               optimizer=optimizer,
    #               metrics=['mae','mse'])



