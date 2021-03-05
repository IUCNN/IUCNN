import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
except:
    pass

def iucnn_train(dataset, 
                labels,
                mode,
                path_to_output,
                validation_split,
                test_fraction,
                seed,
                instance_names,
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
                plot_training_stats,
                plot_labels_against_features):
    #dropout=True
    #dropout_rate=0.1
    
    def build_classification_model():
        architecture = [layers.Dense(n_layers[0], 
                                     activation=act_f, 
                                     input_shape=[train_set.shape[1]], 
                                     use_bias=use_bias)]
        for i in n_layers[1:]:
            architecture.append(layers.Dense(i, activation=act_f))
        architecture.append(layers.Dense(n_class, 
                                         activation=act_f_out))
        model = keras.Sequential(architecture)
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model

    def build_regression_model():
        architecture = [layers.Dense(n_layers[0],
                                     activation=act_f,
                                     input_shape=[train_set.shape[1]], 
                                     use_bias=use_bias)]
        for i in n_layers[1:]:
            architecture.append(layers.Dense(i, activation=act_f))
        
        if act_f_out:
            architecture.append(layers.Dense(1, activation=act_f_out))    #sigmoid or tanh
        else:
            architecture.append(layers.Dense(1))
        model = keras.Sequential(architecture)
        optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mae','mse'])
        return model

    # def build_regression_model():
    #     inputs = layers.Input(shape=(train_set.shape[1],))
    #     x = layers.Dense(n_layers[0],activation=act_f,use_bias=use_bias)(inputs)
    #     if dropout:
    #         x = layers.Dropout(dropout_rate)(x,training=True)
    #     for i in n_layers[1:]:
    #         x = layers.Dense(i, activation=act_f)(x)
    #         if dropout:
    #             x = layers.Dropout(dropout_rate)(x,training=True)
    #     if act_f_out:
    #         outputs = layers.Dense(1, activation=act_f_out)(x)    #sigmoid or tanh
    #     else:
    #         outputs = layers.Dense(1)(x)
    #     #my_loss = keras.losses.mean_squared_error
    #     #my_metric = [keras.metrics.mean_absolute_error,keras.metrics.mean_squared_error]
    #     model = keras.models.Sequential(inputs,outputs)
    #     optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
    #     model.compile(loss='mae',
    #                   optimizer=optimizer,
    #                   metrics=['mae','mse'])
    #     return model

    def determine_optim_rounding_boundary(regressed_labels,true_labels):
        accs = []
        values= np.linspace(-1,1,100)
        for x in values:
            label_predictions = np.round(regressed_labels + x, 0).astype(int).flatten()
            cat_acc = np.sum(label_predictions==true_labels)/len(label_predictions)
            accs.append(cat_acc)
        plt.plot(values,accs)
        add_value = values[np.where(accs==np.max(accs))[0][0]]
        return(add_value)

    def get_regression_accuracy(model,features,labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels):
        prm_est = model.predict(features).flatten()
        prm_est_rescaled = rescale_labels(prm_est,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True)
        real_labels = rescale_labels(labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels,reverse=True).astype(int).flatten()
        label_predictions = np.round(prm_est_rescaled, 0).astype(int).flatten()
        cat_acc = np.sum(label_predictions==real_labels)/len(label_predictions)
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

    def model_init(mode):
        if mode == 'nn-reg':
            model = build_regression_model()
        elif mode == 'nn-class':    
            model = build_classification_model()
        return model 
    
    
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    # randomize data
    if seed > 0:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    if randomize_instances:
        rnd_indx = np.random.choice(range(len(labels)), len(labels), replace=False)
    else:
        rnd_indx = np.arange(len(labels))
    if rescale_features:
        dataset = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))

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

    rnd_dataset = dataset[rnd_indx,:]
    rnd_labels = labels[rnd_indx]
    rnd_instance_names = instance_names[rnd_indx]

    test_size = int(len(labels)*test_fraction)
    train_set = rnd_dataset[:-test_size,:]
    test_set = rnd_dataset[-test_size:,:]


    # format labels
    rnd_labels_cat  = tf.keras.utils.to_categorical(rnd_labels)
    rnd_labels_scaled = rescaled_labels[rnd_indx]
    
    train_labels_cat = rnd_labels_cat[:-test_size,:]
    train_labels_scaled = rnd_labels_scaled[:-test_size]
    train_labels = rnd_labels[:-test_size]
    train_instance_names = rnd_instance_names[:-test_size]

    test_labels_cat = rnd_labels_cat[-test_size:,:]
    test_labels_scaled = rnd_labels_scaled[-test_size:]
    test_labels = rnd_labels[-test_size:]
    test_instance_names = rnd_instance_names[-test_size:]
    
    n_class = rnd_labels_cat.shape[1]
    
    if mode == 'nn-class':
        labels_for_training = train_labels_cat
        labels_for_testing = test_labels_cat
        optimize_for_this = "val_loss"
    elif mode == 'nn-reg':
        labels_for_training = train_labels_scaled
        labels_for_testing = test_labels_scaled
        optimize_for_this = "val_mae"

    # determine stopping point
    if patience == 0:
        # determining optimal number of epochs
        model = model_init(mode)
        #model.summary()
        history_fit = model.fit(train_set, 
                                labels_for_training, 
                                epochs=max_epochs,
                                validation_split=validation_split, 
                                verbose=verbose)
        stopping_point = np.argmin(history_fit.history[optimize_for_this])+1
        # train model
        model = model_init(mode)
        model.summary()
        history = model.fit(train_set, 
                            labels_for_training, 
                            epochs=stopping_point,
                            validation_split=validation_split, 
                            verbose=verbose)        
    else:
        # train model
        model = model_init(mode)
        model.summary()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor=optimize_for_this, patience=patience)
        history_fit = model.fit(train_set, 
                            labels_for_training, 
                            epochs=max_epochs,
                            validation_split=validation_split, 
                            verbose=verbose,
                            callbacks=[early_stop])
        stopping_point = np.argmin(history_fit.history[optimize_for_this])+1
        # train model
        model = model_init(mode)
        model.summary()
        history = model.fit(train_set, 
                            labels_for_training, 
                            epochs=stopping_point,
                            validation_split=validation_split, 
                            verbose=verbose)        

        
            
    if mode == 'nn-class':
        test_loss, test_acc = model.evaluate(test_set, 
                                             labels_for_testing, 
                                             verbose=verbose)
        train_predictions_raw = model.predict(train_set, verbose=verbose)
        train_predictions = np.argmax(train_predictions_raw, axis=1)
        test_predictions_raw = model.predict(test_set, verbose=verbose)
        test_predictions = np.argmax(test_predictions_raw, axis=1)
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_acc_history = np.array(history.history['accuracy'])
        val_acc_history = np.array(history.history['val_accuracy'])
        train_mae_history = np.nan
        val_mae_history = np.nan
        
    elif mode == 'nn-reg':
        test_acc,test_predictions,test_predictions_raw = get_regression_accuracy(model,test_set,labels_for_testing,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
        test_loss = np.nan
        train_acc, train_predictions, train_predictions_raw = get_regression_accuracy(model,train_set,labels_for_training,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
        val_set_cutoff = int(np.round(train_set.shape[0]*validation_split))
        validation_set = train_set[-val_set_cutoff:,:]
        validation_labels = labels_for_training[-val_set_cutoff:]
        val_acc, __, __ = get_regression_accuracy(model,validation_set,validation_labels,rescale_factor,min_max_label,stretch_factor_rescaled_labels)
        train_acc_history = np.nan
        val_acc_history = np.nan
        train_mae_history = np.array(history.history['mae'])
        val_mae_history = np.array(history.history['val_mae'])

    

    if plot_labels_against_features:    
        # define figure dimensions
        n_features = train_set.shape[1]
        n_cols = 4
        n_rows = int(n_features/n_cols)+1
        fig = plt.figure(figsize=(n_cols*2, n_rows*2))
        # specify grid
        grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)
        for i,feature in enumerate(train_set.T):
            fig.add_subplot(grid[i])
            plt.scatter(feature,train_labels,c='black',ec=None,s=10)
            plt.scatter(feature,train_predictions,c='red',ec=None,s=10,alpha=0.5)
            if rescale_features:
                plt.xlim(-0.04,1.04)
        fig.savefig(os.path.join(path_to_output,'predicted_labels_by_feature.png'),bbox_inches='tight', dpi = 200)
        plt.close()
    
    if plot_training_stats:
        fig = plt.figure()
        if mode == 'nn-class':            
            grid = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.4)
            fig.add_subplot(grid[0])
            if patience == 0:
                acc_at_best_epoch = history_fit.history["val_accuracy"][stopping_point]
                plt.plot(history_fit.history['accuracy'],label='train')
                plt.plot(history_fit.history['val_accuracy'], label='val')
                plt.axvline(stopping_point,linestyle='--',color='red')
                plt.axhline(acc_at_best_epoch,linestyle='--',color='red')
            else:
                plt.plot(history.history['accuracy'],label='train')
                plt.plot(history.history['val_accuracy'], label='val')
            plt.title('Accuracy')
            fig.add_subplot(grid[1])
            if patience == 0:
                loss_at_best_epoch = history_fit.history["val_loss"][stopping_point]
                plt.plot(history_fit.history['loss'],label='train')
                plt.plot(history_fit.history['val_loss'], label='val')
                plt.axvline(stopping_point,linestyle='--',color='red')
                plt.axhline(loss_at_best_epoch,linestyle='--',color='red')
            else:
                plt.plot(history.history['loss'],label='train')
                plt.plot(history.history['val_loss'], label='val')                
            plt.title('Loss')                
        elif mode == 'nn-reg':
            if patience == 0:
                mae_at_best_epoch = history_fit.history["val_mae"][stopping_point]
                plt.plot(history_fit.history['mae'],label='train')
                plt.plot(history_fit.history['val_mae'], label='val')
                plt.axvline(stopping_point,linestyle='--',color='red')
                plt.axhline(mae_at_best_epoch,linestyle='--',color='red')
            else:
                plt.plot(history.history['mae'],label='train')
                plt.plot(history.history['val_mae'], label='val')
            plt.title('Mean Average Error') 
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(path_to_output,'training_stats.pdf'),bbox_inches='tight')

    # print("\nVStopped after:", len(history.history['val_loss']), "epochs")
    # print("\nTraining loss: {:5.3f}".format(history.history['loss'][-1]))
    # print("Training accuracy: {:5.3f}".format(history.history['accuracy'][-1]))
    # print("\nValidation loss: {:5.3f}".format(history.history['val_loss'][-1]))
    # print("Validation accuracy: {:5.3f}".format(history.history['val_accuracy'][-1]))

    # print("\nTest accuracy: {:5.3f}".format(test_acc))
    
    model_outpath = os.path.join(path_to_output, model_name)
    model.save( model_outpath )
    print("IUC-NN model saved as:", model_name, "in", path_to_output)
    
    output = [
                test_labels,
                test_predictions,
                test_predictions_raw,
                
                train_acc,
                val_acc,
                test_acc,

                history.history['loss'][-1],
                history.history['val_loss'][-1],
                test_loss,
                
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
                
                {"data":train_set,
                 "labels":train_labels,
                 "label_dict":np.unique(labels).astype(str),
                 "test_data":test_set,
                 "test_labels":test_labels,
                 "id_data":train_instance_names,
                 "id_test_data":test_instance_names,
                 "file_name":model_name,
                 "feature_names":np.arange(dataset.shape[1])+1
                 }
                ]
    
    return output

#plt.scatter(train_labels_scaled,model.predict(train_set).flatten())

