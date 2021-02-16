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
                rescaled_labels,
                path_to_output,
                validation_split,
                test_fraction,
                seed,
                verbose,
                n_labels,
                lab_range,
                model_name,
                max_epochs,
                n_layers,
                use_bias,
                act_f,
                patience,
                randomize_instances,
                mode,
                rescale_features,
                return_categorical,
                plot_training_stats,
                plot_labels_against_features):

    
    def build_classification_model():
        architecture = [layers.Dense(n_layers[0], 
                                     activation=act_f, 
                                     input_shape=[train_set.shape[1]], 
                                     use_bias=use_bias)]
        for i in n_layers[1:]:
            architecture.append(layers.Dense(i, activation=act_f))
        architecture.append(layers.Dense(n_class, 
                                         activation='softmax'))
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
        architecture.append(layers.Dense(1, activation='tanh'))    #sigmoid or tanh
        model = keras.Sequential(architecture)
        optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mae','mse'])
        return model


    def get_regression_accuracy(model,features,labels,n_labels,lab_range):
        prm_est = model.predict(features).flatten()
        prm_est_rescaled = rescale_labels(prm_est,n_labels,lab_range)
        label_predictions = np.round(prm_est_rescaled, 0).astype(int).flatten()
        real_labels = rescale_labels(labels,n_labels,lab_range).astype(int).flatten()
        cat_acc = np.sum(label_predictions==real_labels)/len(label_predictions)
        return cat_acc, prm_est_rescaled
    
    def rescale_labels(labels,n_labels,lab_range):
        if n_labels == 0:
            rescaled_labels = labels
        else:
            rescaled_labels = ((labels/lab_range) +0.5) * (n_labels-1)
        return(rescaled_labels)

    def model_init(mode):
        if mode == 'regression':
            model = build_regression_model()
        elif mode == 'classification':    
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
    
    
    rnd_dataset = dataset[rnd_indx,:]
    rnd_labels = labels[rnd_indx]

    test_size = int(len(labels)*test_fraction)
    train_set = rnd_dataset[:-test_size,:]
    test_set = rnd_dataset[-test_size:,:]


    # format labels
    rnd_labels_cat  = tf.keras.utils.to_categorical(rnd_labels)
    rnd_labels_scaled = rescaled_labels[rnd_indx]
    
    train_labels_cat = rnd_labels_cat[:-test_size,:]
    train_labels_scaled = rnd_labels_scaled[:-test_size]
    train_labels = rnd_labels[:-test_size]

    test_labels_cat = rnd_labels_cat[-test_size:,:]
    test_labels_scaled = rnd_labels_scaled[-test_size:]
    test_labels = rnd_labels[-test_size:]
    
    n_class = rnd_labels_cat.shape[1]
    
    if mode == 'classification':
        labels_for_training = train_labels_cat
        labels_for_testing = test_labels_cat
        optimize_for_this = "val_loss"
    elif mode == 'regression':
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
        stopping_point = np.argmin(history_fit.history[optimize_for_this])
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
        history = model.fit(train_set, 
                            labels_for_training, 
                            epochs=max_epochs,
                            validation_split=validation_split, 
                            verbose=verbose,
                            callbacks=[early_stop])
        
            
    if mode == 'classification':
        test_loss, test_acc = model.evaluate(test_set, 
                                             labels_for_testing, 
                                             verbose=verbose)
        train_predictions = np.argmax(model.predict(train_set, verbose=verbose), axis=1)
        test_predictions = np.argmax(model.predict(test_set, verbose=verbose), axis=1)
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
    elif mode == 'regression':
        test_acc,test_predictions = get_regression_accuracy(model,test_set,labels_for_testing,n_labels,lab_range)
        if return_categorical:
            test_predictions = np.round(test_predictions, 0).astype(int)
        train_acc, train_predictions = get_regression_accuracy(model,train_set,labels_for_training,n_labels,lab_range)
        test_loss = np.nan
        val_acc = np.nan


    res = [history.history['loss'][-1],
           train_acc,
           history.history['val_loss'][-1],
           val_acc,
           test_loss,
           test_acc]
    

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
        if mode == 'classification':            
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
        elif mode == 'regression':
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
    
    model.save( os.path.join(path_to_output, model_name) )
    print("IUC-NN model saved as:", model_name, "in", path_to_output)
    return [test_labels, test_predictions, res]

#plt.scatter(train_labels_scaled,model.predict(train_set).flatten())

