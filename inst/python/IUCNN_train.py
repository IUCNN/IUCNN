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
                path_to_output,
                validation_split,
                test_fraction,
                seed,
                verbose, 
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
        model.compile(loss='mean_squared_error',
                      optimizer="adam",        #rmsprop
                      metrics=['mae','mse'])
        return model


    def get_regression_accuracy(model,features,labels,max_label):
        prm_est = model.predict(features)*max_label
        label_predictions = np.round(prm_est, 0).astype(int)
        real_labels = (labels*max_label).astype(int)
        cat_acc = np.sum(label_predictions==real_labels)/len(label_predictions)
        return cat_acc, prm_est
    
    
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
    reordered_labels = labels[rnd_indx]

    test_size = int(len(labels)*test_fraction)
    train_set = rnd_dataset[:-test_size,:]
    test_set = rnd_dataset[-test_size:,:]
    
    if mode == 'classification':
        # format labels
        rnd_labels  = tf.keras.utils.to_categorical(reordered_labels)
        train_labels_cat = rnd_labels[:-test_size,:]
        train_labels = reordered_labels[:-test_size]
        test_labels = rnd_labels[-test_size:,:]
        n_class = rnd_labels.shape[1]

        model = build_classification_model()
        model.summary()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        history = model.fit(train_set, 
                            train_labels_cat, 
                            epochs=max_epochs,
                            validation_split=validation_split, 
                            verbose=verbose,
                            callbacks=[early_stop])
        test_loss, test_acc = model.evaluate(test_set, 
                                   test_labels, 
                                   verbose=verbose)
        
        prm_est = model.predict(test_set, verbose=verbose)
        test_predictions = np.argmax(prm_est, axis=1)

        train_predictions = np.argmax(model.predict(train_set, verbose=verbose), axis=1)

    elif mode == 'regression':
        # format labels
        max_label = max(reordered_labels)
        train_labels = reordered_labels[:-test_size]
        reordered_labels_scaled = ((reordered_labels/max_label)-0.5)*2
        train_labels_scaled = reordered_labels_scaled[:-test_size]
        test_labels_scaled = reordered_labels_scaled[-test_size:]
        
        model = build_regression_model()
        model.summary()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        history = model.fit(train_set, 
                            train_labels_scaled, 
                            epochs=max_epochs,
                            validation_split=validation_split, 
                            verbose=verbose,
                            callbacks=[early_stop])
        test_acc,test_predictions = get_regression_accuracy(model,test_set,test_labels_scaled,max_label)
        if return_categorical:
            test_predictions = np.round(test_predictions, 0).astype(int)
        test_loss = np.nan
        train_acc, train_predictions = get_regression_accuracy(model,train_set,train_labels_scaled,max_label)
        history.history['accuracy'] = [train_acc]
        history.history['val_accuracy'] = [np.nan]


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
    

    # print("\nVStopped after:", len(history.history['val_loss']), "epochs")
    # print("\nTraining loss: {:5.3f}".format(history.history['loss'][-1]))
    # print("Training accuracy: {:5.3f}".format(history.history['accuracy'][-1]))
    # print("\nValidation loss: {:5.3f}".format(history.history['val_loss'][-1]))
    # print("Validation accuracy: {:5.3f}".format(history.history['val_accuracy'][-1]))

    # print("\nTest accuracy: {:5.3f}".format(test_acc))
    
    res = [history.history['loss'][-1],
           history.history['accuracy'][-1],
           history.history['val_loss'][-1],
           history.history['val_accuracy'][-1],
           test_loss, test_acc]
    
    model.save( os.path.join(path_to_output, model_name) )
    print("IUC-NN model saved as:", model_name, "in", path_to_output)
    return [reordered_labels[-test_size:], test_predictions, res]

