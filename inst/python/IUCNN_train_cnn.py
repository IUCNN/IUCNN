import os, datetime, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf compilation warning
except:
    pass


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def get_accuracy_threshold(probs, labels, threshold=0.75):
    indx = np.where(np.max(probs, axis=1)>threshold)[0]
    res_supported = probs[indx,:]
    labels_supported = labels[indx]
    pred = np.argmax(res_supported, axis=1)
    accuracy = len(pred[pred == labels_supported])/len(pred)
    dropped_frequency = len(pred)/len(labels)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency}


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


def build_cnn_model(input_shape, kernels_conv, pool_size, act_f, act_f_out, n_class, dropout, dropout_rate, pooling_strategy):
    architecture = []
    # # ADD CONVOLUTION
    architecture.append(tf.keras.layers.Conv2D(filters=1,
                                               kernel_size=kernels_conv,
                                               activation=act_f,
                                               input_shape=input_shape[1:]))
    if pooling_strategy == 'max':
        architecture.append(tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                                         strides=(1, 1),
                                                         padding='same'))
    else:
        architecture.append(tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                             strides=(1, 1),
                                                             padding='same'))
    # # ADD CONVOLUTION
    architecture.append(tf.keras.layers.Conv2D(filters=1,
                                               kernel_size=kernels_conv,  # (50,50),
                                               activation=act_f))
    if pooling_strategy == 'max':
        architecture.append(tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                                         strides=(1, 1),
                                                         padding='same'))
    else:
        architecture.append(tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                             strides=(1, 1),
                                                             padding='same'))
    # fully connected layers
    architecture.append(tf.keras.layers.Flatten())
    architecture.append(tf.keras.layers.Dense(20, activation=act_f))
    if dropout:
        architecture.append(MCDropout(dropout_rate))
    architecture.append(tf.keras.layers.Dense(10, activation=act_f))
    if dropout:
        architecture.append(MCDropout(dropout_rate))
    # output layer
    architecture.append(tf.keras.layers.Dense(n_class,
                                              activation=act_f_out))
    model = tf.keras.Sequential(architecture)
    # compile model
    loss_f = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_f,
                  optimizer="adam",
                  metrics=['accuracy'])
    return model


def store_input_files_as_pkl(input_raw,
                             labels):
    import pickle as pkl
    filehandler = open('orchid_data/cnn_test/data/input_raw.pkl', "wb")
    pkl.dump(input_raw, filehandler)
    filehandler.close()
    filehandler = open('orchid_data/cnn_test/data/labels.pkl', "wb")
    pkl.dump(labels, filehandler)
    filehandler.close()


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


def get_cnn_accuracy(model, features, true_labels, mc_dropout, dropout_reps, loss=False):
    if features.shape[0] == 0:
        return np.array([]), np.array([]), np.nan, np.nan
    else:
        if mc_dropout:
            predictions_raw = np.array([model.predict(features) for i in np.arange(dropout_reps)])
            predictions_raw_mean = np.mean(predictions_raw, axis=0)
        else:
            predictions_raw_mean = model.predict(features)
        label_predictions = np.argmax(predictions_raw_mean, axis=1)
        mean_acc_man = np.sum(label_predictions == true_labels) / len(label_predictions)
        if loss:
            if mc_dropout:
                preds = np.array(
                    [model.evaluate(features, true_labels, verbose=0) for i in np.arange(dropout_reps)])
                test_loss, test_acc = np.hsplit(preds, 2)
                mean_loss = np.mean(test_loss)
                # mean_acc = np.mean(test_acc)
            else:
                mean_loss, mean_acc = model.evaluate(features, true_labels, verbose=0)
        else:
            mean_loss = np.nan
            # mean_acc = np.nan
        return label_predictions, predictions_raw_mean, mean_loss, mean_acc_man


def load_input_data_manually():
    import pickle as pkl
    file = open("orchid_data/cnn_test/data/input_raw.pkl",'rb')
    input_raw = pkl.load(file)
    file.close()
    file = open("orchid_data/cnn_test/data/labels.pkl",'rb')
    labels = pkl.load(file)
    file.close()
    max_epochs = 10
    patience = 1
    test_fraction = 0.2
    path_to_output = 'orchid_data/cnn_test/cnn_model'
    act_f = 'relu'
    act_f_out = 'softmax'
    seed = 1234
    dropout = False
    dropout_rate = 0.0
    mc_dropout_reps = 100
    randomize_instances = True
    optimize_for = 'accuracy'
    verbose = 1
    pooling_strategy = 'average'
    cv_k = 2
    balance_classes = False
    no_validation = False
    label_res = 'detail'
    save_model = True



# function to run from R in order to save data objects as numpy arrays
def train_cnn_model(input_raw,
                    labels,
                    max_epochs,
                    patience,
                    cv_k,
                    test_fraction,
                    path_to_output,
                    act_f,
                    act_f_out,
                    seed,
                    dropout,
                    dropout_rate,
                    mc_dropout_reps,
                    randomize_instances,
                    balance_classes,
                    optimize_for,
                    no_validation,
                    pooling_strategy,
                    save_model,
                    verbose
                    ):

    # load settings
    # model hyper-parameters
    cv_k = int(cv_k)
    kernels_conv = (3,3)
    pool_size = (3, 3)
    # training settings
    batch_size = 10 # can be set to a smaller number
    dropout_reps = mc_dropout_reps
    if optimize_for == 'accuracy':
        criterion = 'val_accuracy'
    else:
        criterion = 'val_loss'
    np.random.seed(seed)
    tf.random.set_seed(seed)



    # raw input data
    instance_names = np.array(list(input_raw.keys()))
    data_matrix = np.array([input_raw[i] for i in instance_names]).astype(int)
    labels = np.array(labels).astype(int)
    min_max_label = [min(labels), max(labels)]

    n_labs = len(np.unique(labels))
    if n_labs == 2: # if briad labels, change some default settings
        batch_size = 100

#    label_res = 'detail'
#    if label_res != 'detail':
#        labels[labels < 2] = 0
#        labels[labels > 1] = 1
    # create empty lists for output (per cv fold)
    train_acc_per_fold = []
    train_loss_per_fold = []
    validation_acc_per_fold = []
    validation_loss_per_fold = []
    test_acc_per_fold = []
    test_loss_per_fold = []

    all_test_labels = []
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
    test_count_cv_folds = []

    orig_dataset = data_matrix
    orig_labels = labels



    # define training and test indices
    if cv_k > 1:
        train_index_blocks = iter_test_indices(data_matrix,n_splits = cv_k,shuffle=randomize_instances)
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


    for it, __ in enumerate(train_index_blocks):
        if cv:
            test_indices = train_index_blocks[it] # in case of cv, choose one of the k chunks as test set
            train_indices = np.concatenate(np.array([train_index_blocks[i] for i in list(np.delete(np.arange(len(train_index_blocks)),it))])).astype(int)
            print("Training CV fold %i/%i on %i training instances (%i test instances)..."%(it+1,cv_k,len(train_indices),len(test_indices)),flush=True)
        else:
            test_indices = list(test_indices[it])
            train_indices = list(train_index_blocks[it])
            print("Training model on %i training instances (%i test instances)..."%(len(train_indices),len(test_indices)),flush=True)


        # these are just to keep track of the true, unaltered arrays for output
        orig_train_set = data_matrix[train_indices,:]
        orig_train_labels = labels[train_indices]
        orig_test_set = data_matrix[test_indices,:]
        orig_test_labels = labels[test_indices]
        all_test_labels.append(orig_test_labels)

        # supersample train_ids if balance_class mode is active
        if balance_classes:
            train_indices = supersample_classes(train_indices,labels)

        # define train and test set
        training_data = data_matrix[train_indices,:]
        test_data = orig_test_set
        training_labels = labels[train_indices]
        test_labels = labels[test_indices]
#        train_inst_names = instance_names[train_indices]
#        test_inst_names = instance_names[test_indices]

        # add the extra dimension to the array, describing the 'channel' (#species, #lat, #lon, #channel)
        x_train = training_data.copy().reshape(list(training_data.shape)+[1])
        x_test = test_data.copy().reshape(list(test_data.shape)+[1])
        l = training_labels.copy()
        n_class = len(np.unique(l))
        input_shape = x_train.shape


        # run for set number of iterations, no early stopping
#        log_dir = os.path.join(path_to_output, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#        callbacks=[early_stop, tensorboard_callback]
        if no_validation:
            tf.random.set_seed(seed)
            print('Building model ...', flush=True)
            model = build_cnn_model(input_shape,
                                    kernels_conv,
                                    pool_size,
                                    act_f,
                                    act_f_out,
                                    n_class,
                                    dropout,
                                    dropout_rate,
                                    pooling_strategy)
            if verbose:
                model.summary(print_fn=print(flush=True))
            print('\nDone.\n', flush=True)
            print('Training model ...', flush=True)
            print('Running training for set number of epochs: %i'%max_epochs,flush=True)
            history = model.fit(x_train,
                                l,
                                epochs=max_epochs,
                                verbose=verbose,
                                batch_size=batch_size)
            print('Done.\n', flush=True)
            stopping_point = max_epochs
        else:
            tf.random.set_seed(seed)
            print('Building model ...',flush=True)
            model = build_cnn_model(input_shape,
                                    kernels_conv,
                                    pool_size,
                                    act_f,
                                    act_f_out,
                                    n_class,
                                    dropout,
                                    dropout_rate,
                                    pooling_strategy)
            if verbose:
                model.summary(print_fn=print(flush=True))
            print('\nDone.\n', flush=True)
            early_stop = keras.callbacks.EarlyStopping(monitor=criterion,
                                                       patience=patience)
            print('Training model ...', flush=True)
            if cv:
                history = model.fit(x_train,
                                    l,
                                    epochs=max_epochs,
                                    validation_data=(x_test,test_labels),
                                    verbose=verbose,
                                    callbacks=[early_stop],
                                    batch_size=batch_size)
            else:
                history = model.fit(x_train,
                                    l,
                                    epochs=max_epochs,
                                    validation_split=0.2,
                                    verbose=verbose,
                                    callbacks=[early_stop],
                                    batch_size=batch_size)
            print('Done.\n', flush=True)
            # determine stopping point and export best values
            if criterion == 'val_accuracy':
                stopping_point = np.argmax(history.history['val_accuracy'])
            else:
                stopping_point = np.argmin(history.history['val_loss'])
            print('Best training epoch: ', stopping_point, flush=True)

        if save_model:
            if not os.path.exists(path_to_output):
                os.makedirs(path_to_output)
            model_outpath = os.path.join(path_to_output, 'cnn_model_%i'%it)
            model.save( model_outpath )
            print("CNN model saved at: ", model_outpath,flush=True)
        else:
            model_outpath = ''
        all_model_outpaths.append(model_outpath)



        # get train and validation acc and loss
        train_acc = history.history['accuracy'][stopping_point]
        train_loss = history.history['loss'][stopping_point]
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



        # run test set predictions
        if len(test_labels) > 0:
            print('Predicting labels for test set ...', flush=True)
            test_predictions, test_predictions_raw, test_loss, test_acc = get_cnn_accuracy(model,
                                                                                           x_test,
                                                                                           test_labels,
                                                                                           dropout,
                                                                                           dropout_reps,
                                                                                           loss=True)
            print('Done.\n\n', flush=True)
        else:
            test_loss = np.nan
            test_acc = np.nan
            test_predictions = np.nan
            test_predictions_raw = np.nan
        train_acc_history = np.array(history.history['accuracy'])
        train_mae_history = np.nan
        val_mae_history = np.nan



        # update all output objects with cv-fold results
        train_acc_per_fold.append(train_acc)
        train_loss_per_fold.append(train_loss)
        validation_acc_per_fold.append(val_acc)
        validation_loss_per_fold.append(val_loss)
        test_acc_per_fold.append(test_acc)
        test_loss_per_fold.append(test_loss)

        all_test_predictions.append(test_predictions)
        all_test_predictions_raw.append(test_predictions_raw)

        stopping_points.append(stopping_point + 1)  # add +1 because R does different indexing than python

        training_histories.setdefault('train_rep_%i' % it, np.array(history.history['loss']))
        validation_histories.setdefault('train_rep_%i' % it, val_loss_history)
        train_acc_histories.setdefault('train_rep_%i' % it, train_acc_history)
        val_acc_histories.setdefault('train_rep_%i' % it, val_acc_history)
        train_mae_histories.setdefault('train_rep_%i' % it, train_mae_history)
        val_mae_histories.setdefault('train_rep_%i' % it, val_mae_history)



    # summarize across CV folds
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
        print('Average scores for all folds:', flush=True)
        print('> Test accuracy: %.5f (+- %.5f (std))'%(avg_test_acc,np.std(test_acc_per_fold)), flush=True)
        print('> Test loss: %.5f'%avg_test_loss, flush=True)


    if len(test_labels)>0:
        all_test_labels = np.concatenate(all_test_labels).flatten()
        all_test_predictions = np.concatenate(all_test_predictions)
        all_test_predictions_raw = np.concatenate(all_test_predictions_raw)
        if dropout:
            accthres_tbl = get_confidence_threshold(all_test_predictions_raw,all_test_labels,target_acc=None)
        else:
            accthres_tbl = np.nan
        confusion_matrix = np.array(tf.math.confusion_matrix(all_test_labels,all_test_predictions))
    else:
        all_test_labels = np.nan
        all_test_predictions: np.nan
        all_test_predictions_raw = np.nan
        confusion_matrix = np.zeros([n_class,n_class])
        accthres_tbl = np.nan

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
        data_test = test_data
        labels_test = all_test_labels.flatten()
        train_instance_names = instance_names[train_indices]
        test_instance_names = instance_names[test_indices]



    # sample from categorical
    nreps = 1000
    if dropout:
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


    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(5,5))
    # plt.plot(history.history['accuracy'],label='Training accuracy')
    # plt.plot(history.history['val_accuracy'],label='Validation accuracy')
    # plt.axvline(stopping_point,linestyle='--',color='red',label='Early stopping point')
    # plt.grid(axis='y', linestyle='dashed', which='major', zorder=0)
    # plt.xlabel('Training epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.tight_layout()
    # fig.savefig('orchid_data/cnn_test/training_plot_train_%.2f_val_%.2f_seed_%i_labels_%s.pdf'%(traininig_acc,validation_acc,seed,label_res))
    # print('Training accuracy:',traininig_acc)
    # print('Validation accuracy:',validation_acc)

    # export output
    output = {
        'test_labels': all_test_labels,
        'test_predictions': all_test_predictions,
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

        'rescale_labels_boolean': False,
        'label_rescaling_factor': np.nan,
        'min_max_label': np.array(min_max_label),
        'label_stretch_factor': 1,

        'activation_function': act_f_out,
        'trained_model_path': all_model_outpaths,

        'confusion_matrix': confusion_matrix,
        'mc_dropout': dropout,
        'accthres_tbl': accthres_tbl,
        'true_class_count': true_class_count,
        'predicted_class_count': predicted_class_count,
        'stopping_point': np.array(stopping_points),

        'input_data': {"data": data_train,
                       "labels": labels_train,
                       "label_dict": np.unique(orig_labels).astype(str),
                       "test_data": data_test,
                       "test_labels": labels_test,
                       "id_data": train_instance_names,
                       "id_test_data": test_instance_names,
                       "file_name": os.path.basename(path_to_output),
                       "feature_names": np.nan
                       }
    }

    return output


