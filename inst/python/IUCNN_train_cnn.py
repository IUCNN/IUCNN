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

def build_cnn_model(input_shape, n_filters, kernels_conv, pool_size, act_f, act_f_out, n_class, dropout, dropout_rate):
    # preprocessing
    architecture = [tf.keras.layers.Conv2D(filters=n_filters,
                                           kernel_size=kernels_conv,
                                           activation=act_f,
                                           input_shape=input_shape[1:])]
    architecture.append(tf.keras.layers.AveragePooling2D(pool_size=pool_size, # or MaxPooling2D
                                                         strides=(1, 1)))

    # fully connected
    architecture.append(tf.keras.layers.Flatten())
    architecture.append(tf.keras.layers.Dense(5, activation=act_f))
    if dropout:
        architecture.append(MCDropout(dropout_rate))

    # output layer
    architecture.append(tf.keras.layers.Dense(n_class,
                                              activation=act_f_out))

    model = tf.keras.Sequential(architecture)
    # train model
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

def load_input_data_manually():
    import pickle as pkl
    file = open("orchid_data/cnn_test/data/input_raw.pkl",'rb')
    input_raw = pkl.load(file)
    file.close()
    file = open("orchid_data/cnn_test/data/labels.pkl",'rb')
    labels = pkl.load(file)
    file.close()
    max_epochs = 100
    patience = 20
    test_fraction = 0.2
    model_name = 'orchid_data/cnn_test/cnn_model'
    act_f = 'relu'
    act_f_out = 'softmax'
    seed = 1234
    dropout = False
    dropout_rate = 0.0
    mc_dropout_reps = 100
    randomize_instances = True
    optimize_for = 'accuracy'
    verbose = 1


# function to run from R in order to save data objects as numpy arrays
def train_cnn_model(input_raw,
                    labels,
                    max_epochs,
                    patience,
                    test_fraction,
                    model_name,
                    act_f,
                    act_f_out,
                    seed,
                    dropout,
                    dropout_rate,
                    mc_dropout_reps,
                    randomize_instances,
                    optimize_for,
                    verbose
                    ):

    #load_input_data_manually()

    instance_names = np.array(list(input_raw.keys()))
    data_matrix = np.array([input_raw[i] for i in instance_names]).astype(int)

    labels = np.array(labels).astype(int)

    if optimize_for == 'accuracy':
        criterion = 'val_accuracy'
    else:
        criterion = 'val_loss'

    validation_fraction = 0.2
    dropout_reps = mc_dropout_reps
    # model hyper-parameters
    n_filters = 32
    kernels_conv = (3,3)
    pool_size = (3, 3)
    # training settings
    batch_size = 100 # can be set to a smaller number

    np.random.seed(seed)
    tf.random.set_seed(seed)

    if randomize_instances:
        rnd_indx = np.random.choice(range(len(labels)), len(labels), replace=False)
    else:
        rnd_indx = np.arange(len(labels))

    min_max_label = [min(labels), max(labels)]

    test_size = int(len(labels) * test_fraction)
    train_indices = rnd_indx[:-test_size]
    train_inst_names = instance_names[train_indices]
    test_indices = rnd_indx[-test_size:]
    test_inst_names = instance_names[test_indices]


    #data_matrix = data_matrix.reshape(list(data_matrix.shape[:-1]))


    training_data = data_matrix[train_indices, :]
    training_labels = labels[train_indices]
    test_data = data_matrix[test_indices, :]
    test_labels = labels[test_indices]

    # add the extra dimension to the array, describing the 'channel' (#species, #lat, #lon, #channel)
    x_train = training_data.copy().reshape(list(training_data.shape)+[1])
    x_test = test_data.copy().reshape(list(test_data.shape)+[1])
    l = training_labels.copy()

    n_class = len(np.unique(l))
    input_shape = x_train.shape

    log_dir= os.path.join(model_name, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    model = build_cnn_model(input_shape,
                            n_filters,
                            kernels_conv,
                            pool_size,
                            act_f,
                            act_f_out,
                            n_class,
                            dropout,
                            dropout_rate)

    model.summary()
    early_stop = keras.callbacks.EarlyStopping(monitor=criterion,
                                               patience=patience)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train,
                        l,
                        epochs=max_epochs,
                        validation_split=validation_fraction,
                        verbose=verbose,
                        callbacks=[early_stop, tensorboard_callback],
                        batch_size=batch_size)

    # save the trained model
    model_out_path = os.path.join(model_name,'saved_model_seed_%i'%(seed))
    model.save(model_out_path)

    if criterion == 'val_accuracy':
        stopping_point = np.argmax(history.history['val_accuracy'])
    else:
        stopping_point = np.argmin(history.history['val_loss'])
    traininig_acc = history.history['accuracy'][stopping_point]
    validation_acc = history.history['val_accuracy'][stopping_point]
    traininig_loss = history.history['loss'][stopping_point]
    validation_loss = history.history['val_loss'][stopping_point]



    if test_size == 0:
        confusion_matrix = np.zeros([n_class,n_class])
        accthres_tbl = np.nan
        true_class_count = np.nan
        predicted_class_count = np.nan
        test_label_predictions = np.nan
        test_acc = np.nan
    else:
        if dropout:
            predictions_raw = np.array([model.predict(x_test) for i in np.arange(dropout_reps)])
            predictions_raw_mean = np.mean(predictions_raw, axis=0)
        else:
            predictions_raw_mean = model.predict(x_test)
        test_label_predictions = np.argmax(predictions_raw_mean, axis=1)
        test_label_predictions = np.array(test_label_predictions).reshape(test_label_predictions.shape[0])
        test_labels = np.array(test_labels).reshape(test_labels.shape[0])

        test_acc = np.sum(test_label_predictions.astype(int) == test_labels) / len(test_labels)

        if dropout:
            accthres_tbl = get_confidence_threshold(predictions_raw_mean,test_labels,target_acc=None)
        else:
            accthres_tbl = np.nan
        confusion_matrix = np.array(tf.math.confusion_matrix(test_labels, test_label_predictions))
        true_class_count = [list(test_labels).count(i) for i in np.arange(max(labels)+1)]
        predicted_class_count = [list(test_label_predictions).count(i) for i in np.arange(max(labels)+1)]

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

    output = {
        'test_labels': test_labels,
        'test_predictions': test_label_predictions,
        'test_predictions_raw': np.nan,

        'training_accuracy': traininig_acc,
        'validation_accuracy': validation_acc,
        'test_accuracy': test_acc,

        'training_loss': traininig_loss,
        'validation_loss': validation_loss,
        'test_loss': np.nan,

        'training_loss_history': [history.history['loss']],
        'validation_loss_history': [history.history['val_loss']],

        'training_accuracy_history': [history.history['accuracy']],
        'validation_accuracy_history': [history.history['val_accuracy']],

        'training_mae_history': np.nan,
        'validation_mae_history': np.nan,

        'rescale_labels_boolean': False,
        'label_rescaling_factor': np.nan,
        'min_max_label': np.array(min_max_label),
        'label_stretch_factor': 1,

        'activation_function': act_f_out,
        'trained_model_path': model_out_path,

        'confusion_matrix': confusion_matrix,
        'mc_dropout': dropout,
        'accthres_tbl': accthres_tbl,
        'true_class_count': true_class_count,
        'predicted_class_count': predicted_class_count,
        'stopping_point': np.array([stopping_point+1]),

        'input_data': {"data": training_data,
                       "labels": training_labels,
                       "label_dict": np.unique(labels).astype(str),
                       "test_data": test_data,
                       "test_labels": test_labels,
                       "id_data": train_inst_names,
                       "id_test_data": test_inst_names,
                       "file_name": os.path.basename(model_name),
                       "feature_names": np.nan
                       }
    }
    return output


