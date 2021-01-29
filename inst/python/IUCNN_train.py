import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
except:
    pass

# def iucnn_train(dataset, labels,
#                 path_to_output="",
#                 validation_split=0.1,
#                 test_fraction=0.1,
#                 seed=1234,
#                 verbose=0, #can be 0, 1, 2
#                 model_name="iuc_nn_model",
#                 max_epochs=1000,
#                 n_layers=[60,60,20],
#                 use_bias=1,
#                 act_f="relu",
#                 patience=500):


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
                patience):
    
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
    
    # randomize data
    if seed > 0:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    rnd_indx = np.random.choice(range(len(labels)), len(labels), replace=False)
    rnd_dataset = dataset[rnd_indx,:] 
    rnd_labels  = tf.keras.utils.to_categorical(labels[rnd_indx])

    test_size = int(len(labels)*test_fraction)
    train_set = rnd_dataset[:-test_size,:]
    train_labels = rnd_labels[:-test_size,:]
    test_set = rnd_dataset[-test_size:,:]
    test_labels = rnd_labels[-test_size:,:]
    n_class = rnd_labels.shape[1]
    
    model = build_classification_model()
    model.summary()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    history = model.fit(train_set, 
                        train_labels, 
                        epochs=max_epochs,
                        validation_split=validation_split, 
                        verbose=verbose,
                        callbacks=[early_stop])

    # print("\nVStopped after:", len(history.history['val_loss']), "epochs")
    # print("\nTraining loss: {:5.3f}".format(history.history['loss'][-1]))
    # print("Training accuracy: {:5.3f}".format(history.history['accuracy'][-1]))
    # print("\nValidation loss: {:5.3f}".format(history.history['val_loss'][-1]))
    # print("Validation accuracy: {:5.3f}".format(history.history['val_accuracy'][-1]))

    test_loss, test_acc = model.evaluate(test_set, 
                               test_labels, 
                               verbose=verbose)

    # print("\nTest accuracy: {:5.3f}".format(test_acc))

    prm_est = model.predict(test_set, verbose=verbose)
    predictions = np.argmax(prm_est, axis=1)
    
    res = [history.history['loss'][-1],
           history.history['accuracy'][-1],
           history.history['val_loss'][-1],
           history.history['val_accuracy'][-1],
           test_loss, test_acc]
    
    model.save( os.path.join(path_to_output, model_name) )
    print("IUC-NN model saved as:", model_name, "in", path_to_output)
    return [test_labels, predictions, res]

