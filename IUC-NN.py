import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
except:
    pass


def iucnn_train(dataset, labels,
                path_to_output="",
                validation_split=0.1,
                test_fraction=0.1,
                seed=1234,
                verbose=0, #can be 0, 1, 2
                model_name="iuc_nn_model",
                max_epochs=1000,
                n_layers=[60,60,20],
                use_bias=1,
                act_f="relu",
                patience=500):
    
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
    rnd_indx = np.random.choice(range(len(labels)), len(labels))
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

    print("\nVStopped after:", len(history.history['val_loss']), "epochs")
    print("\nValidation Accuracy: {:5.3f}".format(history.history['val_loss'][-1]))
    print("Validation Accuracy: {:5.3f}".format(history.history['val_accuracy'][-1]))

    loss, acc = model.evaluate(test_set, 
                               test_labels, 
                               verbose=verbose)

    print("\nTest Accuracy: {:5.3f}".format(acc))

    prm_est = model.predict(test_set, verbose=verbose)
    predictions = np.argmax(prm_est, axis=1)

    from sklearn.metrics import confusion_matrix
    cM = confusion_matrix(np.argmax(test_labels, axis=1), predictions)
    print("Confusion matrix (test set):\n", cM)
    rescaled_cM = (np.array(cM).T / np.sum(np.array(cM), 1)).T
    print(rescaled_cM)

    model.save(model_name)
    print("IUC-NN model saved as:", model_name)


def iucnn_predict(feature_set,model_dir,
                  verbose=0,
                  return_prob=False):
    print("Loading model...")
    model = tf.keras.models.load_model(model_dir)
    prm_est = model.predict(feature_set, verbose=verbose)
    predictions = np.argmax(prm_est, axis=1)
    if return_prob:
        return [predictions, prm_est]
    else:
        return [predictions]



## EXAMPLE TRAIN
f = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/2_main_iucn_full_clean_broad_features/2_main_iucn_full_clean_broad_features_fulldsRESCALE.txt"
l = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/2_main_iucn_full_clean_broad_features/2_main_iucn_full_clean_broad_labels_fullds.txt"

f = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/1_main_iucn_full_clean_detailed_features/1_main_iucn_full_clean_detailed_features_fulldsRESCALE.txt"
l = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/1_main_iucn_full_clean_detailed_features/1_main_iucn_full_clean_detailed_labels_fullds.txt"

# read tables (will be done by R)
dataset = np.loadtxt(f)
labels  = np.loadtxt(l) - 1 # class numbers must start from 0

iucnn_train(dataset, labels)



## EXAMPLE PREDICT
model = "/Users/dsilvestro/Software/IUC-NN/iuc_nn_model"
#f = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/2_main_iucn_full_clean_broad_features/2_main_iucn_full_clean_broad_features_fulldsRESCALE.txt"
feature_set = np.loadtxt(f)
res = iucnn_predict(feature_set, model)
