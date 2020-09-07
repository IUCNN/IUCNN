import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
except:
    pass


f = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/2_main_iucn_full_clean_broad_features/2_main_iucn_full_clean_broad_features_fulldsRESCALE.txt"
l = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/2_main_iucn_full_clean_broad_features/2_main_iucn_full_clean_broad_labels_fullds.txt"

f = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/1_main_iucn_full_clean_detailed_features/1_main_iucn_full_clean_detailed_features_fulldsRESCALE.txt"
l = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/1_main_iucn_full_clean_detailed_features/1_main_iucn_full_clean_detailed_labels_fullds.txt"


dataset = np.loadtxt(f)
labels  = np.loadtxt(l) - 1

test_fraction = 0.1
validation_split = 0.1
seed = 1234
verbose = 0
model_name = "iuc_nn_model"
max_epochs = 1000




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


def build_classification_model(train_set, n_class):
      model = keras.Sequential([
      layers.Dense(60, activation='relu', input_shape=[train_set.shape[1]], use_bias=1),
      layers.Dense(60, activation='relu'),
      layers.Dense(20, activation='relu'),
      layers.Dense(n_class, activation='softmax')
      ])
      model.compile(loss='categorical_crossentropy',
                optimizer="adam",
                metrics=['accuracy'])
      return model

model = build_classification_model(train_set, rnd_labels.shape[1])

model.summary()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

history = model.fit(train_set, train_labels, 
                          epochs=max_epochs,
                          validation_split = validation_split, 
                          verbose=verbose,
                          callbacks=[early_stop])

print("\nVStopped after:", len(history.history['val_loss']), "epochs")
print("\nValidation Accuracy: {:5.3f}".format(history.history['val_loss'][-1]))
print("Validation Accuracy: {:5.3f}".format(history.history['val_accuracy'][-1]))

loss, acc = model.evaluate(test_set, test_labels, 
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

def iucnn_predict(model_dir,feature_set):
    print("Loading model...")
    model = tf.keras.models.load_model(model_dir)
    prm_est = model.predict(feature_set, verbose=verbose)
    predictions = np.argmax(prm_est, axis=1)
    return predictions

m = "/Users/dsilvestro/Software/IUC-NN/iuc_nn_model"
#f = "/Users/dsilvestro/Documents/Projects/Ongoing/Zizka-IUC-NN/20200213/2_main_iucn_full_clean_broad_features/2_main_iucn_full_clean_broad_features_fulldsRESCALE.txt"
feature_set = np.loadtxt(f)
res = iucnn_predict(m, feature_set)
