import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os,glob
from PIL import Image
import pandas as pd


def pad_image(image):
    max_dim = np.max(image[:,:,0].shape)
    max_dim_loc = np.argmax(image[:,:,0].shape)
    padding_array = np.zeros([3,2]).astype(int)
    padding_array[np.abs(max_dim_loc-1)] = np.sort(np.abs(image[:,:,0].shape-max_dim))
    image_out = np.pad(image,list(padding_array))
    return(image_out)

def augment_image(image):
    rot_45 = np.rot90(image)
    rot_90 = np.rot90(rot_45)
    rot_135 = np.rot90(rot_90)
    inverse_orig = image[::-1, :, :]
    inverse_45 = rot_45[::-1, :, :]
    inverse_90 = rot_90[::-1, :, :]
    inverse_135 = rot_135[::-1, :, :]
    # inverse_orig_2 = image[:, ::-1, :]
    # inverse_45_2 = rot_45[:, ::-1, :]
    # inverse_90_2 = rot_90[:, ::-1, :]
    # inverse_135_2 = rot_135[:, ::-1, :]
    # inverse_orig_3 = image[::-1, ::-1, :]
    # inverse_45_3 = rot_45[::-1, ::-1, :]
    # inverse_90_3 = rot_90[::-1, ::-1, :]
    # inverse_135_3 = rot_135[::-1, ::-1, :]
    return([image,
            rot_45,
            rot_90,
            rot_135,
            inverse_orig,
            inverse_45,
            inverse_90,
            inverse_135,
            ])
            # inverse_orig_2,
            # inverse_45_2,
            # inverse_90_2,
            # inverse_135_2,
            # inverse_orig_3,
            # inverse_45_3,
            # inverse_90_3,
            # inverse_135_3


def get_raster_files(raster_folder):
    raster_files = glob.glob(os.path.join(raster_folder,'*.csv'))
    raster_data = np.array([pd.read_csv(file_path).values for file_path in raster_files])
    species_names = np.array(['_'.join(os.path.basename(i).split(' ')[:2]) for i in raster_files])
    return(raster_data,species_names)


# get IUCN labels
labels_file = 'input_data_bromeliaceae/data_for_NN/iucn_assessments.csv'
iucn_data = pd.read_csv(labels_file)
iucn_status = iucn_data.redlistCategory.values
species_names_iucn_pre = iucn_data.tax_accepted_name.values
species_names_iucn_pre = np.array(['_'.join(i.split(' ')[:2]) for i in species_names_iucn_pre])
sorted_indices = np.argsort(species_names_iucn_pre)
species_names_iucn_pre = species_names_iucn_pre[sorted_indices]
iucn_status = iucn_status[sorted_indices]
status_translation_dict = {'Least Concern':0,
                            'Near Threatened':1,
                            'Vulnerable':2,
                            'Endangered':3,
                            'Critically Endangered':4,
                            'Data Deficient':999}
labels_pre = np.array([status_translation_dict[i] for i in iucn_status])
dd_species_ids = np.where(labels_pre==999)[0]
dd_species_names = species_names_iucn_pre[dd_species_ids]
labels = np.delete(labels_pre, dd_species_ids)
species_names_iucn = np.delete(species_names_iucn_pre, dd_species_ids)

# get trait data
trait_file = 'input_data_bromeliaceae/data_for_NN/traits.csv'
trait_df = pd.read_csv(trait_file)
trait_df = trait_df.fillna('none')
#trait_df = trait_df.dropna()
trait_data_species = np.array(['_'.join(i.split(' ')[:2]) for i in trait_df.tax_accepted_name.values])
trait_info = trait_df.tr_growth_form.values
sorted_indices = np.argsort(trait_data_species)
trait_data_species = trait_data_species[sorted_indices]
trait_info = trait_info[sorted_indices]
trait_classes = list(np.unique(trait_info))
num_trait_classes = np.arange(len(trait_classes))
trait_dict = dict(zip(trait_classes,num_trait_classes))
trait_info_num = np.array([trait_dict[i] for i in trait_info])

# load raster data for CNN input
raster_folder = 'input_data_bromeliaceae/data_for_NN/inter_polated_rasters'
occ_rasters,taxon_names = get_raster_files(raster_folder)
sorted_indices = np.argsort(taxon_names)
occ_rasters = occ_rasters[sorted_indices]
taxon_names = taxon_names[sorted_indices]
climate_1_file = 'input_data_bromeliaceae/data_for_NN/CHELSA_bio_1.csv'
climate_12_file = 'input_data_bromeliaceae/data_for_NN/CHELSA_bio_12.csv'
climate_1 = np.nan_to_num(pd.read_csv(climate_1_file).values)
climate_12 = np.nan_to_num(pd.read_csv(climate_12_file).values)
joined_channels = np.array([np.moveaxis(np.array([i,climate_1,climate_12]), 0, -1) for i in occ_rasters])


# select for the training species
iucn_species_ids_with_data = np.where(np.in1d(species_names_iucn, taxon_names))[0]
target_species_names = species_names_iucn[iucn_species_ids_with_data]
target_species_labels = labels[iucn_species_ids_with_data]
cnn_data_target_ids = [key for key, val in enumerate(taxon_names) if val in target_species_names]
trait_data_target_ids = [key for key, val in enumerate(trait_data_species) if val in target_species_names]
trait_data_selected = trait_info_num[trait_data_target_ids]
cnn_data_selected = joined_channels[cnn_data_target_ids]
cnn_data_padded = np.array([pad_image(i) for i in cnn_data_selected])
cnn_data_augmented = np.array([augment_image(i) for i in cnn_data_padded])
n_additional_features = 1
cnn_data_augmented = cnn_data_augmented.reshape([np.prod(cnn_data_augmented.shape[:2])]+list(cnn_data_augmented.shape[2:]))

trait_data_selected_augmented = np.repeat(trait_data_selected,8)
target_species_labels_augmented = np.repeat(target_species_labels,8)
labels = tf.keras.utils.to_categorical(target_species_labels_augmented)



# np.min(cnn_data_selected[:,:,:,dim])
# np.max(cnn_data_selected[:,:,:,dim])
#
# a = cnn_data_selected[0]
# for dim in np.arange(len(a.shape)):
#     label_range = np.max(cnn_data_selected[:,:,:,dim]) - np.min(cnn_data_selected[:,:,:,dim])
#     midpoint_range = np.mean(min_max_label)
#     if reverse:
#         rescaled_labels_tmp = (labels - midpoint_range) / modified_range
#         rescaled_labels = (rescaled_labels_tmp + 0.5) * rescale_factor
#     else:
#         rescaled_labels_tmp = (labels / rescale_factor) - 0.5
#         rescaled_labels = rescaled_labels_tmp * modified_range + midpoint_range
#

#___________________________________BUILD THE CNN MODEL____________________________________
# convolution layers (feature generation)
architecture_conv = []
architecture_conv.append(tf.keras.layers.Conv2D(filters=8,kernel_size=(10,10),strides=(10, 10),activation=None,padding='valid'))
architecture_conv.append(tf.keras.layers.BatchNormalization())
architecture_conv.append(tf.keras.layers.ReLU())
# architecture_conv.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(5,5),strides=(5, 5),activation=None,padding='valid'))
# architecture_conv.append(tf.keras.layers.BatchNormalization())
# architecture_conv.append(tf.keras.layers.ReLU())
architecture_conv.append(tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=(3, 3),padding='same'))
architecture_conv.append(tf.keras.layers.Flatten())
conv_model = tf.keras.Sequential(architecture_conv)

# fully connected NN
architecture_fc = []
architecture_fc.append(tf.keras.layers.Dense(40, activation='relu'))
architecture_fc.append(tf.keras.layers.Dense(20, activation='relu'))
architecture_fc.append(tf.keras.layers.Dense(5, activation='softmax'))  # sigmoid or tanh or softplus
fc_model = tf.keras.Sequential(architecture_fc)

# define the input layer and apply the convolution part of the NN to it
input1 = tf.keras.layers.Input(shape=cnn_data_augmented.shape[1:])
cnn_output = conv_model( input1 )

# define the second input that will come in after the convolution
input2 = tf.keras.layers.Input(shape=(n_additional_features, ))
concatenatedFeatures = tf.keras.layers.Concatenate(axis = 1)([cnn_output, input2])

#output = fc_model(cnn_output)
output = fc_model(concatenatedFeatures)

model = tf.keras.models.Model( [ input1 , input2 ] , output )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#conv_model.summary()
#fc_model.summary()
#__________________________________________________________________________________________





#___________________________________TRAIN THE CNN MODEL____________________________________
history = model.fit([cnn_data_augmented,trait_data_selected_augmented],
                    labels,
                    epochs=200,
                    validation_split=0.2,
                    verbose=1,
                    batch_size=40)
# check training epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

# make predictions of train set with trained and badly overfitted model
pred = model.predict([cnn_data_padded,trait_data_selected])
pred_cats = np.argmax(pred,axis=1)
# plt.scatter(target_species_labels,pred_cats)
# plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'r-')
# plt.xlabel('True values')
# plt.ylabel('Predicted values')
# plt.show()



"""


# the input data
n_instances = 10
img_dimension = [40,30]
channels = 3
n_additional_features = 5
# this is our image data
x1_train = np.random.uniform(size=[n_instances]+img_dimension+[channels])
# these are additional 'manually' generated features
x2_train = np.random.uniform(size=[n_instances,n_additional_features])
labels = np.random.uniform(size=n_instances)



a=x1_train[0]
plt.imshow(a)
plt.show()

b = pad_image(a)
plt.imshow(b)
plt.show()

augmented_image_variants = augment_image(b)

fig = plt.figure()
for i,img in enumerate(augmented_image_variants):
    subplot = fig.add_subplot(2,4,i+1)
    plt.imshow(img)
plt.show()


img_path = '/Users/tobiasandermann/Desktop/UGHT-IRON-HOUSE-LETTER-F-LET-F-2T.jpeg'
from PIL import Image
import numpy as np
im = Image.open(img_path)
im = np.array(im,dtype=np.float32)

a=im




std = 0.05
a=np.array([np.random.random(10),np.random.random(10)])
b=np.array([np.random.normal(a[0],std,len(a[0]))for i in np.arange(100)]).flatten()
c=np.array([np.random.normal(a[1],std,len(a[1]))for i in np.arange(100)]).flatten()

plt.scatter(b,c,s=0.1,c='red')
plt.scatter(a[0],a[1])
plt.show()

"""
