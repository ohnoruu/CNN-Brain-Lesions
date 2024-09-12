# imports
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense 
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import ModelCheckpoint
from pymongo import MongoClient
import gridfs
import re
import numpy as np
import os 
import nibabel as nib
import scipy.ndimage as ndi
import logging

# import filenames to load from MongoDB
from filenames import training_names, label_names, testing_names
# import components
from components.mongofunctions import extract_subject_num, upload_nifti_files, retrieve_nifti
from components.cnn_model import DataGenerator, classification, visualization, save_nifti

# configuration for MongoDB
from credentials import MONGO_URI
DB_NAME = 'lesion_dataset'
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)
fs_training_images = gridfs.GridFS(db, collection='training_images')
fs_labels = gridfs.GridFS(db, collection='labels')
fs_testing_images = gridfs.GridFS(db, collection='testing_images')

# TRAINING 
# image_paths and labels were previously imported (see imports)
batch_size = 32 # num of samples in each batch
training_data = []
labels = []

for name in training_names:
    image_data = retrieve_nifti(name, fs_training_images)
    training_data.append(image_data)

for name in label_names:
    label_data = retrieve_nifti(name, fs_labels)
    labels.append(label_data)

train_generator = DataGenerator(training_data, labels, batch_size=batch_size, shuffle=True) # instance of DataGenerator class
#loads, preprocesses images and corresponding labels, shuffles after each epoch. 

shape_reference = retrieve_nifti(training_names[0]) # used to retrieve shape of image for training
input_shape = shape_reference.get_fdata().shape
# subject 2 (index 1) will be used as subject 1 has a different shape than usual
image_shape = train_generator.get_nifti_shape(input_shape)
model, feature_maps = classification(input_shape=image_shape) # creation of 3D ResNet model using input shape
# the function 'classification' returns two values: the model and the feature maps (for later visualization task) in a tuple. Listing both model and feature_maps will unpack the tuple
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile is a built-in method of Keras and is important for training. This method configures the model for training.
# optimizer - adam (adaptive moment estimation) - adjusts learning rate during training
# loss - binary crossentropy - used for binary classification tasks
# metrics - accuracy - used to evaluate the model's performance

epochs = 10
model.fit(train_generator, epochs=epochs) # initiate training
# trains model using data generator for 10 epochs.
# the first parameter of model.fit, x, is the input data. In this case, the input data is an instance of the class DataGenerator, which returns both labels and patient data as NumPy arrays.


# SAVING MODEL AND TRAINING WEIGHTS
checkpoint_directory = 'model_checkpoints' # provide file path later to upload data to computer
os.makedirs(checkpoint_directory, exist_ok=True) # create checkpoint directory

checkpoint_path = os.path.join(checkpoint_directory, 'model_checkpoint.h5')
# specifies path where model checkpoints will be saved

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy', # monitor accuracy
    mode='max', # refers to goal of maximizing accuracy
    save_best_ony=True, # saves most accurate model
    save_weights_only=True,
    verbose=1 # specifies output (1 for print)
)

model.fit(
    train_generator,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)

"""
# VISUALIZATION
# Getting image shape and data:
input_image_path = testing_paths[1] # path to image that will be used for evaluation
provided_nifti_image = nib.load(input_image_path) # load in NIfTI image
image_data = provided_nifti_image.get_fdata() # outputs as 3D NumPy array

# Initiating visualization:
threshold = 0.5
highlighted_image = visualization(image_data, feature_maps, threshold)
# feature_maps was obtained earlier from classification

# SAVING VISUALIZATION AS NIFTI FILE
output_path = 'output file location' # specify file path to save visualization
save_nifti(highlighted_image, output_path, affine=provided_nifti_image.affine)
# using the affine attribute of the original image ensures that the spatial orientation and transformation aligns correctly with the original image
"""
