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
from filenames import training_names, label_names, testing_names, testing_label_names
# import components
from components.mongofunctions import extract_subject_num, upload_nifti_files, retrieve_nifti
from components.modelfunctions import DataGenerator, classification, visualization, save_nifti
from credentials import MONGO_URI

class LesionModel:
    def __init__(self, db_name='lesion_dataset', batch_size=32, epochs=10, checkpoint_dir='checkpoints', output_dir='testview'):
        # MongoDB configuration
        self.mongo_uri = MONGO_URI
        self.db_name = db_name
        self.client = MongoClient(
            self.mongo_uri,
            connectTimeoutMS=999999999,
            socketTimeoutMS=999999999,   
            serverSelectionTimeoutMS=999999999,
            maxIdleTimeMS=999999999  
        )
        self.db = self.client[self.db_name]

        # GridFS instances for MongoDB data locations
        self.fs_training_images = gridfs.GridFS(self.db, collection='training_images')
        self.fs_labels = gridfs.GridFS(self.db, collection='labels')
        self.fs_testing_images = gridfs.GridFS(self.db, collection='testing_images')
        self.fs_testing_labels = gridfs.GridFS(self.db, collection='testing_labels')

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Model parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir  # where checkpoint is saved
        self.output_dir = output_dir  # where visualizations are saved
        self.training_data = []
        self.labels = []
        self.testing_data = []
        self.testing_labels = []
        self.train_generator = None
        self.test_generator = None
        self.model = None

    def load_data_in_batches(self, file_list, fs, is_label=False, batch_size=100):
        """
        Retrieve files from MongoDB in batches.
        """
        data = []
        for i in range(0, len(file_list), batch_size):
            batch = file_list[i:i + batch_size]  # Retrieve files in chunks
            for name in batch:
                data_item, _ = retrieve_nifti(name, fs)
                if data_item is not None:
                    data.append(data_item)
                    logging.info(f"Appended {name} to {'labels' if is_label else 'training/testing data'}.")

            logging.info(f"Processed batch {i//batch_size + 1}/{(len(file_list) + batch_size - 1)//batch_size}.")
        
        return data

    def load_training_data(self):
        logging.info("Loading training data in batches.")

        # Load training images in batches
        self.training_data = self.load_data_in_batches(training_names, self.fs_training_images, is_label=False)
        logging.info(f"Retrieved {len(self.training_data)} training images.")

        # Load labels in batches
        self.labels = self.load_data_in_batches(label_names, self.fs_labels, is_label=True)
        logging.info(f"Retrieved {len(self.labels)} training labels.")

        # Initialize DataGenerator
        self.train_generator = DataGenerator(self.training_data, self.labels, batch_size=self.batch_size, shuffle=True)
        logging.info(f"Training data loaded with {len(self.training_data)} images and {len(self.labels)} labels.")

    def load_testing_data(self):
        logging.info("Loading testing data in batches.")

        # Load testing images in batches
        self.testing_data = self.load_data_in_batches(testing_names, self.fs_testing_images, is_label=False)
        logging.info(f"Retrieved {len(self.testing_data)} testing images.")

        # Load testing labels in batches
        self.testing_labels = self.load_data_in_batches(testing_label_names, self.fs_testing_labels, is_label=True)
        logging.info(f"Retrieved {len(self.testing_labels)} testing labels.")

        # Initialize DataGenerator for testing data
        self.test_generator = DataGenerator(self.testing_data, self.testing_labels, batch_size=self.batch_size, shuffle=False)
        logging.info(f"Testing data loaded with {len(self.testing_data)} images and {len(self.testing_labels)} labels.")

    def build_model(self):
        # get shape reference to provide as a parameter to the classification function. all images in dataset are already resized to the same shape
        shape_reference, _ = retrieve_nifti(training_names[0], self.fs_training_images)
        # the shape of numpy arrays are the same as nifti, so no need to convert to nifti
        self.model = classification(shape_reference)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("Model built and compiled.")

    def train_model(self):
        # create checkpoint directory to save training weights
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model.h5')

        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy', # monitor accuracy
            mode='max', # refers to goal of maximizing accuracy
            save_best_only=True, # saves most accurate model
            save_weights_only=True,
            verbose=1 # specifies output (1 for print)
        )

        history = self.model.fit(
            self.train_generator, # training data returned by DataGenerator as NumPy arrays
            epochs=self.epochs,
            callbacks=[checkpoint_callback],
            validation_data=self.test_generator
        )

        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # -1 indicates last recorded accuracy
        logging.info(f"Training accuracy: {train_accuracy[-1]}")
        logging.info(f"Validation accuracy: {val_accuracy[-1]}")
        logging.info("Training weights saved and model training was completed.")

    def visualize_results(self, image_name, threshold=0.5): # used for visualizing single nifti images (not specific to the ones in the dataset)
        os.makedirs(self.output_dir, exist_ok=True) # ensure output directory exists
        self.model.load_weights(os.path.join(self.checkpoint_dir, 'model.h5'))

        image_data, affine = retrieve_nifti(image_name, self.fs_testing_images)

        highlighted_nifti = visualization(image_data, self.model, threshold)

        highlighted_nifti_img = nib.Nifti1Image(highlighted_nifti, affine)

        filename = os.path.basename(image_name)
        output_path = os.path.join(self.output_dir, filename) # saves to testview
        nib.save(highlighted_nifti_img, output_path)

        logging.info(f"Image visualization {filename} saved to {output_path}.")
    
    def visualize_all(self, testing_names, threshold=0.5): # used to retrieve all images in a MongoDB collection, most likely for testing process
        for name in testing_names:
            self.visualize_results(name, threshold)
        logging.info("All images visualized and saved.")

    def close_connection(self):
        self.client.close()
        logging.info("MongoDB connection closed.")

# Usage
if __name__ == '__main__':
    model = LesionModel()
    model.load_training_data()
    model.load_testing_data()
    model.build_model()
    model.train_model()

    #model.visualize_results('image_name', model.fs_testing_images) # can refer to filenames.py to get image_name
    
    #model.visualize_all(testing_names)

    """
    # Visualization/Testing
    input_image_path = 'path_to_nifti_image'
    provided_nifti_image = nib.load(input_image_path)
    input_image = provided_nifti_image.get_fdata()
    
    # Visualize using the modified function
    highlighted_image = visualization(input_image)
    
    # Save visualization if needed
    output_path = 'output_file_location'
    nib.save(nib.Nifti1Image(highlighted_image, provided_nifti_image.affine), output_path)
    """

"""
OLD CODE:
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
checkpoint_directory = 'checkpoints'
os.makedirs(checkpoint_directory, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_directory, 'model.h5')
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

# model.load_weights(checkpoint_path)


# VISUALIZATION / TESTING   
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