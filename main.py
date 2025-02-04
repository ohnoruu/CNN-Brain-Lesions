# imports
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pymongo import MongoClient
import gridfs
import re
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import logging
import math
import matplotlib.pyplot as plt
import time

# import filenames to load from MongoDB
from filenames import training_names, label_names, testing_names, testing_label_names
# import components
from components.modelfunctions import DataGenerator, segmentation, dice_coefficient, visualize_segmentation, save_nifti
from components.mongofunctions import retrieve_nifti
from credentials import MONGO_URI

class LesionModel:
    def __init__(self, db_name='lesion_dataset', batch_size=32, epochs=20, checkpoint_dir='checkpoints', output_dir='testview'):
        # GPU configuration to prevent memory errors
        # Allows TF to allocate GPU memory as needed
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

        # MongoDB configuration
        self.mongo_uri = MONGO_URI
        self.db_name = db_name
        self.client = MongoClient()
        self.db = self.client[self.db_name]

         # GridFS instances for MongoDB data locations
        self.fs_training_images = gridfs.GridFS(self.db, collection='training_images')
        self.fs_labels = gridfs.GridFS(self.db, collection='labels')
        self.fs_testing_images = gridfs.GridFS(self.db, collection='testing_images')
        self.fs_testing_labels = gridfs.GridFS(self.db, collection='testing_labels')
        
        # Local File directories
        self.train_img_dir = r'tempdata\images'
        self.train_labels_dir = r'tempdata\labels'
        self.test_img_dir = r'tempdata\testingimages'
        self.test_labels_dir = r'tempdata\testinglabels'

        # Logging configuration
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Model parameters
        self.batch_size = batch_size # batch_size is used for both loading the training data and 
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir  # where checkpoint is saved
        self.output_dir = output_dir  # where visualizations are saved
        # for loading data
        self.training_images = []
        self.training_labels = []
        self.testing_images = []
        self.testing_labels = []
        # for model creation
        self.train_generator = None # stores instance of DataGenerator class for training data (80%)
        self.test_generator = None # stores instance of DataGenerator class for testing data (20%)
        self.model = None # stores instance of segmentation model (where model is constructed)

    # LOADING DATA
    # Since MongoDB requires paid plan for large data storage, local repository files are retrieved.
    # Retrieval for smaller instances (ex: retrieving shape for segmentation and visualization) will still be done through MongoDB using the retrieve_nifti function in mongofunctions.py

    def load_training_data(self):
        logging.info("Loading training data.")
        self.train_generator = DataGenerator(
            self.train_img_dir, 
            self.train_labels_dir, 
            batch_size=self.batch_size, 
            shuffle=True,
            augment=False # 1/30/25 temporarily set to False for testing purposes
        )
        logging.info("DataGenerator instance created with training data.")

    def load_testing_data(self):
        logging.info("Loading testing data.")
        self.test_generator = DataGenerator(
            self.test_img_dir, 
            self.test_labels_dir, 
            batch_size=self.batch_size, 
            shuffle=False,
            augment=False
        )
        logging.info("DataGenerator instance created with testing data.")
    
    def build_model(self):
        sample_image_path = os.path.join(self.train_img_dir, training_names[0])
        shape_reference = nib.load(sample_image_path).get_fdata().shape
        self.model = segmentation(shape_reference)

        self.model.compile(
            optimizer='adam', # adaptive learning rate optimization algorithm combining AdaGrad and RMSProp
            loss='binary_crossentropy',  # returns image mask (ex: 1 for lesion, 0 for non-lesion). Good since labels are binary masks
            metrics=['accuracy', dice_coefficient] # dice coefficient is used to evaluate segmentation models
        )
        logging.info("Model built and compiled.")

    def train_model(self):
        # create checkpoint directory to save training weights
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model.weights.h5')

        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy', # monitor accuracy
            mode='max', # refers to goal of maximizing accuracy
            save_best_only=True, # saves most accurate model
            save_weights_only=True,
            verbose=1 # specifies output (1 for print, 0 for silent)
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=3, # number of epochs with no improvement after which training will be stopped
            restore_best_weights=True,
            verbose=1 # progress bar mode (output setting)
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2, # factor by which the learning rate will be reduced
            patience=2, # number of epochs with no improvement after which learning rate will be reduced
            min_lr=1e-6,
            verbose=1 # progress bar mode (output setting)
        )

        history = self.model.fit(
            self.train_generator, # training data returned by DataGenerator as NumPy arrays
            epochs=self.epochs,
            callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler],
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
        self.model.load_weights(os.path.join(self.checkpoint_dir, 'model.weights.h5'))

        image_data, affine = retrieve_nifti(image_name, self.fs_testing_images)

        highlighted_nifti = visualize_segmentation(image_data, self.model, threshold)

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
    try:
        model = LesionModel()
        model.load_training_data()
        model.load_testing_data()
        model.build_model()
        model.train_model()

        #model.visualize_results('image_name', model.fs_testing_images) # can refer to filenames.py to get image_name
        #model.visualize_all(testing_names)

    except Exception as e:
        logging.critical(f"Application halted due to an error: {e}")
        exit(1)

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