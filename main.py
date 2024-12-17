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
import math

# import filenames to load from MongoDB
from filenames import training_names, label_names, testing_names, testing_label_names
# import components
from components.modelfunctions import DataGenerator, segmentation, dice_coefficient, visualize_segmentation, save_nifti
from components.mongofunctions import retrieve_nifti
from credentials import MONGO_URI

class LesionModel:
    def __init__(self, db_name='lesion_dataset', batch_size=32, epochs=10, checkpoint_dir='checkpoints', output_dir='testview'):
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
        self.batch_size = batch_size
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

    def extract_identifier(self, filename, file_type='image'):
        if file_type == 'image':
             # Example: 'sub-1_dwi_sub-1_rec-TRACE_dwi.nii.gz'
            parts = filename.split('_space-')[0].split('_rec-')[0]
        elif file_type == 'label':
            # Example: 'derivatives_lesion_masks_sub-1_dwi_sub-1_space-TRACE_desc-lesion_mask.nii.gz'
            parts = filename.split('_space-')[0].replace('derivatives_lesion_masks_', '')
        else:
            parts = filename.split('_')[0]
        return parts
    
    # LOADING DATA
    # Since MongoDB requires paid plan for large data storage, local repository files are retrieved.
    # Retrieval for smaller instances (ex: retrieving shape for segmentation and visualization) will still be done through MongoDB using the retrieve_nifti function in mongofunctions.py
    def load_training_data(self): # training img data
        image_files = [f for f in os.listdir(self.train_img_dir)]
        label_files = [f for f in os.listdir(self.train_labels_dir)]

        if len(image_files) != len(label_files): # check if number of images and labels match, else stop operation
            logging.error("Number of images and labels do not match. Verify using findmismatch.py.")
            return

        # Create a dictionary for quick lookup of labels
        label_dict = {self.extract_identifier(f, file_type='label'): f for f in label_files}

        # Load and append data based on shared identifier
        for img_file in image_files:
            identifier =  self.extract_identifier(img_file, file_type='image')
            label_file = label_dict.get(identifier)

            if label_file:
                img_path = os.path.join(self.train_img_dir, img_file)
                label_path = os.path.join(self.train_labels_dir, label_file)
                
                # Load NiFTi files
                img_nii = nib.load(img_path)
                label_nii = nib.load(label_path)
                
                img_data = img_nii.get_fdata()
                label_data = label_nii.get_fdata()
                
                self.training_images.append(img_data)
                self.training_labels.append(label_data)
                
                logging.info(f"Loaded {img_file} and {label_file}.")
            else:
                logging.warning(f"No label found for image {img_file}.")
                return

        # training instance of DataGenerator class
        self.train_generator = DataGenerator(
            self.training_images, 
            self.training_labels, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        logging.info("DataGenerator instance created with training data.")

    def load_testing_data(self): # testign data used for evaluating accuracy
        image_files = [f for f in os.listdir(self.test_img_dir)]
        label_files = [f for f in os.listdir(self.test_labels_dir)]

        if len(image_files) != len(label_files): # check if number of images and labels match, else stop operation
            logging.error("Number of images and labels do not match. Verify using findmismatch.py.")
            return

        # Create a dictionary for quick lookup of labels
        label_dict = {self.extract_identifier(f, file_type='label'): f for f in label_files}

        # Load and append data based on shared identifier
        for img_file in image_files:
            identifier =  self.extract_identifier(img_file, file_type='image')
            label_file = label_dict.get(identifier)

            if label_file:
                img_path = os.path.join(self.test_img_dir, img_file)
                label_path = os.path.join(self.test_labels_dir, label_file)
                
                # Load NiFTi files
                img_nii = nib.load(img_path)
                label_nii = nib.load(label_path)
                
                img_data = img_nii.get_fdata()
                label_data = label_nii.get_fdata()
                
                self.testing_images.append(img_data)
                self.testing_labels.append(label_data)
                
                logging.info(f"Loaded {img_file} and {label_file}.")
            else:
                logging.warning(f"No label found for image {img_file}.")
                return
            
        # testing instance of DataGenerator class
        self.test_generator = DataGenerator(
            self.testing_images, 
            self.testing_labels, 
            batch_size=self.batch_size, 
            shuffle=False # no shuffle needed for testing
        )
        logging.info("DataGenerator instance created with testing data.")
    
    def build_model(self):
        # get shape reference to provide as a parameter to the segmentation function. all images in dataset are already resized to the same shape
        shape_reference, _ = retrieve_nifti(training_names[0], self.fs_training_images)
    
        # Check and append channel dimension if missing
        if len(shape_reference) == 3:
            input_shape = shape_reference + (1,)  # Add channel dimension for grayscale
        elif len(shape_reference) == 4:
            input_shape = shape_reference  # Assume channels are already included
        else:
            error_message = f"Unsupported input shape: {shape_reference}"
            logging.error(error_message)
            raise ValueError(error_message)
        
        # the shape of numpy arrays are the same as nifti, so no need to convert to nifti
        self.model = segmentation(input_shape)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy', 
            metrics=['accuracy', dice_coefficient] # dice coefficient is used to evaluate segmentation models
        )
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
            verbose=1 # specifies output (1 for print, 0 for silent)
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