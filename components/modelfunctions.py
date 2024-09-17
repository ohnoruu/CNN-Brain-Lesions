import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense
from tensorflow.keras.models import Model
import scipy.ndimage as ndi
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LOADING IN DATA (FOR TRAINING)
# epoch - one complete pass through the ENTIRE training dataset during the training of the model
# batch: portion of an epoch
# during one epoch, the model will see each training model once and will update its parameters based on the observed data
class DataGenerator(Sequence): # defines custom class that inherits from Keras Sequence class. 
    def __init__(self, images, labels, batch_size=32, shuffle=True):
        self.images = images 
        self.labels = labels 
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        self.indexes = np.arange(len(self.images)) 
        self.on_epoch_end()
        logging.info(f"DataGenerator initialized with {len(self.images)} images and {len(self.labels)} labels.")

    def __len__(self):
        # returns number of batches per epoch 
        logging.info(f"Number of batches per epoch: {len(self.images) // self.batch_size}")
        return len(self.images) // self.batch_size 

    def __getitem__(self, index):
        # generates one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # calculates indexes of the data samples to include in current batch 
        batch_images = [self.images[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]

        #preprocess batch of images
        preprocessed_images = [self.preprocess_image(img.get_fdata()) for img in batch_images]
        logging.info(f"Batch {index} loaded and preprocessed.")
        return np.array(preprocessed_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle == True:
            # if shuffle is true, NumPy will apply shuffle between epochs
            logging.info("Shuffling data.")
            np.random.shuffle(self.indexes)
            
    def preprocess_image(self, image_data):
        # normalization
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        logging.info("Completed preprocessing/normalization.")
        return image_data
"""
Functions in tensorflow.keras
Input - defines input layer
Conv3D - creates 3D Convolutional layer
Model - creates neural network, defines architecture by specifying input and output layers of the model
"""

def classification(input_shape):
    # define 3D ResNet model using Keras
    inputs = Input(shape=input_shape)
    # creates an input layer for the model with the specified input shape
    # defines depth, height, width, and channels (colors)
    
    # for each convolutional layer, each step will be applied using var 'x'
    x = Conv3D(32, kernel_size=(3,3,3), padding='same')(inputs)
    # creates a 3D Convolutional Layer on the input data
    # 32 - number of filters (output channels) 
    # kernel-size=(3,3,3) - defines the size of a 3D convolutional kernel: a small matrix that is convolved with an input image, used to extract features
    # activation = 'sigmoid' - specifies activation function of the layer. sigmoid maps input values from a range between 1 and 0 (used for classification)
    # padding = 'same' - pads the input such as the output has the same dimensions as the input
    x = BatchNormalization()(x)
    # normalizes the activations of previous layer
    x = Activation('relu')(x)
    # applies non-linear activation function
    # helps in learning complex patterns in the dataset
    x = MaxPooling3D(pool_size=(2,2,2))(x)
    # reduces spatial dimensions of data, with a maximum pool size of 2x2x2
    # helps to downsample the data and reduce computational load

    x = Conv3D(64, kernel_size=(3,3,3), padding='same')(x)
    # performs another 3D convolution, now with 64 filters alongside with normalization and relu activation
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    feature_maps = x
    # extract features from the data to use for visualization task

    x = GlobalAveragePooling3D()(x)
    # reduces the dimensions of the data by taking avg of all elements in each feature map
    # data is reduced to a 1D vector --> passed to dense layer for classification
    outputs = Dense(1, activation='sigmoid')(x)
    # performs final classification using sigmoid function (value between 0 and 1)

    model = Model(inputs=inputs, outputs=outputs)
    # model creation
    # input - specifies input layer
    # output - specifies output layer
    return model, feature_maps

def visualization(image_data, feature_maps, threshold):
    """
    Apply thresholding to feature maps and visualize stroke areas in the MRI image.
    
    Parameters:
    - image_data: np.ndarray, original MRI image data
    - feature_maps: np.ndarray, feature maps from the model
    - threshold: float, threshold to apply on feature maps to identify stroke areas
    
    Returns:
    - highlighted_image: np.ndarray, MRI image with highlighted stroke areas
    """
    # convert feature_maps to numpy array if it is a TensorFlow tensor
    if isinstance(feature_maps, tf.Tensor):
        logging.info(f"Provided data {feature_maps} is a TensorFlow tensor. Converting to numpy array.")
        feature_maps = feature_maps.numpy()
    # apply threshold to feature maps to identify stroke areas
    stroke_map = feature_maps > threshold
    # identifies areas of stroke concern by highlighting parts of the brain that indicate a stroke probability higher than the threshold (50%)
    
    stroke_areas = ndi.binary_opening(stroke_map, structure=np.ones((3,3,3)))
    # binary opening - removes small white regions from the image that are surrounded by black pixels
    # used to seperate objects close to each other to visualize stroke areas
    # maps out areas of stroke concern
    highlighted_image = np.copy(image_data)
    highlighted_image[stroke_map] = 255 # highlight stroke areas with white color

    return highlighted_image

def save_nifti(image_data, output_path, affine=None):
    """
    Save the given image data as a NIfTI file. (this is mainly meant for after visualization)
    
    Parameters:
    - image_data: np.ndarray, image data to be saved
    - output_path: str, path to save the NIfTI file
    - affine: np.ndarray, affine transformation matrix for the NIfTI file, defines spatial orientation of the image within the 3D space
    """

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(image_data, affine=affine)
    # Save image
    nib.save(nifti_img, output_path)