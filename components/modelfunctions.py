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
        preprocessed_images = [self.preprocess_image(img) for img in batch_images]
        logging.info(f"Batch {index} loaded and preprocessed.")
        return np.array(preprocessed_images), np.array(batch_labels) # returns batch of images and labels as numpy arrays

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

def classification(input_shape):
    """
    Functions in tensorflow.keras
    Input - defines input layer
    Conv3D - creates 3D Convolutional layer
    Model - creates neural network, defines architecture by specifying input and output layers of the model
    """
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
    # define 3D ResNet model using Keras
    x = GlobalAveragePooling3D()(x)
    # reduces the dimensions of the data by taking avg of all elements in each feature map
    # data is reduced to a 1D vector --> passed to dense layer for classification
    outputs = Dense(1, activation='sigmoid')(x)
    # performs final classification using sigmoid function (value between 0 and 1)

    model = Model(inputs=inputs, outputs=outputs)
    # model creation
    # input - specifies input layer
    # output - specifies output layer
    return model # return keras model object

def visualization(input_image, model, threshold=0.5):
    """
    Apply thresholding to feature maps and visualize stroke areas in the MRI image.
    
    Parameters:
    - input_image: np.ndarray, input MRI image data
    - model: trained keras model weights
    - threshold: float, threshold to apply on feature maps to identify stroke areas
    
    Returns:
    - highlighted_image: np.ndarray, MRI image with highlighted stroke areas
    """
    feature_maps = model.predict(input_image[np.newaxis, ...]) # used to make compatible with model input shape that includes batch dimension

    stroke_map = feature_maps > threshold
    stroke_areas = ndi.binary_opening(stroke_map, structure=np.ones((3,3,3)))
    # maps out areas of stroke concern
    highlighted_image = np.copy(input_image)
    highlighted_image[stroke_areas] = 255 # highlight stroke areas with white color
    
    return highlighted_image # returns highlighted_image as a numpy array, save_nifti can be used to save the image

def save_nifti(image_data, output_path, affine): # this is only used to save the highlighted image to view in repository
    nifti_img = nib.Nifti1Image(image_data, affine=affine)
    nib.save(nifti_img, output_path)
