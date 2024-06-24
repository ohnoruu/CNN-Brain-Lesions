# imports
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense 
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os 
import nibabel as nib
import scipy.ndimage as ndi

# file paths (see image_paths.py, labels.py, and testing_paths.py)
from image_paths import image_paths
from labels import labels
from testing_paths import testing_paths

"""
DataGenerator - summary
__init__: initialized with parameters containing image paths, corresponding labels, batch size, and shuffle options 
__len__: returns # of batches per epoch
__getitem__: generates/retrieves one batch of data from given index
on_epoch__end: if shuffle is enabled, dataset indexes are shuffled after each epoch to prevent the model from memorizing the order of samples
load_nifti_image: loads nifti image and gets image as a data array
"""

# LOADING IN DATA (FOR TRAINING)
# epoch - one complete pass through the ENTIRE training dataset during the training of the model
# batch: portion of an epoch
# during one epoch, the model will see each training model once and will update its parameters based on the observed data
class DataGenerator(Sequence): # defines custom class that inherits from Keras Sequence class. 
    # Instead of loading the entire dataset to train the model, the generator loads and processes a batch of data
    # Feeds each batch to the model for training and moves on
    # Optimal for handling large amounts of data
    # 'self' refers to an instance of DataGenerator class.
    # functions beginning and ending with '__' are dunder methods and are automatically called when itializing an instance of DataGenerator
    def __init__(self, image_paths, labels, batch_size=32, shuffle=True):
        # initializes data generator with image paths, corresponding labels, batch size, and shuffle option
        self.image_paths = image_paths # stores list of file paths to images in the dataset
        self.labels = labels # stores corresponding labels for each image of the dataset (like key or id)
        self.batch_size = batch_size # specifies the size of each batch that will be sent to the model
        self.shuffle = shuffle # indicates whether shuffle should be applied between epochs 
        # important to prevent the model from memorizing the order of the samples, improving generalization
        self.indexes = np.arange(len(image_paths)) # array of indexes representing the positions of the samples in the dataset
        # NumPy's 'arange' function generates an array of indexes from 0 to the length of image paths 
        # each index corresponds to a specific MRI image in the dataset
        self.on_epoch_end() # called to shuffle the dataset each epoch

    def __len__(self):
        return len(self.image_paths) // self.batch_size 
        # returns number of batches per epoch 

    def __getitem__(self, index):
        # generates one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # calculates indexes of the data samples to include in current batch 
        batch_image_paths = [self.image_paths[i] for i in indexes]
        # retrieves the image paths corresponding to selected indexes 
        batch_labels = [self.labels[i] for i in indexes]
        # retrieves labels corresponding to selected indexes 

        #preprocess batch of images
        preprocessed_images = []
        for image_path in batch_image_paths:
            # code inside loop will process each image in a batch 
            image_path = self.load_nifti_image(image_path)
            # load image using load_nifti_image function, which retrieves image file and converts into array
            preprocessed_image = self.preprocess_image(image_path)
            # preprocess image using preprocess_image function, which changes the image format to fit 3D ResNet
            preprocessed_images.append(preprocessed_image)
            # appends preprocessed image to a list of preprocessed images (defined above)

        return np.array(preprocessed_images), np.array(batch_labels)
        # return tuple containing preprocessed images and corresponding batch labels. 
        # both are converted into NumPy arrays and are later used as inputs for training the model

    def on_epoch_end(self):
        if self.shuffle == True:
            # if shuffle is true, NumPy will apply shuffle between epochs
            np.random.shuffle(self.indexes)

    def load_nifti_image(self, image_path):
        nifti_img = nib.load(image_path)
        # load image
        image_data = nifti_img.get_fdata()
        # get image as data array (records dimensions of the 3D nifti image)
        return image_data
    
    def get_nifti_shape(self, image_path):
        nifti_img = nib.load(image_path)
        image_data = nifti_img.get_fdata()
        image_shape = image_data.shape
        print("Image dimensions: ", image_shape)
        return image_shape

    def preprocess_image(self, image):
        # normalization
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
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

# TRAINING 
# image_paths and labels were previously imported (see imports)
batch_size = 32 # num of samples in each batch
train_generator = DataGenerator(image_paths, labels, batch_size=batch_size, shuffle=True) # instance of DataGenerator class
#loads, preprocesses images and corresponding labels, shuffles after each epoch. 
shape_image_path = image_paths[0] # used SOLELY for retrieving the shape of image to provide as an input for training
image_shape = train_generator.get_nifti_shape(shape_image_path)
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