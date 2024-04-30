# imports
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense 
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import numpy as np
import os 
import nibabel as nib

# loading in data
# epoch - one complete pass through the ENTIRE training dataset during the training of the model
# batch: portion of an epoch
# during one epoch, the model will see each training model once and will update its parameters based on the observed data
class DataGenerator(Sequence): # defines custom class that inherits from Keras Sequence class. 
    # Instead of loading the entire dataset to train the model, the generator loads and processes a batch of data
    # Feeds each batch to the model for training and moves on
    # Optimal for handling large amounts of data
    def __init__(self, image_paths, labels, batch_size=32, shuffle=True):
        # initializes data generator with image paths, corresponding labels, batch size, and shuffle option
        # 'self' refers to an instance of DataGenerator class. 
        self.image_paths = image_paths # stores list of file paths to images in the dataset
        self.labels = labels # stores corresponding labels for each image of the dataset (like key or id)
        self.batch_size = batch_size # specifies the size of each batch that will be sent to the model
        self.shuffle = shuffle # indicates whether shuffle should be applied between epochs 
        # important to prevent the model from memorizing the order of the samples, improving generalization
        self.indexes = np.arange(len(image_paths)) # array of indexes representing the positions of the samples in the dataset
        self.on_epoch_end() # called to shuffle the dataset each epoch

    def __len__(self):
        # returns number of batches per epoch 
        return len(self.image_paths) // self.batch_size 

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
            image_path = self.load_nifti_image(image_data)
            # load image using load_nifti_image function, which retrieves image file and converts into array
            preprocessed_image = self.preprocess_image(image_data)
            # preprocess image using preprocess_image function, which changes the image format to fit 3D ResNet
            preprocessed_images.append(preprocessed_image)
            # appends preprocessed image to a list of preprocessed images (defined above)

        return np.array(preprocessed_images), np.array(batch_labels)
        # return tuple containing preprocessed images and corresponding batch labels. 
        # both are converted into NumPy arrays

    def on_epoch_end(self):
        if self.shuffle == True:
            # if shuffle is true, NumPy will apply shuffle between epochs
            np.random.shuffle(self.indexes)

    def load_nifti_image(self, image_path):
        # load image
        nifti_img = nib.load(image_path)
        # get image data array
        image_data = nifti_img.get_fdata()
        return image_data

    def preprocess_image(self, image_data):
        # apply preprocessing steps later
        return processed_image


"""
# image preprocessing 
import nibabel as nib

# load image 
nifti_img = nib.load('path/to/nifti_file.nii')
# get image data array (reminder: NIfTI images are 4D arrays)
image_data = nifti_img.get_fdata()
# get dimensions of image data array
# "channels" refers to number of color channels (ex: RGB)
# MRI is typically grayscale, so channels=1
depth, height, width, channels = image_data.shape 
# .shape returns a tuple of the dimensions of the array, so this line of code will unpack it
print(f"Dimensions: Depth={depth}, Height={height}, Width={width}")
"""

# define 3D ResNet model
def resnet_3d(input_shape):
    inputs = Input(shape=input_shape)

    outputs = Conv3D(1, kernel_size=(3,3,3), activation='sigmoid', padding='same')(inputs)

    #create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# define input shape
#input_shape = (depth, height, width, channels) #define dimensions of input img

#create 3D ResNet model
model = resnet_3d(input_shape)
