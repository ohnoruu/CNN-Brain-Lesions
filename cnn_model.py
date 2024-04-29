from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense 
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import numpy as np
import os 
import nibabel as nib

# loading in data
class DataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(image_paths))
        self.on_epoch_end()


    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_paths = [self.image_paths[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]

        #preprocess batch of images
        preprocessed_images = []
        for image_path in batch_image_paths:
            image_path = self.load_nifti_image(image_data)
            preprocessed_image = self.preprocess_image(image_data)
            preprocessed_images.append(preprocessed_image)

        return np.array(preprocessed_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_nifti_image(self, image_path):
        # load image
        nifti_img = nib.load(image_path)
        # get image data array
        image_data = nifti_img.get_fdata()
        return image_data

    def preprocess_image(self, image_data):
        # apply preprocessing steps
        depth, height, width, channels = image_data.shape
        # resize image to fit 3D ResNet model

        return input_shape


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