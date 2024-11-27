import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, Conv3DTranspose
from tensorflow.python.keras.models import Model
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LOADING IN DATA (FOR TRAINING)
# epoch - one complete pass through the ENTIRE training dataset during the training of the model
# batch: portion of an epoch
# during one epoch, the model will see each training model once and will update its parameters based on the observed data
class DataGenerator(Sequence): # defines custom class that inherits from Keras Sequence class. 
    def __init__(self, images, labels, batch_size=32, shuffle=True):
        if (len(images) != len(labels)):
            raise ValueError("Number of images and labels must be equal.")
        self.images = images 
        self.labels = labels 
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images)) 
        self.on_epoch_end()
        logging.info(f"DataGenerator initialized with {len(self.images)} images and {len(self.labels)} labels.")

    def __len__(self):
        # returns number of batches per epoch 
        num_batches = math.ceil(len(self.images) / self.batch_size)
        logging.info(f"Number of batches per epoch: {num_batches}")
        return num_batches

    def __getitem__(self, index):
        # generates one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # calculates indexes of the data samples to include in current batch 
        batch_images = [self.images[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]

        #preprocess batch of images
        preprocessed_images = [self.preprocess_image(img) for img in batch_images]
        preprocessed_labels = [self.preprocess_label(label) for label in batch_labels]
        # converts labels to binary masks for segmentation models using preprocess_labels function
        logging.info(f"Batch {index} loaded and preprocessed.")
        return np.array(preprocessed_images), np.array(preprocessed_labels) # returns batch of images and labels as numpy arrays

    def preprocess_label(self, label_data):
        """
        Preprocesses the label mask for segmentation.
        For segmentation models, labels are binarized. 

        Parameters:
        - label_data: np.ndarray, raw label mask.

        Returns:
        - binary_label: np.ndarray, binary mask.
        """
        # Binarize the label mask
        binary_label = (label_data > 0).astype(np.float64)
        logging.info("Completed preprocessing of label mask.")
        return binary_label
    
    def on_epoch_end(self):
        if self.shuffle == True:
            # if shuffle is true, NumPy will apply shuffle between epochs
            logging.info("Shuffling data.")
            np.random.shuffle(self.indexes)
            
    def preprocess_image(self, image_data):
        # check for division by zero
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        range_val = max_val - min_val

        if range_val == 0:
            error_message = "Preprocessing failed: Image has zero variance, leading to division by zero."
            logging.error(error_message)
            raise ValueError(error_message)
        
        # continue with normalization if valid
        image_data = (image_data - min_val) / range_val
        logging.info("Completed preprocessing/normalization.")
        return image_data

def segmentation(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv3D(32, kernel_size=(3,3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(64, kernel_size=(3,3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    # Decoder
    x = Conv3DTranspose(64, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(32, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Output Layer for Segmentation
    outputs = Conv3D(1, kernel_size=(1,1,1), activation='sigmoid', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Calculates the Dice Coefficient.

    Parameters:
    - y_true: tensor, ground truth masks.
    - y_pred: tensor, predicted masks.
    - smooth: float, smoothing factor to prevent division by zero.

    Returns:
    - dice: float, Dice Coefficient.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return dice

def visualize_segmentation(input_image, predicted_mask, threshold=0.5):
    """
    Overlays the predicted segmentation mask on the input image.
    
    Parameters:
    - input_image: np.ndarray, original MRI image.
    - predicted_mask: np.ndarray, segmentation mask from the model.
    - threshold: float, threshold to binarize the mask.
    
    Returns:
    - highlighted_image: np.ndarray, image with lesions highlighted.
    """
    mask = predicted_mask > threshold
    highlighted_image = np.copy(input_image)
    highlighted_image[mask] = 255  # Highlight lesions in white
    return highlighted_image

def save_nifti(image_data, output_path, affine):
    """
    Saves the highlighted NIfTI image and generates visualization slices.
    
    Parameters:
    - image_data: np.ndarray, image data to save (highlighted image).
    - output_path: str, file path where the NIfTI image will be saved.
    - affine: np.ndarray, affine transformation matrix for the image.
    """
    # Save the highlighted image as a NIfTI file
    nifti_img = nib.Nifti1Image(image_data, affine=affine)
    nib.save(nifti_img, output_path)
    logging.info(f"NIfTI image saved to {output_path}.")

    # Create a directory for visualizations if it doesn't exist
    viz_dir = os.path.splitext(output_path)[0] + '_viz'
    os.makedirs(viz_dir, exist_ok=True)
    logging.info(f"Visualization directory created at {viz_dir}.")

    # Determine the number of slices along each axis
    num_slices = image_data.shape[2]  # Assuming axis 2 is the sagittal plane

    # Define slice indices to visualize (e.g., middle slice)
    slice_indices = [num_slices // 4, num_slices // 2, (3 * num_slices) // 4]

    for idx in slice_indices:
        plt.figure(figsize=(6, 6))
        plt.imshow(image_data[:, :, idx], cmap='gray')
        plt.axis('off')
        plt.title(f'Slice {idx}')

        # Save the slice as a PNG image
        slice_path = os.path.join(viz_dir, f'slice_{idx}.png')
        plt.savefig(slice_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        logging.info(f"Visualization slice saved to {slice_path}.")
