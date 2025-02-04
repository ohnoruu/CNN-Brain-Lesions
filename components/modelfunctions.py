import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, Conv3DTranspose, Cropping3D, ZeroPadding3D, Concatenate, Dropout
from tensorflow.keras.models import Model
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()]) # logging to console

# LOADING IN DATA (FOR TRAINING)
# epoch - one complete pass through the ENTIRE training dataset during the training of the model
# batch: portion of an epoch
# during one epoch, the model will see each training model once and will update its parameters based on the observed data
class DataGenerator(Sequence):
    def __init__(self, image_dir, label_dir, batch_size=32, shuffle=True, rotation_range=10, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rotation_range = rotation_range # range of rotation for data augmentation. 20 is a safe range to prevent distortion
        self.augment = augment
        self.images = [f for f in os.listdir(image_dir)]
        self.labels = [f for f in os.listdir(label_dir)]
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()
        logging.info(f"DataGenerator initialized with {len(self.images)} images and {len(self.labels)} labels.")

    def __len__(self):
        num_batches = math.ceil(len(self.images) / self.batch_size)
        #logging.info(f"Number of batches per epoch: {num_batches}")
        return num_batches

    def __getitem__(self, index):
        # generating batches
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.images[i] for i in indexes]
        batch_labels = [self._get_label_file(img) for img in batch_images]

        # preprocessing (image normalization, label binarization)
        preprocessed_images = [self.preprocess_image(nib.load(os.path.join(self.image_dir, img)).get_fdata()) for img in batch_images]
        preprocessed_labels = [self.preprocess_label(nib.load(os.path.join(self.label_dir, lbl)).get_fdata()) for lbl in batch_labels]

        if self.augment:
        # data augmentation: random rotation
            augmented_pairs = [self.random_rotate(img, lbl) for img, lbl in zip(preprocessed_images, preprocessed_labels)]
            preprocessed_images.extend(img for img, lbl in augmented_pairs)
            preprocessed_labels.extend(lbl for img, lbl in augmented_pairs)

        #logging.info(f"Batch {index} loaded and preprocessed.")
        return np.array(preprocessed_images), np.array(preprocessed_labels)

    def _get_label_file(self, image_file):
        identifier = self.extract_identifier(image_file, file_type='image')
        logging.debug(f"Extracted identifier for image file {image_file}: {identifier}")
        for label_file in self.labels:
            if identifier in self.extract_identifier(label_file, file_type='label'):
                #logging.info(f"Label file {label_file} found for {image_file}.")
                return label_file
        error_message = f"Label file not found for {image_file}."
        logging.error(error_message)
        raise ValueError(error_message)

    def extract_identifier(self, filename, file_type='image'):
        if file_type == 'image':
            parts = filename.split('_dwi_')[0]  # Extract up to '_dwi_'
        elif file_type == 'label':
            parts = filename.split('_space-')[0].replace('derivatives_lesion_masks_', '')  # Extract up to '_space-' and remove prefix
        else:
            parts = filename.split('_')[0]
        logging.debug(f"Extracted parts for {file_type} file {filename}: {parts}")
        return parts

    def preprocess_label(self, label_data):
        label_data = np.resize(label_data, (224, 224, 26))
        binary_label = (label_data > 0).astype(np.float32)
        #logging.info("Completed preprocessing of label mask.")
        return binary_label

    def preprocess_image(self, image_data):
        image_data = np.resize(image_data, (224, 224, 26, 1))
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        range_val = max_val - min_val

        if range_val == 0:
            error_message = "Preprocessing failed: Image has zero variance, leading to division by zero."
            logging.error(error_message)
            raise ValueError(error_message)

        normalized_image = (image_data - min_val) / range_val
        #logging.info("Completed preprocessing/normalization.")
        return normalized_image
    
    def random_rotate(self, image, label):
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        # Rotate 4D MRI images slice-by-slice
        if image.ndim == 4:  
            rotated_image = np.stack(
                [ndi.rotate(image[:, :, i, :], angle, axes=(0,1), order=1, reshape=False, mode='nearest') 
                for i in range(image.shape[2])], axis=2)
        else:
            rotated_image = ndi.rotate(image, angle, axes=(0,1), order=1, reshape=False, mode='nearest')
        
        # Rotate 3D label masks as a whole volume
        rotated_label = ndi.rotate(label, angle, axes=(0,1), order=0, reshape=False, mode='nearest')

        return rotated_image, rotated_label

    def on_epoch_end(self):
        if self.shuffle:
            logging.info("Shuffling data.")
            np.random.shuffle(self.indexes)

def match_depth(tensor1, tensor2):
    """Ensures tensor1 and tensor2 have the same depth dimension (third spatial axis)."""
    depth1, depth2 = int(tensor1.shape[3]), int(tensor2.shape[3])
    depth_diff = abs(depth1 - depth2)

    if depth1 > depth2:  # tensor1 is deeper → pad tensor2
        pad_front = depth_diff // 2
        pad_back = depth_diff - pad_front
        tensor2 = ZeroPadding3D(padding=(0, 0, (pad_front, pad_back)))(tensor2)
    elif depth1 < depth2:  # tensor2 is deeper → pad tensor1
        pad_front = depth_diff // 2
        pad_back = depth_diff - pad_front
        tensor1 = ZeroPadding3D(padding=(0, 0, (pad_front, pad_back)))(tensor1)

    return tensor1, tensor2  # Return both tensors to ensure correct shape

def segmentation(input_shape):
    inputs = Input(shape=input_shape)

    # add Dropout after activation layers and before pooling layers to prevent overfitting
    # Current dropout range of 30% to 40%, although can be adjusted.

    # Encoder
    x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.3)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)  # Reduces spatial dimensions by half

    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.3)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)  # Reduces spatial dimensions by half again

    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.4)(x) # add higher dropout rate as complexity increases

    # Decoder
    x = Conv3DTranspose(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)  # Doubles spatial dimensions
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.3)(x)

    x = Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)  # Doubles spatial dimensions again
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.3)(x)

    # Matching depth dimension
    x = ZeroPadding3D(padding=((0, 0), (0, 0), (0, 2)))(x) # increases depth dimension

    # Ensure the output has the same number of channels as the target
    outputs = Conv3D(1, kernel_size=(1, 1, 1), activation='sigmoid', padding='same')(x)  # Single channel output

    model = Model(inputs=inputs, outputs=outputs)
    return model

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Calculates the Dice Coefficient.

    evaluates similarity betweeen model predictions and ground truth (labels)
    1. intersection: number of overlapping pixels between the predicted mask and the ground truth mask
    2. union: total number of pixels in both masks
    3. dice coefficient: 2 * intersection / union
    
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
