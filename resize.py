import numpy as np 
import nibabel as nib
import scipy.ndimage
import os 

from image_paths import image_paths
from labels import labels
from testing_paths import testing_paths

def resize_image(image, target_size):
    if image.shape != target_size:
        if len(image.shape) == 3: # in case some labels are 3D files
            target_size = target_size[:3] # remove channel dimension
        zoom_factors = [t / s for t, s in zip(target_size, image.shape)]
        resized_data = scipy.ndimage.zoom(image, zoom_factors, order=3) # uses cubic interpolation
        return resized_data
    else:
        print("Image already has the target size (224, 224, 26, 1)")
        return image

"""
if __name__ == '__main__':
    nifti_file_path = image_paths[0] # sub-1 (exception)
    # Most common dimensions: 224, 224, 26,1 (TRACE files)
    nifti_img = nib.load(nifti_file_path)
    image_data = nifti_img.get_fdata()
    target_size = [224, 224, 26, 1] # most common size
    resized_image = resize_image(image_data, target_size)
    print(f'Original shape: {image_data.shape}')
    print(f'Resized shape: {resized_image.shape}')
# first test successful
"""
target_size = [224, 224, 26, 1]
# MRI training images
for image in image_paths:
    if image.shape != target_size: 
        nifti_img = nib.load(image)
        image_data = nifti_img.get_fdata()
        resized_image = resize_image(image_data, target_size)
        # convert back to numpy array 
