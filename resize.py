import numpy as np 
import nibabel as nib
import skimage.transform as skTrans
import os 

from image_paths import image_paths
from labels import labels
from testing_paths import testing_paths

image_output_dir = 'data\MRI-DWI'
labels_output_dir = r'data\annotations'
testing_output_dir = r'data\testing'

def resize_image(image, target_size):
    if image.shape != target_size:
        # cubic interpolation, preserve intensity range
        resized_image = skTrans.resize(image, target_size, order=3, preserve_range=True)
        return resized_image
    else:
        print("Image already has the target size (224, 224, 26, 1)")
        return image
    
target_size = (224, 224, 26, 1)
# SIZE FOR LABELS SHOULD BE (224, 224, 26) WITHOUT 4th DIMENSION
target_label_size = (224, 224, 26)

# MRI training images
for path in image_paths:
    nifti_img = nib.load(path)
    image_data = nifti_img.get_fdata()
    if image_data.shape != target_size:
        try:
            resized_image = resize_image(image_data, target_size)
            print(f"Resized {path} with dimensions {image_data.shape} to {resized_image.shape}")
            resized_nifti = nib.Nifti1Image(resized_image, nifti_img.affine)
            output_path = os.path.join(image_output_dir, os.path.basename(path))
            nib.save(resized_nifti, output_path)
            print(f"Saved {path} to {output_path}")
        except Exception as e:
            print(f"Error resizing {path}: {e}")
    else: 
        print(f"{path} already meets target size {target_size}, skipping resizing process.")

# labels
for path in labels:
    nifti_img = nib.load(path)
    image_data = nifti_img.get_fdata()
    if image_data.shape != target_label_size:
        try:
            resized_image = resize_image(image_data, target_label_size)
            print(f"Resized {path} with dimensions {image_data.shape} to {resized_image.shape}")
            resized_nifti = nib.Nifti1Image(resized_image, nifti_img.affine)
            output_path = os.path.join(labels_output_dir, os.path.basename(path))
            nib.save(resized_nifti, output_path)
            print(f"Saved {path} to {output_path}")
        except Exception as e:
            print(f"Error resizing {path}: {e}")
    else: 
        print(f"{path} already meets target size {target_label_size}, skipping resizing process.")

# labels
for path in testing_paths:
    nifti_img = nib.load(path)
    image_data = nifti_img.get_fdata()
    if image_data.shape != target_size:
        try:
            resized_image = resize_image(image_data, target_size)
            print(f"Resized {path} with dimensions {image_data.shape} to {resized_image.shape}")
            resized_nifti = nib.Nifti1Image(resized_image, nifti_img.affine)
            output_path = os.path.join(testing_output_dir, os.path.basename(path))
            nib.save(resized_nifti, output_path)
            print(f"Saved {path} to {output_path}")
        except Exception as e:  
            print(f"Error resizing {path}: {e}")
    else: 
        print(f"{path} already meets target size {target_size}, skipping resizing process.")

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