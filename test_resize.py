import numpy as np 
import nibabel as nib
import scipy.ndimage

from image_paths import image_paths

def resize_image(image, target_size):
    if image.shape != target_size:
        resized_data = scipy.ndimage.zoom(image, [
            target_size[0] / image.shape[0], # width
            target_size[1] / image.shape[1], # height
            target_size[2] / image.shape[2], # depth
            target_size[3] / image.shape[3] # channels
        ])
        new_image = nib.Nifti1Image(resized_data, image.affine)

if __name__ == '__main__':
    nifti_file_path = image_paths[0] # sub-1 (exception)
    # Most common dimensions: 224, 224, 26,1 (TRACE files)
    nifti_img = nib.load(nifti_file_path)
    image_data = nifti_img.get_fdata()
    target_size = [224, 224, 26, 1] # most common size
    resized_image = resize_image(image_data, target_size)
    print(f'Original shape: {image_data.shape}')
    print(f'Resized shape: {resized_image.shape}')
