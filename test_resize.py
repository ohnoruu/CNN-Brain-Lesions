import numpy as np 
import nibabel as nib
import torch
from torch.nn.functional import interpolate

from image_paths import image_paths

def resize_image(image, target_size):
    # convert numpy array to PyTorch tensor
    data_tensor = torch.tensor(image, dtype=torch.float32)
    
    if data_tensor.dim() == 3: # if image is (depth, height, width) / 3D (may apply to some labels)
        data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
    elif data_tensor.dim() == 4: # 4D image
        data_tensor = data_tensor.unsqueeze(0)

    # resize tensor 
    new_size = target_size[:3]
    resized_tensor = interpolate(data_tensor, size=new_size, mode='trilinear', align_corners=True)  

    # remove batch and channel dimensions
    resized_tensor = resized_tensor.squeeze(0).squeeze(0)

    # convert tensor back to numpy array
    resized_data = resized_tensor.numpy
    return resized_data

if __name__ == '__main__':
    nifti_file_path = image_paths[0] # sub-1 (exception)

    # Most common dimensions: 224, 224, 26,1 (TRACE files)

    nifti_img = nib.load(nifti_file_path)
    image_data = nifti_img.get_fdata()

    target_size = [224, 224, 26, 1]

    resized_image = resize_image(image_data, target_size)

    print(f'Original shape: {image_data.shape}')
    print(f'Resized shape: {resized_image.shape}')