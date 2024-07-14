import gzip
import shutil
import os 
import zipfile
import nibabel as nib

from image_paths import image_paths

def extract_gz(file_path):
    output_path = file_path.replace('.gz', '')
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file_path)

for path in image_paths:
    if path.endswith('.gz'):
        extract_gz(path)
        new_path = path.replace('.gz', '')
    else:
        new_path=path

    try:
        nifti_img = nib.load(new_path)
        image_data = nifti_img.get_fdata()
        print(f'Loaded {new_path} with shape {image_data.shape}')
    except Exception as e:
        print(f'Error loading {new_path}: {e}')