import requests
import os
import nibabel as nib

from image_paths import image_urls
from labels import label_urls

from image_paths import image_paths
from labels import labels

image_output_dir = 'data\MRI-DWI'
labels_output_dir = r'data\annotations'

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)

def download_file(url, output_dir):
    filename = os.path.join(output_dir, os.path.basename(url).replace(':', '_'))
    if not os.path.exists(filename):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {url} to {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}. Error: {e}")
    else:
        print(f"File {filename} already exists, skipping download.")

"""
# MRI Images
for url in image_urls:
    download_file(url, image_output_dir)

# annotations
for url in label_urls:
    download_file(url, labels_output_dir)

print("Upload complete.")
"""

# test file validity
test_image = image_paths[0]
test_label = labels[0]

nifti_img = nib.load(test_image)
img_data = nifti_img.get_fdata()
nifti_label = nib.load(test_label)
label_data = nifti_label.get_fdata()

print(f"Loaded {test_image} with shape {img_data.shape}")
print(f"Loaded {test_label} with shape {label_data.shape}") 
# Success! Loaded image shape successfully.