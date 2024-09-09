import os 
import nibabel as nib
from pymongo import MongoClient
import gridfs
import bson
from io import BytesIO
import io
import logging
import re 
import tempfile
import numpy as np
import hashlib

from filenames import training_names, label_names, testing_names

# Configuration
from credentials import MONGO_URI
DB_NAME = 'lesion_dataset'

image_dir = 'data\MRI-DWI'
labels_dir = r'data\annotations'
testing_dir = r'data\testing'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

# Create GridFS instances for each collection to store images into their respective destination
fs_training_images = gridfs.GridFS(db, collection='training_images')
fs_labels = gridfs.GridFS(db, collection='labels')
fs_testing_images = gridfs.GridFS(db, collection='testing_images')

def extract_subject_num(file):
    filename = os.path.basename(file)
    pattern = r'sub-(\d+)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        logging.error(f"Error retrieving subject number from {filename}")
        return None

def upload_nifti_files(directory, fs):
    for file in os.listdir(directory):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            file_path = os.path.join(directory, file)
            logging.info(f"Currently processing {file}")
            try:
                nifti_img=nib.load(file_path)
                # convert to numpy array
                image_data = nifti_img.get_fdata()
                # convert array to bytes
                binary_data = image_data.tobytes()
                # store binary to MongoDB
                subject_number = extract_subject_num(file) # get subject num to assign ids
                fs.put(binary_data, _id=subject_number, filename=file)
                logging.info(f"Uploaded {file} to MongoDB.")
            except Exception as e:
                logging.error(f"Error uploading {file} to MongoDB: {e}")

def retrieve_nifti(filename, fs):
    # Find filename
    grid_out = fs.find_one({"filename": filename})
    if grid_out:
        try:
            logging.info(f"Loading data for {filename}")
            # Read binary data
            binary_data = grid_out.read()

            # Check length of binary_data
            logging.info(f"Binary data length: {len(binary_data)}")

            # Convert bytes back to numpy array
            image_data = np.frombuffer(binary_data, dtype=np.float64)  # Check dtype
            logging.info(f"Image data length: {len(image_data)}")

            # Determine the shape based on the filename or some criteria
            if 'derivatives' in filename:  # for 3D formatted labels
                original_shape = (224, 224, 26)
            else:  # Assuming images are 4D
                original_shape = (224, 224, 26, 1)

            # Reshape the numpy array accordingly
            image_data = image_data.reshape(original_shape)

            # Create a NIfTI image from the numpy array
            nifti_img = nib.Nifti1Image(image_data, affine=np.eye(4))  # Use a default affine if needed
            logging.info(f"Loaded {filename} with shape {nifti_img.get_fdata().shape}")
            return nifti_img
        except Exception as e:
            logging.error(f"Error loading {filename} from MongoDB: {e}")
    else:
        logging.warning(f"File {filename} not found in MongoDB.")
        return None

"""
filename = 'derivatives_lesion_masks_sub-187_dwi_sub-187_space-TRACE_desc-lesion_mask.nii.gz'
testnum = extract_subject_num(filename)
print(testnum)
# successful
"""

"""
upload_nifti_files(image_dir, fs_training_images)
logging.info("MRI images (for testing use) uploaded.")
upload_nifti_files(labels_dir, fs_labels)   
logging.info("Annotations/labels uploaded.")
upload_nifti_files(testing_dir, fs_testing_images)
logging.info("Testing images uploaded.")
# upload was successful, move to testing
"""


for name in training_names:
    retrieve_nifti(name, fs_training_images)

for name in label_names:
    retrieve_nifti(name, fs_labels)

for name in testing_names:
    retrieve_nifti(name, fs_testing_images)
# all retrievals were successful


#retrieve_nifti('derivatives_lesion_masks_sub-1005_dwi_sub-1005_space-TRACE_desc-lesion_mask.nii.gz', fs_labels)

file = fs_labels.find_one({"filename": "derivatives_lesion_masks_sub-181_dwi_sub-181_space-TRACE_desc-lesion_mask.nii.gz"})
print(file)