import os
import logging
import re
import nibabel as nib
import numpy as np
from pymongo import MongoClient
import gridfs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Configuration
from credentials import MONGO_URI
DB_NAME = 'lesion_dataset'

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

# Create GridFS instances for each collection for each collection location
fs_training_images = gridfs.GridFS(db, collection='training_images')
fs_labels = gridfs.GridFS(db, collection='labels')
fs_testing_images = gridfs.GridFS(db, collection='testing_images')

# MongoDB functions
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

def retrieve_nifti(filename, fs): # Searches for file by name and retrieves from MongoDB
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