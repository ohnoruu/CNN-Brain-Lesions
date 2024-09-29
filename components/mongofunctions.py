import os
import logging
import re
import nibabel as nib
import numpy as np
from pymongo import MongoClient
import gridfs

from filenames import training_names, label_names, testing_names
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
                nifti_img = nib.load(file_path)
                # Convert to numpy array
                image_data = nifti_img.get_fdata()
                # Convert array to bytes
                binary_data = image_data.tobytes()
                # Extract affine matrix
                affine_matrix = nifti_img.affine.tolist()  # Convert to list for JSON serialization
                
                # Store binary data and affine matrix to MongoDB
                subject_number = extract_subject_num(file)  # Assign subject number as ID
                fs.put(binary_data, _id=subject_number, filename=file, metadata={'affine': affine_matrix})
                logging.info(f"Uploaded {file} to MongoDB with affine matrix.")
            except Exception as e:
                logging.error(f"Error uploading {file} to MongoDB: {e}")

def upload_single_nifti_file(file_path, fs):
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        logging.info(f"Currently processing {file_path}")
        try:
            nifti_img = nib.load(file_path)
            # Convert to numpy array
            image_data = nifti_img.get_fdata()
            # Convert array to bytes
            binary_data = image_data.tobytes()
            # Extract affine matrix
            affine_matrix = nifti_img.affine.tolist()  # Convert to list for JSON serialization
            
            # Store binary data and affine matrix to MongoDB
            file_name = os.path.basename(file_path)
            subject_number = extract_subject_num(file_name)  # Assign subject number as ID
            fs.put(binary_data, _id=subject_number, filename=file_name, metadata={'affine': affine_matrix})
            logging.info(f"Uploaded {file_name} to MongoDB with affine matrix.")
        except Exception as e:
            logging.error(f"Error uploading {file_name} to MongoDB: {e}")
    else:
        logging.error(f"File {file_path} is not a NiFTi file.")

def retrieve_nifti(filename, fs):
    # Find filename
    grid_out = fs.find_one({"filename": filename})
    if grid_out:
        try:
            logging.info(f"Loading data for {filename}")
            # Read binary data
            binary_data = grid_out.read()
            
            # Check length of binary data
            logging.info(f"Binary data length: {len(binary_data)}")

            # Convert bytes back to numpy array
            image_data = np.frombuffer(binary_data, dtype=np.float64)
            logging.info(f"Image data length: {len(image_data)}")

            # Determine the shape based on the filename or some criteria
            if 'derivatives' in filename:  # for 3D formatted labels
                original_shape = (224, 224, 26)
            else:  # Assuming images are 4D
                original_shape = (224, 224, 26, 1)

            # Reshape the numpy array
            image_data = image_data.reshape(original_shape)

            # Retrieve the affine matrix from metadata
            affine_matrix = np.array(grid_out.metadata['affine'])  # Convert list back to NumPy array
            logging.info(f"Affine matrix loaded for {filename}")

            # Create a NIfTI image from the numpy array and the affine matrix
            nifti_img = nib.Nifti1Image(image_data, affine=affine_matrix)
            logging.info(f"Loaded {filename} with shape {nifti_img.get_fdata().shape} and affine matrix.")
            return nifti_img
        except Exception as e:
            logging.error(f"Error loading {filename} from MongoDB: {e}")
    else:
        logging.warning(f"File {filename} not found in MongoDB.")
        return None

# Retrieve and save NiFTi images
def save_retrieved_nifti(filename, fs, output_dir):
    nifti_img = retrieve_nifti(filename, fs)
    if nifti_img:
        output_path = os.path.join(output_dir, filename)
        nib.save(nifti_img, output_path)
        logging.info(f"Saved {filename} to {output_path}")
        return output_path
    return None

#upload_single_nifti_file('tempdata\images\sub-1_dwi_sub-1_rec-TRACE_dwi.nii.gz', fs_training_images)
#retrieve_nifti('sub-1_dwi_sub-1_rec-TRACE_dwi.nii.gz', fs_training_images)
#save_retrieved_nifti('sub-1_dwi_sub-1_rec-TRACE_dwi.nii.gz', fs_training_images, 'testview')
#upload_single_nifti_file('tempdata\labels\derivatives_lesion_masks_sub-2_dwi_sub-2_space-TRACE_desc-lesion_mask.nii.gz', fs_training_images)
#retrieve_nifti('derivatives_lesion_masks_sub-2_dwi_sub-2_space-TRACE_desc-lesion_mask.nii.gz', fs_training_images)
#save_retrieved_nifti('derivatives_lesion_masks_sub-2_dwi_sub-2_space-TRACE_desc-lesion_mask.nii.gz', fs_training_images, 'testview')
#retrieve_nifti('sub-1_dwi_sub-1_rec-TRACE_dwi.nii.gz', fs_training_images)

#TESTS WERE SUCCESSFUL, UPLOAD AND RETRIEVAL WORKS

"""
# Ensure output directory exists
output_dir = 'testview'
os.makedirs(output_dir, exist_ok=True)

# Example usage
save_retrieved_nifti(training_names[0], fs_training_images, output_dir)
save_retrieved_nifti(label_names[0], fs_labels, output_dir)
save_retrieved_nifti(testing_names[0], fs_testing_images, output_dir)
"""

upload_nifti_files('tempdata\images', fs_training_images)
logging.info("MRI images (for training use) uploaded.")
upload_nifti_files('tempdata\labels', fs_labels)
logging.info("Annotations/labels uploaded.")
upload_nifti_files('tempdata\testingimages', fs_testing_images)

