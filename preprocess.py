import nibabel as nib

# Load NIfTI file
nifti_img = nib.load('C:\Users\rubyd\Downloads\sub-1_rec-ADC_dwi.nii.gz')

# Access image data
img_data = nifti_img.get_fdata()

# Preprocess imge data as necessary for CNN input

from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

# Resize image 
#change dimensions later
resized_img = resize(img_data, (256, 256, 256))

# Normalize image data
scaler = MinMaxScaler()
normalized_img = scaler.fit_transform(resize_img.flatten().reshape(-1,1).reshape(resized_img.shape))

