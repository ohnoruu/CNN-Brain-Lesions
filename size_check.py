import os

# check folder sizes
folder_path = 'data\MRI-DWI' # directed path
files_and_dirs = os.listdir(folder_path)
files = [f for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]
file_count = len(files)
print(f"Number of files in {folder_path}: {file_count}")