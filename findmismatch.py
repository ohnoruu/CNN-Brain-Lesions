import os
import re

def get_file_subjects(folder_path):
    """
    Get a dictionary of file names indexed by subject number.
    Assumes file names are in the format 'sub-<number>_index_<number>.ext'.
    """
    file_subjects = {}
    for file_name in os.listdir(folder_path):
        pattern = r'sub-(\d+)'
        match = re.search(pattern, file_name)
        if match:
            subject_num = match.group(1)
            file_subjects[subject_num] = file_name
        else:
            print(f"Error retrieving subject number from {file_name}")
    return file_subjects

def find_mismatched_files(folder1, folder2):
    folder1_files = get_file_subjects(folder1)
    folder2_files = get_file_subjects(folder2)

    mismatched_files = []
    extra_files_folder1 = []
    extra_files_folder2 = []

    # Find mismatched files and extra files in folder1
    for key in folder1_files:
        if key not in folder2_files:
            extra_files_folder1.append(folder1_files[key])
        else:
            if folder1_files[key] != folder2_files[key]:
                mismatched_files.append((folder1_files[key], folder2_files[key]))

    # Find extra files in folder2
    for key in folder2_files:
        if key not in folder1_files:
            extra_files_folder2.append(folder2_files[key])

    return mismatched_files, extra_files_folder1, extra_files_folder2

# Example usage
folder1 = r'tempdata\testingimages'
folder2 = r'tempdata\testinglabels'

mismatched_files, extra_files_folder1, extra_files_folder2 = find_mismatched_files(folder1, folder2)

print("Mismatched files:")
for files in mismatched_files:
    print(files)

print("\nExtra files in folder1:")
for file in extra_files_folder1:
    print(file)

print("\nExtra files in folder2:")
for file in extra_files_folder2:
    print(file)