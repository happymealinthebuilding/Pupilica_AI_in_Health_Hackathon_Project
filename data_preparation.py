#data_preparation.py
import os
import zipfile
import pandas as pd
from shutil import copyfile, rmtree
import splitfolders

def prepare_data(zip_file_path, ground_truth_path, organized_data_path, split_data_path):
    """
    Prepares the ISIC dataset by unzipping, organizing, and splitting it.
    
    Args:
        zip_file_path (str): Path to the zipped dataset file.
        ground_truth_path (str): Path to the ground truth CSV.
        organized_data_path (str): Directory to save organized data.
        split_data_path (str): Directory to save train/val/test splits.
    """
    unzip_dir = '/content/unzipped_temp'
    print("Starting dataset setup...")

    # Unzip the dataset
    print(f"Unzipping {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print("Unzipping complete.")

    # Find the unzipped folder and load ground truth
    unzipped_folders = [d for d in os.listdir(unzip_dir) if os.path.isdir(os.path.join(unzip_dir, d))]
    data_dir = os.path.join(unzip_dir, unzipped_folders[0])
    df = pd.read_csv(ground_truth_path)
    class_names = [col for col in df.columns if col not in ['image', 'UNK']]

    # Organize images into class directories
    print("Organizing images into class directories...")
    for class_name in class_names:
        os.makedirs(os.path.join(organized_data_path, class_name), exist_ok=True)
    
    for index, row in df.iterrows():
        image_name = row['image'] + '.jpg'
        source_path = os.path.join(data_dir, image_name)
        for class_name in class_names:
            if row[class_name] == 1:
                dest_path = os.path.join(organized_data_path, class_name, image_name)
                if os.path.exists(source_path):
                    copyfile(source_path, dest_path)
                break
    print("Dataset organization complete.")

    # Clean up temporary directory
    rmtree(unzip_dir)
    print("Temporary unzipped directory removed.")

    # Split the dataset into train/val/test
    print("Splitting the dataset...")
    splitfolders.ratio(organized_data_path, output=split_data_path, seed=42, ratio=(.8, .1, .1), group_prefix=None)
    print("Dataset split complete.")

    return class_names