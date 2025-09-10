import os
import pandas as pd
from shutil import copyfile

# Define the paths
data_dir = '/Users/azratuncay/Desktop/hackathon/ISIC_2019_Training_Input'
ground_truth_csv = '/Users/azratuncay/Desktop/hackathon/ISIC_2019_Training_GroundTruth.csv'
output_dir = '/Users/azratuncay/Desktop/hackathon/organized_dataset'

# Load the ground truth CSV file
df = pd.read_csv(ground_truth_csv)

# Create output directories for each class
class_names = [col for col in df.columns if col not in ['image', 'UNK']]
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Loop through the rows and move images
for index, row in df.iterrows():
    image_name = row['image'] + '.jpg'
    source_path = os.path.join(data_dir, image_name)
    
    # Find the correct class for the image
    for class_name in class_names:
        if row[class_name] == 1:
            dest_path = os.path.join(output_dir, class_name, image_name)
            copyfile(source_path, dest_path)
            print(f"Moved {image_name} to {class_name}")
            break

print("Dataset organization complete.")