import numpy as np
import os
import shutil
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

# Set the path to the dataset
dataset_path = './data/kagglecatsanddogs_5340/PetImages'

# Set paths for cat and dog folders within the dataset
cats_folder = os.path.join(dataset_path, 'Cat')
dogs_folder = os.path.join(dataset_path, 'Dog')

# Set the percentage of the dataset to be used for testing
test_set_percentage = 0.2

# Set paths for train and test folders
train_folder = os.path.join(dataset_path, 'train')
test_folder = os.path.join(dataset_path, 'test')

# Create necessary directories
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

train_folder_dog = os.path.join(train_folder, 'Dog')
train_folder_cat = os.path.join(train_folder, 'Cat')
test_folder_dog = os.path.join(test_folder, 'Dog')
test_folder_cat = os.path.join(test_folder, 'Cat')

os.makedirs(train_folder_dog, exist_ok=True)
os.makedirs(train_folder_cat, exist_ok=True)
os.makedirs(test_folder_dog, exist_ok=True)
os.makedirs(test_folder_cat, exist_ok=True)

# Function to copy files from original dataset to subfolders
def copy_files(file_list, source_folder, destination_folder):
    for i, file_name in enumerate(file_list):
        source_path = os.path.join(source_folder, file_name)
        print("Source path: ", source_path)

        destination_path = os.path.join(destination_folder, file_name)
        print("Destination path: ", destination_path)

        shutil.copyfile(source_path, destination_path)

# List files in dog and cat folders
dogs_files = os.listdir(dogs_folder)
cats_files = os.listdir(cats_folder)

# Shuffle the file lists randomly
random.shuffle(dogs_files)
random.shuffle(cats_files)

# Calculate the number of test samples for dogs and cats
num_test_dogs = int(test_set_percentage * len(dogs_files))
num_test_cats = int(test_set_percentage * len(cats_files))

# Split the file lists into train and test sets for dogs and cats
train_dogs = dogs_files[num_test_dogs:]
test_dogs = dogs_files[:num_test_dogs]

train_cats = cats_files[num_test_cats:]
test_cats = cats_files[:num_test_cats]

# Copy files to respective train and test sets
copy_files(train_dogs, dogs_folder, os.path.join(train_folder, 'Dog'))
copy_files(test_dogs, dogs_folder, os.path.join(test_folder, 'Dog'))

copy_files(train_cats, cats_folder, os.path.join(train_folder, 'Cat'))
copy_files(test_cats, cats_folder, os.path.join(test_folder, 'Cat'))
