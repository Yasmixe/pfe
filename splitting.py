import os
import shutil

# Set the percentage of data to allocate to the validation set
validation_percentage = 20

#Set the paths to the original data folder, train folder, validation folder, mask folder, train mask folder and validation mask folder
data_folder = "/home/Yasmine/PycharmProjects/PFEE/data 64000 images/augmented_mask_2"
train_folder = "/home/Yasmine/PycharmProjects/PFEE/train_mask"
val_folder = "/home/Yasmine/PycharmProjects/PFEE/val_mask"

import os
import shutil
from sklearn.model_selection import train_test_split


# Set the percentage of data to allocate to the validation set
validation_percentage = 20

# Get a list of all the image files in the data folder
image_files = [f for f in os.listdir(data_folder) if f.endswith('.png')]

# Get the number of validation images based on the validation percentage
num_val_images = int(len(image_files) * validation_percentage / 100)

# Split the data into train and validation sets
train_files = image_files[num_val_images:]
val_files = image_files[:num_val_images]

# Move the train files to the train folder
for file in train_files:
    shutil.move(os.path.join(data_folder, file), os.path.join(train_folder, file))

# Move the validation files to the validation folder
for file in val_files:
    shutil.move(os.path.join(data_folder, file), os.path.join(val_folder, file))
