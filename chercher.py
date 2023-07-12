import os

# Define the directories where the images and masks are stored
data_dir = "/home/Yasmine/PycharmProjects/PFEE/Data/augmented_train/melanoma/"
mask_dir = "/home/Yasmine/PycharmProjects/PFEE/prediction_masks/train/melanoma/"

# Loop through each image file in the data directory
for file_name in os.listdir(data_dir):
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        # Get the image number from the file name
        img_num = file_name.split("_")[1].split(".")[0]

        # Construct the mask file name using the image number
        mask_name = "mask_" + img_num + ".jpg"
        mask_path = os.path.join(mask_dir, mask_name)

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            # If the mask file does not exist, show the image file name and delete the image file
            print("Mask not found for " + file_name)
            os.remove(os.path.join(data_dir, file_name))
