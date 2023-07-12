import os

folder_path = r"C:\Users\TUF\Documents\pfee\data 20000 images\augmented_mask"
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

num_images = sum([len(files) for _, _, files in os.walk(folder_path)
                  if any(f.endswith(tuple(valid_extensions)) for f in files)])

print("Number of images in folder:", num_images)