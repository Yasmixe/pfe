import cv2
import os
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import keras.backend as K
from skimage.transform import resize
from tqdm import tqdm
import cv2
import cv2
import os
import numpy as np
W=H=256
# Define the paths
image_dir = r"C:\Users\TUF\Documents\pfee\wissal\resized"
mask_dir = r"C:\Users\TUF\Documents\pfee\wissal\re"
output_dir = r"C:\Users\TUF\Documents\pfee\wissal\appl"

# Load the model
def iou(y_true, y_pred, smooth=1):
    y_true = K.expand_dims(y_true, axis=-1)
    y_pred = K.expand_dims(y_pred, axis=-1)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou_score = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou_score

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x                                ## (1, 256, 256, 3)

np.random.seed(42)
tf.random.set_seed(42)

with keras.utils.custom_object_scope({'iou': iou}):
    model = tf.keras.models.load_model(r"C:\Users\TUF\Documents\pfee\mymodel.h5")

# Loop through the images and apply the predicted masks
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Load the original image
        image_path = os.path.join(image_dir, filename)
        original_image = cv2.imread(image_path)

        # Predict the mask
        x = read_image(image_path)
        y_pred = model.predict(x)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        # Save the predicted mask
        mask_filename = "mask_" + filename[:-4] + ".jpg"
        mask_path = os.path.join(mask_dir, mask_filename)
        cv2.imwrite(mask_path, y_pred.astype(np.uint8) * 255)

        # Load the predicted mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        mask = mask.astype(np.uint8)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        blurred_image = cv2.GaussianBlur(masked_image, (5, 5), 0)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(blurred_image, -1, kernel)

        # Save the preprocessed image in the output folder
        preprocessed_path = os.path.join(output_dir, "preprocessed_" + filename)
        cv2.imwrite(preprocessed_path, sharpened_image)

print("Preprocessing completed!")
