import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score,recall_score, precision_score, SCORERS
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,MaxPool2D, Conv2DTranspose, Input, Activation, Concatenate, CenterCrop
from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model, CustomObjectScope
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import keras
import keras.backend as K
def iou(y_true, y_pred, smooth=1):
    y_true = K.expand_dims(y_true, axis=-1)
    y_pred = K.expand_dims(y_pred, axis=-1)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou_score = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou_score

with keras.utils.custom_object_scope({'iou': iou}):
    model = tf.keras.models.load_model(r"C:\Users\TUF\Documents\iou= 85%\mymodel.h5")


from skimage import io
from sklearn.metrics import confusion_matrix

# Charger l'image, le masque original et le masque prédit
image = io.imread(r'C:\Users\TUF\Documents\pfee\ABCD\images\data_41.jpg')
mask = io.imread(r'C:\Users\TUF\Documents\pfee\ABCD\masks\mask_41.png')
original_mask = cv2.resize(mask, (256, 256))

predicted_mask = io.imread(r'C:\Users\TUF\Documents\pfee\ABCD\predicted\data_41_mask.jpg')

# Calculer l'IoU
intersection = np.logical_and(original_mask, predicted_mask)

union = np.logical_or(original_mask, predicted_mask)
iou = np.sum(intersection) / np.sum(union)
print("Intersection sur l'Union (IoU):", iou)
# Calculer la précision, le rappel et le score F1
tp = np.sum(intersection)
fp = np.sum(predicted_mask) - tp
fn = np.sum(original_mask) - tp
precision = tp / (tp + fp)

# Calculer l'accuracy
accuracy = accuracy_score(original_mask.ravel(), predicted_mask.ravel())
print("Accuracy:", accuracy)

# Calculer le Jaccard score
jaccard = jaccard_score(original_mask.ravel(), predicted_mask.ravel(), average='weighted')
print("Jaccard score:", jaccard)