import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import os
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout , Flatten , MaxPooling2D , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, InceptionV3, InceptionResNetV2
from keras.applications import EfficientNetB0
import cv2
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.regularizers import l2
from keras.layers import MaxPool2D
import tensorflow as tf
import keras.backend as K
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
img_path= r"C:\Users\TUF\Documents\pfee\data_55.jpg"
# Load an image
model = InceptionV3(weights='imagenet')
DIM = 299
img = load_img(img_path, target_size=(DIM, DIM))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
#print(decode_predictions(preds))


with tf.GradientTape() as tape:
  last_conv_layer = model.get_layer('conv2d_93')
  iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
  model_out, last_conv_layer = iterate(x)
  class_out = model_out[:, np.argmax(model_out[0])]
  grads = tape.gradient(class_out, last_conv_layer)
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((8, 8))
plt.matshow(heatmap)
plt.show()
img = cv2.imread(img_path)
INTENSITY = 0.2

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

img = heatmap * INTENSITY + img

cv2.imshow('hh',cv2.imread(img_path))
cv2.imshow('image',img)
cv2.waitKey(0)
