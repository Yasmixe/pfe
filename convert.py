import tensorflow as tf
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

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.regularizers import l2
from keras.layers import MaxPool2D

# Load the trained TensorFlow model
model = tf.keras.models.load_model(r'C:\Users\TUF\Documents\pfee\model4.h5')

# Define the input and output tensors
input_tensor = model.inputs[0]
output_tensor = model.outputs[0]

# Create a converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the input shape and output type properties
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Convert the model to TensorFlow Lite format
tflite_model = converter.convert()

# Save the converted model to a file
with open('my_model.tflite', 'wb') as f:
    f.write(tflite_model)
