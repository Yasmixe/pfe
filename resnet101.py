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
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping,ModelCheckpoint

PATH = r"C:\Users\TUF\Documents\pfee\masked_images"

PATH_TRAIN = r"C:\Users\TUF\Documents\pfee\masked_images\train"
PATH_VALID = r"C:\Users\TUF\Documents\pfee\masked_images\valid"
PATH_TEST = r"C:\Users\TUF\Documents\pfee\masked_images\test"


batch_size = 32
target_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(PATH_TRAIN,
                                              target_size=target_size,
                                              batch_size=batch_size)

valid_gen = test_datagen.flow_from_directory(PATH_VALID,
                                             target_size=target_size,
                                             batch_size=batch_size)

test_gen = test_datagen.flow_from_directory(PATH_TEST,
                                            target_size=target_size,
                                            batch_size=batch_size)

from keras.applications.resnet import ResNet101
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam

# Load the ResNet18 model
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add your own top layers for your specific classification task
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax')) 

# Compile the model
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='resnet101.h5',
        monitor='accuracy', save_best_only=True, verbose=1)
]
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
        train_gen,
        epochs=50,
        callbacks = callbacks_list,
        steps_per_epoch=4096 // batch_size,
        validation_data=valid_gen,
        validation_steps=2048 // batch_size,
        verbose=1
    )
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()