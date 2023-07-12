import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.models import load_model
import cv2
import os
import random
import tensorflow as tf
import keras.backend as K
from skimage.transform import resize
from tqdm import tqdm

PATH_TRAIN = r"C:\Users\TUF\Documents\pfee\resized data\train"
PATH_VALID = r"C:\Users\TUF\Documents\pfee\resized data\valid"
PATH_TEST = r"C:\Users\TUF\Documents\pfee\resized data\test"

conv_base = ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='sigmoid'))



for layer in conv_base.layers[:]:
    layer.trainable = False


model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])




batch_size = 20
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

history = model.fit_generator(train_gen,
                              epochs=2,
                              steps_per_epoch = 4096 // batch_size,
                              validation_data = valid_gen,
                              validation_steps = 2048 // batch_size)


for layer in conv_base.layers[:165]:
    layer.trainable = False
for layer in conv_base.layers[165:]:
    layer.trainable = True

model.compile(optimizer=optimizers.Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='resnet50v2.h5',
        monitor='accuracy', save_best_only=True, verbose=1)
]

history = model.fit_generator(train_gen,
                              epochs=100,
                              steps_per_epoch = 4096 // batch_size,
                              validation_data = valid_gen,
                              validation_steps = 2048 // batch_size)

model.save('MEL_resnet.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
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
