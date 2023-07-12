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


PATH = r"C:\Users\TUF\Documents\pfee\masked_images"

PATH_TRAIN = r"C:\Users\TUF\Documents\pfee\masked\train"
PATH_VALID = r"C:\Users\TUF\Documents\pfee\masked\valid"
PATH_TEST = r"C:\Users\TUF\Documents\pfee\masked\test"

import os

def dataset_display(path, sample, type):
    img_path = path + '\\' + type + '\\'
    img_name = os.listdir(img_path)[sample]
    img_path_full = img_path + img_name
    img = load_img(img_path_full, target_size=(252, 252))
    imgplot = plt.imshow(img)
    plt.title(type)
    plt.show()
    return img_path_full


sample_num = 77
batch_size = 64
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

import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint

base_model = InceptionResNetV2(weights='imagenet',include_top=False, input_shape=(224,224,3))
x = base_model.output
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
predictions = Dense(2, activation='softmax', kernel_initializer='random_uniform')(x)
model = Model(inputs=base_model.input, outputs=predictions)

import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
model = load_model(r'C:\Users\TUF\Documents\pfee\melanoma_resnet50.h5')
# Compile the model
  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
        train_gen,
        epochs=50,
        steps_per_epoch=4096 // batch_size,
        validation_data=valid_gen,
        validation_steps=2048 // batch_size,
        verbose=1,
    )
# Plot the accuracy

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b-', label='Training acc')
plt.plot(epochs, val_acc, 'r-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save("newmay.h5")