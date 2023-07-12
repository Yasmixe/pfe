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

PATH_TRAIN = r"C:\Users\TUF\Documents\pfee\masked_images\train"
PATH_VALID = r"C:\Users\TUF\Documents\pfee\masked_images\valid"
PATH_TEST = r"C:\Users\TUF\Documents\pfee\masked_images\test"

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

model = load_model(r"C:\Users\TUF\Documents\pfee\inceptionresnet.h5")
def display_results(img_num, check_type):
    def load_image(img_path_full, show = False):
        img = load_img(img_path_full, target_size = (224, 224))
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis = 0)
        img_tensor /= 255
        
        return img_tensor
    
    PROOF_PATH = dataset_display(PATH_TEST, img_num, check_type)
    pred = model.predict(load_image(PROOF_PATH))
    pred = round(float(pred[0][0]),4)*100
    actual = 100 if check_type == 'melanoma' else 0
    diff = abs(round(pred-actual,4))
    
    y = ['Predicted','Actual','Accuracy']
    x = [pred+1,actual+1,(100-diff)]
    
    f = plt.figure()
    f.set_figwidth(3.4)
    f.set_figheight(1)
    plt.title('chance of melanoma')
    plt.barh(y,x,color=['white', 'lightgrey','b' if 100-diff > 76 else 'r'],edgecolor='black')
    plt.xlim([0,100])
    plt.show()
    
    print (f'Predicted chance of melanoma: {pred}%')
    print (f"Actual: {actual}%")
    print (f'Difference: {diff}%')
    print('\n')

import random

# Create a list of indices in the range [0, 16) excluding the range [8, 16)
indices = list(range(8)) + list(range(16, 30))

# Shuffle the indices
random.shuffle(indices)

# Display the results for the shuffled indices
for i in indices[:4]:
    display_results(i, 'not melanoma')

for i in indices[4:]:
    display_results(i, 'melanoma')
