import keras
from flask import Flask, render_template, request , jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import custom_object_scope
import keras
import keras.backend as K
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x  

def iou(y_true, y_pred, smooth=1):
    y_true = K.expand_dims(y_true, axis=-1)
    y_pred = K.expand_dims(y_pred, axis=-1)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou_score = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou_score

with keras.utils.custom_object_scope({'iou': iou}):
    model_1= tf.keras.models.load_model(r"C:\Users\TUF\Documents\pfee\mymodel.h5")
#Segmenter l'image d'abord: 
image_path = 'C:\Users\TUF\Documents\pfee\AAAAA\mell\mask_262.jpeg'

x = read_image(image_path)
y_pred = model_1.predict(x)[0] > 0.5
y_pred = np.squeeze(y_pred, axis=-1)
y_pred = y_pred.astype(np.int32)

# save the mask
mask_path = "/home/Yasmine/PycharmProjects/PFEE/mask2.jpg"
import os
extension = ".jpg" # Change extension to desired format (e.g. .jpg, .bmp, etc.)
cv2.imwrite(os.path.join(mask_path + extension), y_pred.astype(np.uint8) * 255)

model = load_model(r'C:\Users\TUF\Documents\pfee\melanoma_resnet50.h5')
    
img = cv2.imread(r"/home/Yasmine/PycharmProjects/PFEE/mask2.jpg")
img = cv2.resize(img, (224, 224))  # resize image to match model's expected sizing
img = img.reshape(1, 224, 224, 3)
img = img / 225 
img2 = tf.cast(img, tf.float32)
b= model.predict(img2)
pre = np.argmax(b)
pred = round(float(b[0][0]),4)*100

if pre==0:
    print (f'Predicted chance of melanoma: {pred}%')
else: 
   print (f'Predicted chance of melanoma: {pred}%')
