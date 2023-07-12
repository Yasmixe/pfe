import keras
from flask import Flask, render_template, request , jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import custom_object_scope
from flask_cors import CORS, cross_origin

app = Flask(__name__)
model = load_model(r'C:\Users\TUF\Documents\accuracy93\melanoma_resnet50.h5')

# Define the custom layer as a function
def custom_scale(x):
    return x * tf.constant(2.0, dtype=x.dtype)
upload_folder = r"C:\Users\TUF\Documents\pfee\templates"

def prediction(image, model):
    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))  # resize image to match model's expected sizing
    img = img.reshape(1, 224, 224, 3)
    img = img / 225 
    img2 = tf.cast(img, tf.float32)
    b= model.predict(img2)
    pre = np.argmax(b)
    pred = round(float(b[0][0]),4)*100

    if pre==0:
       return 'Predicted chance of melanoma: {pred} %'
        #print (f'Predicted chance of melanoma: {pred}%')
    else: 
        return 'not melanoma'
        #print (f'Predicted chance of melanoma: {pred}%')

@app.route('/', methods=["GET", "POST"])
@cross_origin()
def welcome() :
    return jsonify({'data' : "Welcome"})
@app.route("/predict", methods=["GET" , "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(upload_folder, image_file.filename)
            image_file.save(image_location)
            prediction_result = prediction(image_location, model)
            print(prediction_result)  # print the result to console for debugging purposes
           # return render_template("home.html", pre=prediction_result)
            return jsonify({'result': prediction_result, 'image': ''})

    return jsonify({'message' : "No image input"})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

#python iniware 
#faut loader le model f main bach machi a chaque fois on lload le model again et again
