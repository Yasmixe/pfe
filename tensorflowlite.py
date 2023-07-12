import tensorflow as tf

# Load the saved TensorFlow model
model = tf.saved_model.load(r'C:\Users\TUF\Documents\pfee\model3.h5')
# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(r'C:\Users\TUF\Documents\pfee\model3.h5')
tflite_model = converter.convert()

# Save the converted model to a file
with open('converted_model.tflite', 'wb') as f:
  f.write(tflite_model)
