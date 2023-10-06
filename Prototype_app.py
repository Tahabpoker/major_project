import cv2
import numpy as np 
from flask import Flask
from PIL import Image
import tensorflow as tf 
from flask import render_template, request

app = Flask(__name__)

model = tf.keras.models.load_model("models/imageclassifier.h5")
pred = ""

@app.route("/",methods=['GET'])
def start():
    return render_template("Prototype_processing.html")

@app.route("/", methods=['POST'])
def end():

    imageFile = request.files.get('img',None)
    image_path = "./testimages/" + imageFile.filename
    imageFile.save(image_path)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)  # Make sure the number of channels is specified (usually 3 for RGB images)
    img = tf.image.resize(img, (256, 256))        # Resize the image
    img = img / 255.0                            # Normalize the pixel values to [0, 1]
    prediction = model.predict(np.expand_dims(img, 0))[0][0]

    if prediction < 0.5:
        pred = "dose not have pneumonia"
    else:
        pred = "have pneumonia"
        
    return render_template("Prototype_processing.html",my_string = pred)

if __name__ == '__main__':
    app.run(debug=True)
