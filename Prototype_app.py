# import cv2
import numpy as np 
from flask import Flask
from PIL import Image
import tensorflow as tf 
from flask import render_template, request

app = Flask(__name__)

model = tf.keras.models.load_model("models/final_diseases_detection_model.h5")
pred = ""


@app.route('/login')
def login_page():
    return render_template('login.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route("/")
def home__page():
    return render_template("home.html")


@app.route("/home")
def home_page():
    return render_template("home.html")


@app.route("/process",methods=['GET'])
def processing_page_start():
    return render_template("process.html")


@app.route("/process", methods=['POST'])
def processing_page_end():

    # imageFile = request.files.get('img',None)
    # image_path = "./testimages/" + imageFile.filename
    # imageFile.save(image_path)

    # img = tf.io.read_file(image_path)
    # img = tf.image.decode_image(img, channels=3)  # Make sure the number of channels is specified (usually 3 for RGB images)
    # img = tf.image.resize(img, (256, 256))        # Resize the image
    # img = img / 255.0                            # Normalize the pixel values to [0, 1]
    # prediction = model.predict(np.expand_dims(img, 0))[0][0]

    # if prediction < 0.5:
    #     pred = "dose not have pneumonia"
    # else:
    #     pred = "have pneumonia"
        
    return render_template("process.html",my_string = pred)

@app.route("/about")
def about_page():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)
    # runs at localhost:5000 or 127.0.0.1:5000
