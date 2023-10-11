# import cv2
import numpy as np 
from flask import Flask
from PIL import Image
import tensorflow as tf 
from flask import render_template, request
import cv2

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
   

    # Load your PNG image
    imageFile = request.files.get('img',None)
    image_path = "./testImages/" + imageFile.filename
    imageFile.save(image_path)
      # Replace with the path to your PNG image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB format

    # Resize the image to 224x224 to match the model's input shape
    image = cv2.resize(image, (224, 224))

    image = image / 255.0  # Normalize pixel values to [0, 1]

    # Make a prediction using your model
    prediction = model.predict(np.expand_dims(image, axis=0))  # Add an extra dimension for batch size

    # Get the predicted class label and associated probability/confidence
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]  # Probability of the predicted class

    # If you want to map the class index to a class name
    class_names = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"]
    predicted_class_name = class_names[predicted_label]

    pred = predicted_class_name
    
        
    return render_template("process.html",my_string = pred)

@app.route("/about")
def about_page():
    return render_template("about.html")
#to run this app 
if __name__ == '__main__':
    app.run(debug=True)
    # runs at localhost:5000 or 127.0.0.1:5000



