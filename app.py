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
    image_path = "./testimages/" + imageFile.filename
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

    # Now 'predicted_class_name' contains the predicted class name for the single image,
    # and 'confidence' contains the associated confidence.
    pred = predicted_class_name
    # print(f'Predicted Class: {predicted_class_name}')
    # print(f'Confidence: {confidence * 100:.2f}%')

   

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
