# import cv2
import numpy as np 
from flask import Flask
from PIL import Image
import tensorflow as tf 
import keras
from keras.optimizers import Adam
from flask import render_template, request, redirect, url_for
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model("models/ds121_1k.h5")
# model1 = tf.keras.models.load_model("models/brainTumor.h5",compile = False)

# model1.compile(optimizer=Adam(lr=0.00005),  # Experiment with different learning rates
#             loss='categorical_crossentropy',
#             metrics=['accuracy'])

# Define a mapping of predictions to routes
prediction_routes_chest_xRay = {
    "Atelectasis": 'page0',
    "Cardiomegaly": 'page1',
    "Consolidation": 'page2',
    'No Finding': 'page3',
    'Hernia': 'page4',
    'Pneumonia': 'page5',
    'Mass': 'page6',
    'Effusion': 'page7',
    'Nodule': 'page8',
    'Infiltration': 'page9',
    'Pneumothorax': 'page10',
}
prediction_routes_brain_MRI = {
    'glioma': 'glioma',
    'meningioma': 'meningioma', 
    'notumor': 'notumor', 
    'pituitary': 'pituitary'
}


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
def chest_processing_page_start():
    return render_template("process.html")


@app.route("/process", methods=['POST'])
def chest_processing_page_end():
   
    # Load your PNG image
    global name
    name =  request.form.get('name',None)
    global pId
    pId = request.form.get('pId',None)
    global age
    age = request.form.get('age',None)
    global gender 
    gender = request.form.get('gender',None)

    imageFile = request.files.get('img',None)
    if imageFile:
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
        # confidence = prediction[0][predicted_label]  # Probability of the predicted class

        # If you want to map the class index to a class name
        class_names= ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'No Finding', 'Hernia', 'Pneumonia', 'Mass', 'Effusion', 'Nodule', 'Infiltration', 'Pneumothorax']
        
        # class_names = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"]
        # predicted_class_name = class_names[predicted_label]
        # predicted_class_name = class_names[predicted_label]
        predicted_class_name = "Atelectasis"
        # pred = predicted_class_name
          
        # return render_template("process.html",my_string = pred)
        # Replace this with your actual prediction logic
        if  predicted_class_name in prediction_routes_chest_xRay:
            route_name = prediction_routes_chest_xRay[predicted_class_name]
            return redirect(url_for(route_name))#pred=pred, Make sure 'pred' is passed correctly        

@app.route("/process11",methods=['GET'])
def brain_processing_page_start():
    return render_template("process11.html")

@app.route("/process11", methods=['POST'])
def brain_processing_page_end():
    global name
    name =  request.form.get('name',None)
    global pId
    pId = request.form.get('pId',None)
    global age
    age = request.form.get('age',None)
    global gender 
    gender = request.form.get('gender',None)

    imageFile = request.files.get('img',None)
    if imageFile:
        # image_path = "./testImages/" + imageFile.filename
        # imageFile.save(image_path)
        # # Replace with the path to your PNG image
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB format

        # # Resize the image to 224x224 to match the model's input shape
        # image = cv2.resize(image, (224, 224))

        # image = image / 255.0  # Normalize pixel values to [0, 1]

        # # Make a prediction using your model
        # prediction = model.predict(np.expand_dims(image, axis=0))
        # predicted_label = np.argmax(prediction)
        # class_names = ["glioma","meningioma","notumor","pituitary"]
        # predicted_class_name = class_names[predicted_label]
        predicted_class_name = "glioma"
        # Now, you can use this 'img' for prediction with your model
        if  predicted_class_name in prediction_routes_brain_MRI:
                route_name = prediction_routes_brain_MRI[predicted_class_name]
                return redirect(url_for(route_name))#pred=pred, Make sure 'pred' is passed correctly        


@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/page0", methods=["GET"])
def page0():
    pred = request.args.get('pred')
    return render_template("00.html",name = name,pId = pId,age = age,gender =gender ) 

@app.route("/page1", methods=["GET"])
def page1():
    pred = request.args.get('pred')
    return render_template("01.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page2", methods=["GET"])
def page2():
    pred = request.args.get('pred')
    return render_template("02.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page3", methods=["GET"])
def page3():
    pred = request.args.get('pred')
    return render_template("03.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page4", methods=["GET"])
def page4():
    pred = request.args.get('pred')
    return render_template("04.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page5", methods=["GET"])
def page5():
    pred = request.args.get('pred')
    return render_template("05.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page6", methods=["GET"])
def page6():
    pred = request.args.get('pred')
    return render_template("06.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page7", methods=["GET"])
def page7():
    pred = request.args.get('pred')
    return render_template("07.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page8", methods=["GET"])
def page8():
    pred = request.args.get('pred')
    return render_template("08.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page9", methods=["GET"])
def page9():
    pred = request.args.get('pred')
    return render_template("09.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/page10", methods=["GET"])
def page10():
    pred = request.args.get('pred')
    return render_template("10.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/glioma", methods=["GET"])
def glioma():
    pred = request.args.get('pred')
    return render_template("glioma.html", name = name,pId = pId,age = age,gender = gender)

@app.route("/meningioma", methods=["GET"])
def meningioma():
    pred = request.args.get('pred')
    return render_template("meningioma.html", name = name,pId = pId,age = age,gender = gender)


@app.route("/notumor", methods=["GET"])
def notumor():
    pred = request.args.get('pred')
    return render_template("notumor.html", name = name,pId = pId,age = age,gender = gender)


@app.route("/pituitary", methods=["GET"])
def pituitary():
    pred = request.args.get('pred')
    return render_template("pituitary.html", name = name,pId = pId,age = age,gender = gender)


#to run this app 
if __name__ == '__main__':
    app.run(debug=True)
    # runs at localhost:5000 or 127.0.0.1:5000



