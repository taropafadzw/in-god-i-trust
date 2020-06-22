import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import pickle
import joblib


app = Flask(__name__)

def get_model():
    global model
    loaded_model = load_model('flowerclassify_model.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print(" * Loading keras model...")
get_model()

@app.route("/predict", methods=["GET","POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    prediction = loaded_model.predict(processed_image).tolist()

    response = {
        'prediction': {
           
            'daisy': prediction[0][0],
            'dandelion': prediction[0][1],
            'rose': prediction[0][2],
            'sunflower': prediction[0][3],
            'tulip': prediction[0][4],
         }
    }
    return jsonify(response)
