import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from twilio.rest import Client
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
app = Flask(__name__)
model = load_model(
    "/Users/aishaandatt/Downloads/goldenhack/bird.h5")


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        phone = request.form.get('phone')
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)

        print("prediction", preds)
        train_dir = "/Users/aishaandatt/Downloads/archive/train"
        generator = ImageDataGenerator()
        train_ds = generator.flow_from_directory(
            train_dir, target_size=(224, 224), batch_size=32)
        index = list(train_ds.class_indices.keys())

        print(np.argmax(preds))

        text = "Bird Species is : " + str(index[np.argmax(preds)])
       # break
    return text


if __name__ == '__main__':
    app.run(debug=True, threaded=False, host='192.168.29.83')
