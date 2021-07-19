from flask import Flask, render_template, request, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

cnn_model = load_model('cat_or_dog.h5')


def predict_model(path, model):
    test_image = image.load_img(path, target_size=(64, 64))

    # converting the image into 2D array redable by the model
    test_image = image.img_to_array(test_image)

    # changing the image into required shape of batch of 1
    test_image = np.expand_dims(test_image, axis=0)

    # rescaling the image
    test_image = test_image / 255.0

    result = model.predict(test_image)

    if result >= 0.5:
        return 'DOG'
    else:
        return 'CAT'


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    destination = os.path.join(APP_ROOT, 'static/')

    if not os.path.isdir(destination):
        os.mkdir(destination)

    for file in request.files.getlist('file'):

        file_name = file.filename
        finalpath = '/'.join([destination, file_name])
        file.save(finalpath)

    result = predict_model(finalpath, cnn_model)

    return render_template('prediction.html', result=result, filename=file_name)


if __name__ == '__main__':
    app.run(debug=True)
