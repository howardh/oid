import os
import numpy as np
import dill
import urllib
import urllib.request

from flask import Flask
from flask import request

import train_food

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    #print('a')
    #img = np.random.randn(1, 3, 224, 224).astype(np.float32)
    #print('b')
    #onnx_model = onnx.load('onnx/classifier-fruit.onnx')
    #print('c')
    #outputs = caffe2.python.onnx.backend.run_model(model, [img])
    train_food.predict('weights/classifier-fruit-11.pt','labels/labels-food.pkl','875806_R.jpg')

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/predict')
    def predict():
        url = request.args.get('url')
        # Download URL
        file_name = 'tempimage'
        urllib.request.urlretrieve(url, file_name) 
        # Predict and return prediction
        description = train_food.predict('weights/classifier-fruit-11.pt','labels/labels-food.pkl',file_name)
        return description

    return app

