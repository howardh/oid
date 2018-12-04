import os
import numpy as np
import dill
import urllib
import urllib.request

from flask import Flask
from flask import request

import train_food

weights_file = os.path.join('/home/ubuntu/data/weights','classifier-fruit-11.pt')
labels_file = os.path.join('labels', 'labels-food.pkl')
example_photo_file = '875806_R.jpg'

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

    # Make on prediction in order to load the model
    train_food.predict(weights_file, labels_file, example_photo_file)

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/predict')
    def predict():
        url = request.args.get('url')
        # Download URL
        file_name = 'tempimage' # FIXME: Won't work with multiple concurrent requests. Use a UUID and delete afterwards?
        urllib.request.urlretrieve(url, file_name) 
        # Predict and return prediction
        description = train_food.predict(weights_file,labels_file,file_name)
        return description

    return app

if __name__=='__main__':
    app = create_app()
    app.run(host='0.0.0.0')
