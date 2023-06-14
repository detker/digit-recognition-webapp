############################################################
# Hand-written digits recognition web application,
# running on flask with deep neural network implementation.
# Author: Wojciech Krolikowski (@detker)
# 3.11.3 Python 
# Ver. 1.0
# Date: 13.06.2023
# E-mail: it.krolikowski@gmail.com
###########################################################



#Importing necessary libraries
import flask
from flask import Flask, render_template, url_for, request
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


#Initializing the garbage part of base64 encoded model.
base64_init = 21

#Initializing the Default Graph.
graph = tf.compat.v1.get_default_graph()

#Flask instance
app = Flask(__name__, template_folder='templates')

#Index route - rendering usr_inputing template
@app.route('/')
def index():
    return render_template('index.html')

#Predict route - using neural network to make a prediction and show results.
@app.route('/predict', methods=['POST'])
def predict():
    with graph.as_default():
    	if request.method == 'POST':
            
            #Loading model from binary using pickle lib.
            f = open(f'lenet5_trained.pkl', 'rb')
            model = pickle.load(f)
            f.close()        

            #Accessing raw base64 encoded image input from user.
            usr_input = request.form['url']
            #Removing garbage from base64 encoded image.
            usr_input = usr_input[base64_init:]
            #Decoding from base64 to 2-Dimensional array, one color channel, grayscale.
            decoded = base64.b64decode(usr_input)
            image = np.asarray(bytearray(decoded), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            #Thickening outlines of an input digit.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            image = cv2.dilate(image, kernel)
            #Downscaling input to match the model input shape (28x28px).
            resized_img = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)
            resized_img = np.asarray(resized_img, dtype="uint16")
            #Adjusting non-zero pixels values, making them brighter.
            resized_img[resized_img[:, :] > 0] += 100
            resized_img[resized_img[:, :] > 255] = 255
            #Reshaping the array to match model input dimensions.
            vect = np.asarray(resized_img, dtype="uint8")
            vect = vect.reshape(1, 28, 28, 1).astype('float32')
            #Pixels values normalization (now, the value range is from 0 to 1, 
            # prevents neurons to desaturate too early)
            vect /= 255

            #Saving pre-processed user input image.
            plt.imshow(vect[0, :, :])
            plt.savefig('cyfra-'+str(time.time())+'.png')

            #Prediction of neural network based on user input.
            model_prediction = model.predict(vect)
            
            #Getting the index of the maximum probability (indexes represent 
            # numbers behind the prediction).
            guess = np.argmax(model_prediction[0])

    return render_template('results.html', prediction=guess)

if __name__ == '__main__':
	app.run(debug=True)