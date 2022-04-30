import os

from flask import render_template, request
from flask import current_app as app

from run import model
from prediction import predict_image

@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():

  imagefile = request.files['imagefile']
  image_path = 'static/images/' + imagefile.filename

  imagefile.save(image_path)

  pic = os.path.join('static/images', imagefile.filename)
  prediction = predict_image(pic, model)

  return render_template('index.html', user_image=pic, prediction_text='The test image is {}'.format(prediction))
