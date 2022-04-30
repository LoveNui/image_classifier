import glob
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


def predict_image(image, model):
    '''Make prediction

    Params:
        image: path to image
        model: model to make prediction

    Returns:
        name of predicted class
    '''

    list_classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
     'watermelon']

    test_image = load_img(image, target_size = (224, 224))
    test_image = img_to_array(test_image)

    test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    test_image = preprocess_input(test_image)

    result = model.predict(test_image)[0].argmax()

    class_names = sorted(list_classes)
    name_id_map = dict(zip(class_names, range(len(class_names))))

    labels = list(name_id_map.items())

    for label, i in labels:
        if i == result:
            return label