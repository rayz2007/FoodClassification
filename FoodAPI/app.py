import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from skimage import io
from PIL import Image
import numpy as np
import io

app = flask.Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!!!!'


global model
model = tf.keras.models.load_model('/Users/rayzhang/Downloads/weights2.epoch-13-val_loss-0.82.hdf5')

food_list = {'apple_pie': 0, 'baby_back_ribs': 1, 'baklava': 2, 'beef_carpaccio': 3, 'beef_tartare': 4, 'beet_salad': 5,
             'beignets': 6, 'bibimbap': 7, 'bread_pudding': 8, 'breakfast_burrito': 9, 'bruschetta': 10,
             'caesar_salad': 11, 'cannoli': 12, 'caprese_salad': 13, 'carrot_cake': 14, 'ceviche': 15, 'cheesecake': 16,
             'cheese_plate': 17, 'chicken_curry': 18, 'chicken_quesadilla': 19, 'chicken_wings': 20,
             'chocolate_cake': 21, 'chocolate_mousse': 22, 'churros': 23, 'clam_chowder': 24, 'club_sandwich': 25,
             'crab_cakes': 26, 'creme_brulee': 27, 'croque_madame': 28, 'cup_cakes': 29, 'deviled_eggs': 30,
             'donuts': 31, 'dumplings': 32, 'edamame': 33, 'eggs_benedict': 34, 'escargots': 35, 'falafel': 36,
             'filet_mignon': 37, 'fish_and_chips': 38, 'foie_gras': 39, 'french_fries': 40, 'french_onion_soup': 41,
             'french_toast': 42, 'fried_calamari': 43, 'fried_rice': 44, 'frozen_yogurt': 45, 'garlic_bread': 46,
             'gnocchi': 47, 'greek_salad': 48, 'grilled_cheese_sandwich': 49, 'grilled_salmon': 50, 'guacamole': 51,
             'gyoza': 52, 'hamburger': 53, 'hot_and_sour_soup': 54, 'hot_dog': 55, 'huevos_rancheros': 56, 'hummus': 57,
             'ice_cream': 58, 'lasagna': 59, 'lobster_bisque': 60, 'lobster_roll_sandwich': 61,
             'macaroni_and_cheese': 62, 'macarons': 63, 'miso_soup': 64, 'mussels': 65, 'nachos': 66, 'omelette': 67,
             'onion_rings': 68, 'oysters': 69, 'pad_thai': 70, 'paella': 71, 'pancakes': 72, 'panna_cotta': 73,
             'peking_duck': 74, 'pho': 75, 'pizza': 76, 'pork_chop': 77, 'poutine': 78, 'prime_rib': 79,
             'pulled_pork_sandwich': 80, 'ramen': 81, 'ravioli': 82, 'red_velvet_cake': 83, 'risotto': 84, 'samosa': 85,
             'sashimi': 86, 'scallops': 87, 'seaweed_salad': 88, 'shrimp_and_grits': 89, 'spaghetti_bolognese': 90,
             'spaghetti_carbonara': 91, 'spring_rolls': 92, 'steak': 93, 'strawberry_shortcake': 94, 'sushi': 95,
             'tacos': 96, 'takoyaki': 97, 'tiramisu': 98, 'tuna_tartare': 99, 'waffles': 100}

food_list = dict((y,x) for x,y in food_list.items())


@app.route('/predict', methods=["GET", "POST"])
def predict_food():
    response = {}
    image = flask.request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    prediction, prob = predict(image)
    response['prediction'] = prediction
    response['probability'] = prob
    return flask.jsonify(response)


def predict(img):
    img = img.resize((299,299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    img_pred = model.predict(img)
    top_pred = np.argmax(img_pred, axis=1)[0]
    prob = str(round(max(img_pred)[top_pred] * 100, 2)) + '% probability'
    prediction = food_list[top_pred]
    return prediction, prob


if __name__ == '__main__':
    app.run()
