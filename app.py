import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Define the custom loss function as before
class CustomSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, ignore_class=None, reduction=tf.keras.losses.Reduction.AUTO, name='sparse_categorical_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.ignore_class = ignore_class

    def call(self, y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)

# Custom object dictionary
custom_objects = {
    'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy
}
st.header('Image Classification Model')
# Load model with custom objects
model = load_model(r'D:\ML_Project\Image_classification\Image_classify.h5', custom_objects=custom_objects)

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
    'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
    'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

img_height = 180
img_width = 180

# Streamlit user input
image_path = st.text_input('Enter Image path', 'Apple.jpg')

if image_path:
    try:
        # Load image
        image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, 0)

        # Make predictions
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])

        # Display results
        st.image(image_path, width=200)
        st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
        st.write('With accuracy of ' + str(np.max(score) * 100))
    except Exception as e:
        st.write(f"Error loading or processing image: {e}")
