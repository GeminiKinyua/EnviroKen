import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

def load_model():
    model = tf.keras.models.load_model('enviroken-classifier.h5')
    return model

def predict_image(model, img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0 
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)
    return class_index


st.title("ENVIROKEN")
st.write("This is a simple image classification web app that predicts the waste type.")
st.write("The goal is to integrate a reward system that determines a reward based on the predicted class.")

model = load_model()

class_labels = ['e-Waste', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Predicting...")
    image = image.resize((224, 224))
    label = predict_image(model, image)
    if label is not None:
        st.write('Predicted Class: %s' % class_labels[label[0]])

        # Reward based on class
        if st.button('Get Reward'):
            if class_labels[label[0]] == 'e-Waste':
                reward = 'You get reward 1!'
            elif class_labels[label[0]] == 'Glass':
                reward = 'You get reward 2!'
            elif class_labels[label[0]] == 'Metal':
                reward = 'You get reward 3!'
            elif class_labels[label[0]] == 'Organic':
                reward = 'You get reward 4!'
            elif class_labels[label[0]] == 'Paper':
                reward = 'You get reward 5!'
            elif class_labels[label[0]] == 'Plastic':
                reward = 'You get reward 6!'
            else:
                reward = 'You get no reward!'

            st.write(reward)
