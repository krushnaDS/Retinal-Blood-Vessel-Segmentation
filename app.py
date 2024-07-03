import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image

# Define image dimensions
H = 512
W = 512

# Function to read and preprocess the image
def read_image(image):
    x = np.array(image)
    x = cv2.resize(x, (W, H))  # Resize if necessary
    x = x / 255.0
    x = x.astype(np.float32)
    return x

# Load the model
model_path = "/path/to/your/model.h5"  # Update this path to your model
model = tf.keras.models.load_model(model_path)

# Streamlit app
st.title("Retina Blood Vessel Segmentation")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    x = read_image(image)

    # Prediction
    y_pred = model.predict(np.expand_dims(x, axis=0))[0]
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = np.squeeze(y_pred, axis=-1) * 255

    # Display the prediction
    st.image(y_pred, caption='Predicted Mask', use_column_width=True)
