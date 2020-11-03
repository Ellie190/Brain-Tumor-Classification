
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import deque
import streamlit as st
import pandas as pd
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


# Run app on cmd/anaconda prompt
# Terminal command: streamlit run bt1.py

import streamlit as st
st.title("Brain Tumor Classification")
st.header("Convolutional Neural Network (CNN) for Brain Tumor Classification")
st.text("Upload a brain MRI scan for image classification as Brain Tumor or Healthy")


def import_and_classify(img, weights_file):
    # Load the model
    model = tf.keras.models.load_model(weights_file)
    

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

uploaded_file = st.file_uploader("Choose a brain MRI", type=("jpg","png","jpeg"))
st.text("Accepted Brain MRI scans | only jpg & jpeg & png files")

# To show sample 
mri_scan = st.checkbox("How does an MRI scan look")
if mri_scan:
    Image_1 = Image.open('Test Samples/Images/Y256.jpg')
    st.image(Image_1, width=300, caption = 'Sample MRI scan')


if st.button('Check Results'):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=400, caption='Uploaded MRI')
        st.write("")
        label = import_and_classify(image, 'BrainTumor') # Model Directory 
        if label == 1:
            st.write("The MRI scan has a brain tumor")
        else:
            st.write("The MRI scan is healthy")
