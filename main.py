import streamlit as st
from PIL import Image, ImageOps
import pickle as pkl
import numpy as np

st.title('Handwritten Digit Recognition')

input = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input)

st.header('Upload an image')
image = st.file_uploader('Choose an image', type=(['png', 'jpg', 'jpeg']))

if image is not None:
  image = Image.open(image)
  st.image(image, caption='Test image')

  if st.button('Predict'):
    image = image.resize((8*8, 1))
    image = ImageOps.grayscale(image) 
    vector = np.array(image)
    label = str((model.predict(vector))[0])

    st.header('Result')
    st.text(label)
