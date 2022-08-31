import tensorflow as tf
import streamlit as st
from constants import Models
from constants import Streamlit_var
# Front end


disease_name = st.sidebar.selectbox(
    Streamlit_var.Select_disease,
    ("Pneumonia","Effusion","Edema")
)

st.write(Streamlit_var.selectbox, disease_name)

@st.cache(allow_output_mutation=True)
def load_model_effusion():
  model_v = tf.keras.models.load_model(Models.effusion_model)
  return model_v

@st.cache(allow_output_mutation=True)
def load_model_edema():
    model_v_2 = tf.keras.models.load_model(Models.edema_model)
    return model_v_2
with st.spinner(Streamlit_var.spinner):
    model = load_model_effusion()

st.title(Streamlit_var.title)

file = st.file_uploader(Streamlit_var.file_name, type=["jpg", "png","jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

model_v = load_model_effusion()
model_v_2 = load_model_edema()

def import_and_predict_effusion(image_data, model_v):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]

    yhat = model_v.predict(img_reshape)

    if yhat > 0.5:
        st.write(Streamlit_var.Normal)
    else:
        st.write(Streamlit_var.Effusion)

    return yhat

def import_and_predict_edema(image_data, model_v_2):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]

    yhat = model_v_2.predict(img_reshape)

    if yhat > 0.5:
        st.write(Streamlit_var.Normal)
    else:
        st.write(Streamlit_var.Edema)

    return yhat


if disease_name == "Effusion":
    if file is None:
        st.text(Streamlit_var.uplaod_file)
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict_effusion(image, model)
        st.write()

elif disease_name == "Edema":

    if file is None:
        st.text(Streamlit_var.uplaod_file)
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict_edema(image, model)
        st.write()




# train and test the model for pneumonia
