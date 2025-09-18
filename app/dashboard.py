import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import io
import joblib
from keras.models import load_model
from keras.preprocessing import image


@st.cache_resource
def load_crop_model():
    try:
        return joblib.load("models/crop_model.pkl")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_disease_model():
    try:
        return load_model("models/plant_disease_model.h5")
    except FileNotFoundError:
        return None

crop_model = load_crop_model()
plant_disease_model = load_disease_model()

DISEASE_CLASSES = ['Early_blight', 'Healthy']

@st.cache_data
def load_sensor_data():
    return pd.read_csv("data/sensor_data.csv")

sensor_df = load_sensor_data()


st.title("AI-Powered Agriculture Prototype")


st.subheader(" Sensor Data (Soil & Weather)")
st.line_chart(sensor_df.set_index("timestamp")[["soil_moisture", "temp", "humidity"]])

latest = sensor_df.iloc[-1]
if latest['soil_moisture'] < 15:
    st.error("Drought Risk: Low Soil Moisture")
elif latest['humidity'] > 70 and latest['leaf_wetness'] == 1:
    st.warning("Pest/Disease Risk: High humidity + wet leaves")
else:
    st.success("Conditions Normal")

# NDVI Map
if os.path.exists("data/ndvi_map.png"):
    st.subheader("NDVI Crop Health Map")
    st.image("data/ndvi_map.png", use_container_width=True)
else:
    st.info("Run the notebook to generate NDVI map.")

# Plant Disease Detection
st.header("Plant Disease Detection")

uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button(" Detect Disease"):
        with st.spinner("Analyzing image..."):
            if plant_disease_model is None:
                st.error("Plant disease model not available. Please upload the model file.")
            else:
                try:
                   
                    img = image.load_img(io.BytesIO(uploaded_file.getvalue()), target_size=(128, 128))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array /= 255.0

                  
                    predictions = plant_disease_model.predict(img_array)
                    score = np.max(predictions)
                    predicted_class_index = np.argmax(predictions)
                    predicted_class = DISEASE_CLASSES[predicted_class_index]

                   
                    st.subheader("Detection Results:")
                    if predicted_class == "Healthy":
                        st.success(f"Prediction: The plant appears **Healthy**")
                    else:
                        st.error(f"Prediction:  The plant has **{predicted_class}**")
                    st.info(f"Confidence: {score:.2f}")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
