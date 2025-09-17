import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from PIL import Image
import io

st.title(" AI-Powered Agriculture Prototype")

BASE_URL = "http://localhost:8000"


st.subheader("Sensor Data (soil & weather)")
sensor_df = pd.read_csv("data/sensor_data.csv")
st.line_chart(sensor_df.set_index("timestamp")[["soil_moisture", "temp", "humidity"]])

latest = sensor_df.iloc[-1]
if latest['soil_moisture'] < 15:
    st.error(" Drought Risk: Low Soil Moisture")
elif latest['humidity'] > 70 and latest['leaf_wetness'] == 1:
    st.warning(" Pest/Disease Risk: High humidity + wet leaves")
else:
    st.success(" Conditions Normal")


if os.path.exists("data/ndvi_map.png"):
    st.subheader(" NDVI Crop Health Map")
    st.image("data/ndvi_map.png", use_container_width=True)
else:
    st.info("Run the notebook to generate NDVI map.")


st.header(" Plant Disease Detection")

uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
   
    if st.button("Detect Disease"):
        st.spinner("Analyzing image...")
        try:
           
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{BASE_URL}/detect_disease/", files=files)

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = result["confidence"]
                
                
                st.subheader("Detection Results:")
                if prediction == "Healthy":
                    
                    st.success(f"Prediction: The plant appears Healthy")
                else:
                    st.error(f"Prediction: The plant has Disease")
                st.info(f"Confidence: {confidence:.2f}")

            else:
                st.error(f"Error from server: {response.json().get('error', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please make sure the backend is running.")