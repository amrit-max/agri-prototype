from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import io

app = FastAPI()


try:
    crop_model = joblib.load("../models/crop_model.pkl")
except FileNotFoundError:
    crop_model = None

try:
    plant_disease_model = load_model("../models/plant_disease_model.h5")
except FileNotFoundError:
    plant_disease_model = None


DISEASE_CLASSES = ['Early_blight', 'Healthy']

sensor_df = pd.read_csv("../data/sensor_data.csv")

@app.get("/")
def root():
    return {"message": "Agri Prototype Backend Running"}

@app.get("/sensor")
def get_sensor_data(n: int = 10):
    return sensor_df.tail(n).to_dict(orient="records")

@app.get("/predict")
def predict(features: str):
    if crop_model is None:
        return {"error": "Crop model not available"}
    try:
        x = np.array([list(map(float, features.split(",")))])
        pred = crop_model.predict(x)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/detect_disease/")
async def detect_disease(file: UploadFile = File(...)):
    if plant_disease_model is None:
        return JSONResponse(status_code=503, content={"error": "Plant disease model not available"})
    try:
       
        contents = await file.read()
        img = image.load_img(io.BytesIO(contents), target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  

        
        predictions = plant_disease_model.predict(img_array)
        score = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        predicted_class = DISEASE_CLASSES[predicted_class_index]

        return {"prediction": predicted_class, "confidence": float(score)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})