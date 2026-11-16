from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from backend.schemas import PredictionInput

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model file path
MODEL_PATH = BASE_DIR / "model" / "weight_height_model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)

# Create FastAPI application
app = FastAPI(title="weight_height_model")

#GET → Checks whether API is running.
#POST → Sends weight to the API to get a prediction.


@app.get("/")
def home():
    return {
        "message": "Weight Height Prediction API is running"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    # Convert input into DataFrame
    user_input = pd.DataFrame({
        "Weight": [data.weight]
    })

    # Make prediction using trained model
    predicted_height = model.predict(user_input)

    return {
        "weight": data.weight,
        "predicted_height": float(predicted_height[0]),
        "unit": "cm"
    }