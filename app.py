from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Payment Risk Prediction API")
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")


@app.get("/")
def home():
    return {
        "message": "Payment Risk Prediction API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: dict):
    input_data = pd.DataFrame([data])

    categorical_cols = ["employment_status","income_band"]

    input_data[categorical_cols] = encoder.transform(input_data[categorical_cols])

    prediction = model.predict(input_data)[0]

    return {
        "payment_risk": str(prediction)
    }