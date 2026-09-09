from fastapi import FastAPI
from pydantic import BaseModel

APP_NAME = "student-ml-api"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)


class PredictRequest(BaseModel):
    value: float


@app.get("/")
def root():
    return {
        "message": "student-ml-api is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION,
    }


@app.get("/version")
def version():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    prediction = payload.value * 2  # simple placeholder "model"
    return {
        "input": payload.value,
        "prediction": prediction,
    }