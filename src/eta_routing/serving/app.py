"""FastAPI application exposing an endpoint for ETA predictions."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from ..models.predict_model import load_model, predict
from ..features.feature_engineering import engineer_features

app = FastAPI(title="ETA Prediction API", version="0.1.0")


MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "model.pkl"


class ETARequest(BaseModel):
    pickup_datetime: datetime = Field(..., description="ISO8601 pickup timestamp")
    dropoff_datetime: Optional[datetime] = Field(
        None, description="Optional dropoff timestamp used only for computing features"
    )
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    trip_distance: float


class ETAResponse(BaseModel):
    predicted_duration_min: float
    predicted_dropoff_datetime: datetime


def _load_service_model():
    """Load the regression model from disk.  Cached in the app state."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train a model first.")
    return load_model(str(MODEL_PATH))


@app.on_event("startup")
def startup_event():
    # Load model at startup
    global MODEL
    MODEL = _load_service_model()


@app.post("/predict", response_model=ETAResponse)
def predict_eta(req: ETARequest):
    """Predict the travel duration (in minutes) and dropoff time for a single trip."""
    # Build DataFrame for single example
    data = pd.DataFrame([
        {
            "pickup_datetime": req.pickup_datetime,
            "dropoff_datetime": req.dropoff_datetime
            or req.pickup_datetime + timedelta(minutes=req.trip_distance / 0.5),
            "pickup_longitude": req.pickup_longitude,
            "pickup_latitude": req.pickup_latitude,
            "dropoff_longitude": req.dropoff_longitude,
            "dropoff_latitude": req.dropoff_latitude,
            "trip_distance": req.trip_distance,
        }
    ])
    try:
        preds = predict(MODEL, data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    predicted_duration = float(preds[0])
    predicted_dropoff = req.pickup_datetime + timedelta(minutes=predicted_duration)
    return ETAResponse(
        predicted_duration_min=predicted_duration,
        predicted_dropoff_datetime=predicted_dropoff,
    )