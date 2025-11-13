"""Tests for model training and prediction."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.eta_routing.models.train_model import train_model
from src.eta_routing.models.predict_model import load_model, predict


def test_train_and_predict(tmp_path):
    # Create a tiny synthetic dataset
    df = pd.DataFrame({
        "pickup_datetime": ["2021-01-01 00:00:00", "2021-01-01 01:00:00"],
        "dropoff_datetime": ["2021-01-01 00:10:00", "2021-01-01 01:20:00"],
        "pickup_longitude": [-73.9, -74.0],
        "pickup_latitude": [40.8, 40.75],
        "dropoff_longitude": [-73.95, -74.05],
        "dropoff_latitude": [40.82, 40.77],
        "trip_distance": [2.5, 5.0],
    })
    data_path = tmp_path / "train.csv"
    df.to_csv(data_path, index=False)
    model_path = tmp_path / "model.pkl"
    train_model(str(data_path), str(model_path), test_size=0.5, random_state=0)
    model = load_model(str(model_path))
    preds = predict(model, df)
    assert isinstance(preds, np.ndarray)
    assert preds.size == 2