"""Tests for the FastAPI endpoint."""

from fastapi.testclient import TestClient
import os
from src.eta_routing.serving.app import app


def test_predict_endpoint(tmp_path, monkeypatch):
    # monkeypatch the model path to a temporary trained model
    from src.eta_routing.models.train_model import train_model
    from src.eta_routing.models.predict_model import load_model
    import pandas as pd

    # build small dataset and train a model
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
    # monkeypatch the global model path and reload the model
    from src.eta_routing.serving import app as app_module
    app_module.MODEL_PATH = model_path
    # reload the model
    app_module.MODEL = load_model(str(model_path))
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "pickup_datetime": "2021-01-02T10:00:00",
            "pickup_longitude": -73.95,
            "pickup_latitude": 40.82,
            "dropoff_longitude": -73.96,
            "dropoff_latitude": 40.83,
            "trip_distance": 3.0
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_duration_min" in data
    assert "predicted_dropoff_datetime" in data