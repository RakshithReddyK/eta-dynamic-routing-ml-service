"""Tests for feature engineering."""

import pandas as pd
from src.eta_routing.features.feature_engineering import engineer_features


def test_engineer_features_basic():
    df = pd.DataFrame({
        "pickup_datetime": ["2021-01-01 00:00:00"],
        "dropoff_datetime": ["2021-01-01 00:15:00"],
        "pickup_longitude": [-73.9],
        "pickup_latitude": [40.8],
        "dropoff_longitude": [-73.95],
        "dropoff_latitude": [40.82],
        "trip_distance": [2.5],
    })
    X, y = engineer_features(df)
    assert X.shape[0] == 1
    assert y.iloc[0] == 15.0
    assert "pickup_hour" in X.columns
    assert "trip_distance" in X.columns