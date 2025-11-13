"""Tests for data loading and validation."""

from pathlib import Path
import pandas as pd

from src.eta_routing.data.dataset_loader import load_dataset


def test_load_dataset_csv(tmp_path):
    # Create a simple CSV
    data_path = tmp_path / "data.csv"
    df = pd.DataFrame({"pickup_datetime": ["2021-01-01 00:00:00"], "dropoff_datetime": ["2021-01-01 00:10:00"], "pickup_longitude": [-73.9], "pickup_latitude": [40.8], "dropoff_longitude": [-73.95], "dropoff_latitude": [40.82], "trip_distance": [2.5]})
    df.to_csv(data_path, index=False)
    loaded = load_dataset(str(data_path))
    assert not loaded.empty
    assert list(loaded.columns) == list(df.columns)