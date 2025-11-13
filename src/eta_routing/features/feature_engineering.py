"""Helper functions to transform raw trip data into model‑ready features."""

from __future__ import annotations

import pandas as pd
from typing import Tuple, Optional


def engineer_features(df: pd.DataFrame, *, return_target: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Compute feature columns from the raw dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe with required columns.
    return_target : bool, default True
        Whether to return the target column (duration in minutes).  Set to False when
        preparing data for inference.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame containing numerical features for the model.
    y : pandas.Series or None
        Series containing the target values, or None if ``return_target`` is False.

    Notes
    -----
    The function expects the input DataFrame to have at least the following columns:
    ``pickup_datetime``, ``dropoff_datetime``, ``pickup_longitude``, ``pickup_latitude``,
    ``dropoff_longitude``, ``dropoff_latitude`` and either ``trip_distance`` or ``actual_duration_min``.
    If ``actual_duration_min`` exists, it will be used as the target; otherwise the duration
    will be computed from the timestamp columns.
    """
    df = df.copy()
    # Parse datetimes
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    # Compute duration (minutes) if not provided
    if "actual_duration_min" in df.columns:
        duration = df["actual_duration_min"].astype(float)
    else:
        duration = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60.0
    # Feature extraction
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek
    X = df[[
        "pickup_hour",
        "pickup_day_of_week",
        "trip_distance",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]].astype(float)
    if return_target:
        return X, duration
    return X, None