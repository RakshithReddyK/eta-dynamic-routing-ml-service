"""Functions to load datasets for ETA prediction."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Union


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Load a dataset from a CSV or Parquet file.

    Parameters
    ----------
    path: Union[str, Path]
        Path to the dataset file.  Supported formats are CSV and Parquet.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If a Parquet file is provided but pyarrow is not installed.
    """
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Data file {fp} not found.")
    if fp.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet requires the 'pyarrow' package.  "
                "Install it or convert the file to CSV."
            ) from exc
        table = pq.read_table(fp)
        df = table.to_pandas()
    else:
        df = pd.read_csv(fp)
    return df