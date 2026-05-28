from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_DATA_FILENAME = "mall_customers.csv"
KAGGLE_CACHE_FILE = Path.home() / ".cache" / "kagglehub" / "datasets" / "shrutimechlearn" / "customer-data" / "versions" / "1" / "Mall_Customers.csv"


def ensure_data_file(dataset_filename: str = DEFAULT_DATA_FILENAME) -> Path:
    """Ensure the dataset exists in the local data directory and return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / dataset_filename
    if destination.exists():
        return destination

    if not KAGGLE_CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find cached dataset at {KAGGLE_CACHE_FILE}. "
            "Download the dataset with kagglehub or place it in the data directory."
        )
    shutil.copy(KAGGLE_CACHE_FILE, destination)
    return destination


def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the dataset from a CSV file into a pandas DataFrame."""
    path = Path(path) if path is not None else ensure_data_file()
    return pd.read_csv(path)
