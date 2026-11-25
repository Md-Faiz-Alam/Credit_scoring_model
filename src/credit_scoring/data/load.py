"""Read the raw CSV and normalise it to snake_case column names."""

import hashlib
from pathlib import Path

import pandas as pd

from credit_scoring.data import columns as c
from credit_scoring.data.schema import RAW_SCHEMA


def file_sha256(path: Path) -> str:
    """Fingerprint of the input file, recorded so every model traces back to its exact data."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw dataset.

    Accepts both the original Kaggle ``cs-training.csv`` (unnamed index column) and the
    re-saved variant with an ``ID`` header. Unknown columns raise an error, so an upstream
    schema change can't slip through.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `uv run poe download` or see README."
        )
    df = pd.read_csv(path)
    unknown = set(df.columns) - set(c.RAW_TO_CLEAN)
    if unknown:
        raise ValueError(f"Unexpected columns in raw data: {sorted(unknown)}")
    df = df.rename(columns=c.RAW_TO_CLEAN)
    return RAW_SCHEMA.validate(df)
