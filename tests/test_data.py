import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from credit_scoring.config import Settings
from credit_scoring.data import columns as c
from credit_scoring.data.clean import clean
from credit_scoring.data.load import load_raw
from credit_scoring.data.make_dataset import build
from credit_scoring.data.schema import CLEAN_SCHEMA
from credit_scoring.data.split import split_data

# ------------------------------------------------------------------ load


def test_load_renames_columns(raw_csv: Path) -> None:
    df = load_raw(raw_csv)
    assert c.TARGET in df.columns
    assert "past_due_30_59" in df.columns
    assert "SeriousDlqin2yrs" not in df.columns


def test_load_accepts_original_kaggle_header(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    # Original cs-training.csv: unnamed leading index column instead of "ID"
    path = tmp_path / "kaggle.csv"
    raw_df.drop(columns="ID").to_csv(path)  # pandas writes an unnamed index column
    assert c.ID in load_raw(path).columns


def test_load_rejects_unknown_columns(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    path = tmp_path / "bad.csv"
    raw_df.assign(Surprise=1).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Surprise"):
        load_raw(path)


def test_load_missing_file_has_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="poe download"):
        load_raw(tmp_path / "nope.csv")


# ------------------------------------------------------------------ clean


@pytest.fixture
def cleaned(raw_csv: Path) -> tuple[pd.DataFrame, object]:
    return clean(load_raw(raw_csv))


def test_clean_report_counts_each_planted_problem(raw_csv: Path) -> None:
    _, report = clean(load_raw(raw_csv))
    assert report.dropped_missing_target == 1
    assert report.dropped_duplicates == 1
    assert report.delinquency_codes_flagged == 1
    assert report.invalid_age_set_missing == 1
    assert report.rows_out == report.rows_in - 2


def test_clean_flags_special_codes_and_blanks_counts(raw_csv: Path) -> None:
    df, _ = clean(load_raw(raw_csv))
    flagged = df[df[c.DELINQUENCY_CODE_FLAG] == 1]
    assert len(flagged) == 1
    assert flagged[list(c.DELINQUENCY_COLS)].isna().all(axis=None)
    assert df[list(c.DELINQUENCY_COLS)].max().max() < 96


def test_clean_output_matches_contract(raw_csv: Path) -> None:
    df, _ = clean(load_raw(raw_csv))
    assert tuple(df.columns) == c.CLEAN_COLUMNS
    CLEAN_SCHEMA.validate(df)  # raises on violation


def test_clean_is_row_level_no_fitted_statistics(raw_csv: Path) -> None:
    """Cleaning a subset must give the same rows as cleaning everything.

    This guarantees cleaning learns nothing from the data, so running it before the split
    can't leak test information into training.
    """
    raw = (
        load_raw(raw_csv)
        .dropna(subset=[c.TARGET])
        .drop_duplicates(subset=[col for col in load_raw(raw_csv).columns if col != c.ID])
    )
    full, _ = clean(raw)
    half, _ = clean(raw.iloc[: len(raw) // 2])
    pd.testing.assert_frame_equal(full.iloc[: len(half)], half)


def test_clean_schema_rejects_bad_values(raw_csv: Path) -> None:
    df, _ = clean(load_raw(raw_csv))
    df.loc[0, "revolving_utilization"] = -1.0
    with pytest.raises(pandera.errors.SchemaError):
        CLEAN_SCHEMA.validate(df)


# ------------------------------------------------------------------ split


def test_split_sizes_disjoint_and_stratified(raw_csv: Path) -> None:
    df, _ = clean(load_raw(raw_csv))
    df = df.assign(_row=np.arange(len(df)))
    s = split_data(df, valid_size=0.15, test_size=0.15, random_state=1)

    assert len(s.train) + len(s.valid) + len(s.test) == len(df)
    ids = [set(part["_row"]) for part in s]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    overall = df[c.TARGET].mean()
    for part in s:
        assert abs(part[c.TARGET].mean() - overall) < 0.03


def test_split_is_deterministic(raw_csv: Path) -> None:
    df, _ = clean(load_raw(raw_csv))
    a = split_data(df, valid_size=0.15, test_size=0.15, random_state=7)
    b = split_data(df, valid_size=0.15, test_size=0.15, random_state=7)
    pd.testing.assert_frame_equal(a.test, b.test)


# ------------------------------------------------------------------ end to end


def test_build_writes_outputs_and_lineage(raw_csv: Path, tmp_path: Path) -> None:
    settings = Settings(data_dir=raw_csv.parent.parent, reports_dir=tmp_path / "reports")
    manifest = build(settings)

    for name in ("train", "valid", "test"):
        part = pd.read_parquet(settings.processed_dir / f"{name}.parquet")
        CLEAN_SCHEMA.validate(part)
    assert len(manifest["raw_sha256"]) == 64
    assert sum(s["rows"] for s in manifest["splits"].values()) == 400

    quality = json.loads((tmp_path / "reports" / "data_quality.json").read_text())
    assert quality["cleaning"]["dropped_duplicates"] == 1
    assert "debt_ratio_median_income_missing" in quality["profile"]["findings"]
