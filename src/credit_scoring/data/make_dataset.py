"""Build the modelling datasets: raw CSV -> validated, cleaned, split parquet files.

Outputs
  data/processed/{train,valid,test}.parquet
  data/processed/manifest.json   lineage: input hash, settings, row counts
  reports/data_quality.json      cleaning log + data profile

Usage:  uv run poe data
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from credit_scoring import __version__
from credit_scoring.config import Settings, get_settings
from credit_scoring.data import columns as c
from credit_scoring.data.clean import clean
from credit_scoring.data.load import file_sha256, load_raw
from credit_scoring.data.quality import profile
from credit_scoring.data.split import split_data

log = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build(settings: Settings) -> dict[str, Any]:
    raw_path = settings.raw_data_path
    log.info("Loading %s", raw_path)
    raw = load_raw(raw_path)

    df, cleaning = clean(raw)
    log.info("Cleaning: %s", cleaning.as_dict())

    splits = split_data(
        df,
        valid_size=settings.valid_size,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    out_dir = settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for name, part in splits._asdict().items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        split_stats[name] = {
            "rows": len(part),
            "defaults": int(part[c.TARGET].sum()),
            "default_rate": round(float(part[c.TARGET].mean()), 6),
        }
        log.info("%-5s %7d rows  default rate %.2f%%", name, len(part), 100 * part[c.TARGET].mean())

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": __version__,
        "raw_file": raw_path.name,
        "raw_sha256": file_sha256(raw_path),
        "random_state": settings.random_state,
        "valid_size": settings.valid_size,
        "test_size": settings.test_size,
        "columns": list(c.CLEAN_COLUMNS),
        "splits": split_stats,
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(
        settings.reports_dir / "data_quality.json",
        {"cleaning": cleaning.as_dict(), "profile": profile(df), "splits": split_stats},
    )
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build(get_settings())


if __name__ == "__main__":
    main()
