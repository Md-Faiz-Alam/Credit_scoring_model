"""Feature analysis report: Information Value and WoE bins, fit on the training split.

Outputs
  reports/iv_summary.csv   one row per feature: IV and strength
  reports/woe_bins.csv     every bin: count, bad rate, WoE, IV contribution

Usage:  uv run poe features
"""

import logging

import pandas as pd

from credit_scoring.config import Settings, get_settings
from credit_scoring.features.pipeline import scorecard_features, split_xy
from credit_scoring.features.woe import WoEEncoder

log = logging.getLogger(__name__)


def build_report(settings: Settings) -> pd.DataFrame:
    train = pd.read_parquet(settings.processed_dir / "train.parquet")
    X, y = split_xy(train)
    pipe = scorecard_features().fit(X, y)
    woe: WoEEncoder = pipe.named_steps["woe"]

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    iv = woe.iv_table()
    iv.to_csv(settings.reports_dir / "iv_summary.csv", index=False)
    woe.bin_table().to_csv(settings.reports_dir / "woe_bins.csv", index=False)
    return iv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    iv = build_report(get_settings())
    log.info("Information Value (train split):\n%s", iv.to_string(index=False))


if __name__ == "__main__":
    main()
