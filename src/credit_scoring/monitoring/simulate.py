"""Simulate production traffic for monitoring demos.

Held-out test applicants are converted to API requests and scored through the real
``ScoringService``, which writes the same prediction log the live API writes. Scenarios:

* ``baseline``: applicants as they are. Should look stable against training.
* ``downturn``: an economic shock. Incomes fall 15%, balances and utilization rise,
  and more borrowers miss payments. Should trigger drift alerts.
* ``young_segment``: a marketing campaign that attracts younger applicants. Their
  behaviour is unchanged, but the population mix shifts (age and score drift).

Usage:  uv run poe simulate --scenario downturn
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from credit_scoring.api.schemas import Applicant
from credit_scoring.api.service import PredictionLog, ScoringService
from credit_scoring.config import PROJECT_ROOT, Settings, get_settings
from credit_scoring.data import columns as c
from credit_scoring.models.bundle import CreditRiskModel

log = logging.getLogger(__name__)

SCENARIOS = ("baseline", "downturn", "young_segment")


def apply_scenario(df: pd.DataFrame, scenario: str, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if scenario == "baseline":
        return out
    if scenario == "downturn":
        income = out["monthly_income"]
        monthly_debt = np.where(income > 0, out["debt_ratio"] * income, out["debt_ratio"])
        out["monthly_income"] = income * 0.85
        new_debt = monthly_debt * 1.10
        out["debt_ratio"] = np.where(
            out["monthly_income"] > 0, new_debt / out["monthly_income"], new_debt
        )
        util = out["revolving_utilization"]
        # stress genuine balances only; leave implausible data-error values (> 10x) as they are
        out["revolving_utilization"] = np.where(util <= 1.2, (util * 1.25).clip(upper=1.5), util)
        extra_late = rng.random(len(out)) < 0.12
        out.loc[extra_late, "past_due_30_59"] = out.loc[extra_late, "past_due_30_59"] + 1
        return out
    if scenario == "young_segment":
        weights = np.where(out["age"] < 35, 6.0, 1.0)
        idx = rng.choice(len(out), size=len(out), replace=True, p=weights / weights.sum())
        return out.iloc[idx].reset_index(drop=True)
    raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")


def to_applicants(df: pd.DataFrame) -> tuple[list[Applicant], int]:
    """Clean-dataset rows -> API requests. Rows the API would reject are skipped."""
    applicants, rejected = [], 0
    for row in df.to_dict(orient="records"):
        try:
            applicants.append(Applicant(**_request_fields({str(k): v for k, v in row.items()})))
        except (ValidationError, ValueError):
            rejected += 1
    return applicants, rejected


def _none_if_nan(v: Any) -> Any:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def _request_fields(row: dict[str, Any]) -> dict[str, Any]:
    income = _none_if_nan(row["monthly_income"])
    debt_ratio = float(row["debt_ratio"])
    coded = bool(row[c.DELINQUENCY_CODE_FLAG])
    lates = {col: 0 if coded else int(row[col]) for col in c.DELINQUENCY_COLS}
    return {
        "age": int(row["age"]),  # NaN age -> ValueError -> skipped
        "monthly_income": income,
        "monthly_debt_payments": debt_ratio * income if income else debt_ratio,
        "revolving_utilization": float(row["revolving_utilization"]),
        "open_credit_lines": int(row["open_credit_lines"]),
        "real_estate_loans": int(row["real_estate_loans"]),
        "dependents": None if _none_if_nan(row["dependents"]) is None else int(row["dependents"]),
        "delinquency_code_reported": coded,
        **lates,
    }


def simulate(
    settings: Settings, scenario: str, n: int, out: Path, seed: int = 0, batch: int = 500
) -> int:
    rng = np.random.default_rng(seed)
    test = pd.read_parquet(settings.processed_dir / "test.parquet").drop(columns=[c.TARGET])
    sample = test.sample(n=min(n, len(test)), random_state=seed).reset_index(drop=True)
    applicants, rejected = to_applicants(apply_scenario(sample, scenario, rng))

    out.unlink(missing_ok=True)
    service = ScoringService(CreditRiskModel.load(settings.models_dir), PredictionLog(out))
    for start in range(0, len(applicants), batch):
        service.score(applicants[start : start + batch], request_id=f"sim-{scenario}-{start}")
    log.info(
        "Scenario %s: scored %d applicants (%d rejected by API validation) -> %s",
        scenario,
        len(applicants),
        rejected,
        out,
    )
    return len(applicants)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate scored production traffic")
    parser.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    parser.add_argument("-n", type=int, default=5000, help="number of applicants")
    parser.add_argument("--out", type=Path, help="log file (default logs/simulated_<scenario>)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = args.out or PROJECT_ROOT / "logs" / f"simulated_{args.scenario}.jsonl"
    simulate(get_settings(), args.scenario, args.n, out, args.seed)


if __name__ == "__main__":
    main()
