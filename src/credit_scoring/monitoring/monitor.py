"""Drift monitoring job: prediction log vs. training reference profile.

Checks
  * Score PSI: has the risk mix of applicants moved?
  * Feature CSI: which inputs moved? This explains a score shift.
  * Missing-value rates: upstream data pipes breaking often show up here first.
  * Approval rate and mean PD vs. training.

Writes reports/monitoring/{report.md, report.json, *.png}. With ``--publish`` it also
writes to docs/monitoring/ so a report can be committed. The exit code is 1 when the
overall status reaches ``--fail-on``, so a scheduler or CI job can alert on it.

Labels (did the applicant actually default?) arrive 12-24 months later, so these
leading indicators are what a credit team watches in the meantime.

Usage:  uv run poe monitor --log logs/predictions.jsonl --fail-on action
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_scoring.config import PROJECT_ROOT, Settings, get_settings
from credit_scoring.evaluation import plots
from credit_scoring.features.pipeline import split_xy
from credit_scoring.models.bundle import CreditRiskModel
from credit_scoring.monitoring.profile import (
    PROFILE_FILE,
    ReferenceProfile,
    build_profile,
)
from credit_scoring.monitoring.psi import ACTION_AT, MONITOR_AT, Status, bin_shares, psi

log = logging.getLogger(__name__)

SEVERITY: dict[str, int] = {"stable": 0, "monitor": 1, "action": 2}
MISSING_RATE_ALERT = 0.05  # absolute change in missing share that counts as a data issue

# Business KPIs. PSI is insensitive to a moderate shift spread evenly across all bins:
# in a simulated downturn every PSI stayed < 0.1 while mean PD rose 41%. So outcome KPIs
# are checked too.
APPROVAL_RATE_CHANGE = {"monitor": 0.03, "action": 0.05}  # absolute change
MEAN_PD_CHANGE = {"monitor": 0.15, "action": 0.30}  # relative change


def _kpi_status(change: float, limits: dict[str, float]) -> Status:
    if abs(change) >= limits["action"]:
        return "action"
    if abs(change) >= limits["monitor"]:
        return "monitor"
    return "stable"


@dataclass
class VariableDrift:
    name: str
    psi: float
    status: Status
    missing_rate_ref: float
    missing_rate_now: float
    mean_ref: float | None
    mean_now: float | None


@dataclass
class DriftReport:
    generated_at: str
    log_file: str
    rows: int
    window_start: str | None
    window_end: str | None
    model_version_reference: str
    model_versions_seen: list[str]
    status: Status
    score: VariableDrift
    approval_rate_ref: float
    approval_rate_now: float
    approval_status: Status
    mean_pd_ref: float
    mean_pd_now: float
    mean_pd_status: Status
    features: list[VariableDrift] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


def read_log(path: Path, since: datetime | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction log not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"Prediction log is empty: {path}")
    features = pd.DataFrame([r["features"] for r in records]).astype("float64")
    meta = pd.DataFrame(
        {
            "ts": pd.to_datetime([r["ts"] for r in records], utc=True),
            "model_version": [r["model_version"] for r in records],
            "pd": [r["pd"] for r in records],
            "score": [r["score"] for r in records],
            "decision": [r["decision"] for r in records],
        }
    )
    df = pd.concat([meta, features], axis=1)
    if since is not None:
        df = df[df["ts"] >= pd.Timestamp(since)]
    return df.reset_index(drop=True)


def load_or_build_profile(settings: Settings) -> ReferenceProfile:
    """Load the profile saved at training. Older models trained before profiles existed
    get one built from the training split."""
    if (settings.models_dir / PROFILE_FILE).exists():
        return ReferenceProfile.load(settings.models_dir)
    log.warning("No reference profile next to the model; building it from the training split")
    model = CreditRiskModel.load(settings.models_dir)
    X, _ = split_xy(pd.read_parquet(settings.processed_dir / "train.parquet"))
    pred = model.predict(X)
    profile = build_profile(
        X,
        pred["score"].to_numpy(),
        pred["pd"].to_numpy(),
        model.threshold,
        str(model.metadata.get("model_version", "unknown")),
    )
    profile.save(settings.models_dir)
    return profile


def _drift(name: str, ref: Any, values: np.ndarray) -> VariableDrift:
    present = values[~np.isnan(values)]
    result = psi(np.asarray(ref.shares), bin_shares(values, ref.edges))
    return VariableDrift(
        name=name,
        psi=result.psi,
        status=result.status,
        missing_rate_ref=ref.missing_rate,
        missing_rate_now=round(float(np.isnan(values).mean()), 6),
        mean_ref=ref.mean,
        mean_now=round(float(present.mean()), 6) if present.size else None,
    )


def compare(profile: ReferenceProfile, df: pd.DataFrame, log_file: str) -> DriftReport:
    score = _drift("score", profile.score, df["score"].to_numpy(dtype=float))
    features = sorted(
        (_drift(f, ref, df[f].to_numpy(dtype=float)) for f, ref in profile.features.items()),
        key=lambda d: d.psi,
        reverse=True,
    )
    approval_now = float((df["decision"] == "approve").mean())
    mean_pd_now = float(df["pd"].mean())

    alerts: list[str] = []
    if score.status != "stable":
        alerts.append(f"Score PSI {score.psi:.3f} ({score.status}): applicant risk mix has shifted")
    for d in features:
        if d.status != "stable":
            alerts.append(f"{d.name}: CSI {d.psi:.3f} ({d.status})")
        if abs(d.missing_rate_now - d.missing_rate_ref) >= MISSING_RATE_ALERT:
            alerts.append(
                f"{d.name}: missing rate {d.missing_rate_ref:.1%} -> {d.missing_rate_now:.1%} "
                "(check the upstream data feed)"
            )
    versions = sorted(df["model_version"].unique().tolist())
    if versions != [profile.model_version]:
        alerts.append(
            f"Log contains model versions {versions}; reference profile is for "
            f"{profile.model_version}"
        )

    approval_status = _kpi_status(approval_now - profile.approval_rate, APPROVAL_RATE_CHANGE)
    pd_change = mean_pd_now / profile.mean_pd - 1 if profile.mean_pd else 0.0
    mean_pd_status = _kpi_status(pd_change, MEAN_PD_CHANGE)
    if approval_status != "stable":
        alerts.insert(
            0,
            f"Approval rate {profile.approval_rate:.1%} -> {approval_now:.1%} ({approval_status})",
        )
    if mean_pd_status != "stable":
        alerts.insert(
            0,
            f"Mean PD {profile.mean_pd:.2%} -> {mean_pd_now:.2%} "
            f"({pd_change:+.0%}, {mean_pd_status}): portfolio risk has moved",
        )

    statuses = [score.status, *(d.status for d in features), approval_status, mean_pd_status]
    worst: Status = max(statuses, key=lambda st: SEVERITY[st])
    return DriftReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        log_file=log_file,
        rows=len(df),
        window_start=df["ts"].min().isoformat() if len(df) else None,
        window_end=df["ts"].max().isoformat() if len(df) else None,
        model_version_reference=profile.model_version,
        model_versions_seen=versions,
        status=worst,
        score=score,
        approval_rate_ref=profile.approval_rate,
        approval_rate_now=round(approval_now, 6),
        approval_status=approval_status,
        mean_pd_ref=profile.mean_pd,
        mean_pd_now=round(mean_pd_now, 6),
        mean_pd_status=mean_pd_status,
        features=features,
        alerts=alerts,
    )


_BADGE = {"stable": "🟢 STABLE", "monitor": "🟡 MONITOR", "action": "🔴 ACTION"}


def to_markdown(r: DriftReport) -> str:
    rows = "\n".join(
        f"| {d.name} | {d.psi:.3f} | {_BADGE[d.status]} | {d.missing_rate_ref:.1%} -> "
        f"{d.missing_rate_now:.1%} | {_fmt(d.mean_ref)} -> {_fmt(d.mean_now)} |"
        for d in [r.score, *r.features]
    )
    alerts = "\n".join(f"* {a}" for a in r.alerts) or "* None"
    return f"""# Drift monitoring report

**Overall status: {_BADGE[r.status]}**

| | |
|---|---|
| Generated | {r.generated_at} |
| Prediction log | `{r.log_file}` ({r.rows:,} scored applications) |
| Window | {r.window_start} to {r.window_end} |
| Reference | training population, model `{r.model_version_reference}` |
| Approval rate | {r.approval_rate_ref:.1%} (training) -> **{r.approval_rate_now:.1%}** (current) {_BADGE[r.approval_status]} |
| Mean PD | {r.mean_pd_ref:.2%} (training) -> **{r.mean_pd_now:.2%}** (current) {_BADGE[r.mean_pd_status]} |

## Alerts

{alerts}

## Stability by variable

PSI < {MONITOR_AT} stable · {MONITOR_AT}-{ACTION_AT} monitor · > {ACTION_AT} action.

| variable | PSI | status | missing rate | mean |
|---|---|---|---|---|
{rows}

| | |
|---|---|
| ![Drift by variable](figures/stability.png) | ![Score shift](figures/score_shift.png) |

KPI limits: approval rate change ≥ {APPROVAL_RATE_CHANGE["monitor"]:.0%} monitor / ≥ {APPROVAL_RATE_CHANGE["action"]:.0%} action (absolute);
mean PD change ≥ {MEAN_PD_CHANGE["monitor"]:.0%} monitor / ≥ {MEAN_PD_CHANGE["action"]:.0%} action (relative).

## How to read this

* **Score PSI** says *whether* the applicant risk mix moved. **Feature CSI** says *why*.
* **KPIs** catch what PSI misses. A uniform shift across all score bands (a recession)
  can keep every PSI low while mean PD and approvals move sharply.
* Drift isn't automatically a model problem. If the inputs moved but the model's
  relationships still hold, PDs stay calibrated. Confirm with realised defaults when
  labels mature, before retraining.
* A jump in a **missing rate** usually means an upstream data issue, not a real
  population change. Fix the feed before touching the model.
"""


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:,.3g}"


def run(
    settings: Settings,
    log_file: Path,
    out_dirs: list[Path],
    since: datetime | None = None,
) -> DriftReport:
    profile = load_or_build_profile(settings)
    df = read_log(log_file, since)
    report = compare(profile, df, log_file.name)
    current_score_shares = bin_shares(df["score"].to_numpy(dtype=float), profile.score.edges)
    for out in out_dirs:
        figs = out / "figures"
        plots.stability_chart(
            {d.name: d.psi for d in [report.score, *report.features]},
            figs / "stability.png",
            MONITOR_AT,
            ACTION_AT,
        )
        plots.score_shift_chart(
            profile.score.shares,
            [float(x) for x in current_score_shares],
            profile.score.edges,
            figs / "score_shift.png",
        )
        (out / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Drift monitoring: prediction log vs training")
    parser.add_argument("--log", type=Path, default=settings.prediction_log)
    parser.add_argument("--since", type=datetime.fromisoformat, help="ISO date, e.g. 2026-11-01")
    parser.add_argument("--fail-on", choices=["never", "monitor", "action"], default="never")
    parser.add_argument("--publish", action="store_true", help="also write docs/monitoring/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.log is None:
        parser.error("no prediction log configured; pass --log")

    out_dirs = [settings.reports_dir / "monitoring"]
    if args.publish:
        out_dirs.append(PROJECT_ROOT / "docs" / "monitoring")
    report = run(settings, args.log, out_dirs, args.since)

    log.info("Status: %s  (score PSI %.3f, %d rows)", report.status, report.score.psi, report.rows)
    for alert in report.alerts:
        log.warning(alert)
    log.info("Report: %s", out_dirs[-1] / "report.md")
    if args.fail_on != "never" and SEVERITY[report.status] >= SEVERITY[args.fail_on]:
        sys.exit(1)


if __name__ == "__main__":
    main()
