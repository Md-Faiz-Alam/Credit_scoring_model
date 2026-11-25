"""Train, compare, calibrate and package the credit risk model.

Flow
  1. Fit the champion candidate (WoE logistic scorecard) and the challenger (LightGBM).
  2. Compare them on the validation split. The scorecard wins unless LightGBM's AUC gain
     reaches `selection.min_auc_gain_for_challenger`.
  3. Recalibrate the winner on validation, only if its ECE is above tolerance.
  4. Choose the cost-optimal PD cut-off on validation.
  5. Quality gate: stop if validation AUC or ECE miss the configured bar.
  6. Score the untouched test split once, for the reported numbers.
  7. Save models/credit_model.joblib (+ .json metadata) and reports/, and log it all
     to MLflow.

Usage:  uv run poe train                      (full run)
        uv run poe train --n-trials 5         (quick run)
"""

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from credit_scoring import __version__
from credit_scoring.config import PROJECT_ROOT, Settings, get_settings
from credit_scoring.evaluation.metrics import calibration_table, classification_summary
from credit_scoring.features.pipeline import split_xy
from credit_scoring.features.transformers import INPUT_COLUMNS
from credit_scoring.models.bundle import CreditRiskModel
from credit_scoring.models.calibrate import calibrate_if_needed
from credit_scoring.models.config import DEFAULT_CONFIG_PATH, TrainConfig
from credit_scoring.models.gbm import build_lightgbm
from credit_scoring.models.scorecard import build_scorecard, points_table
from credit_scoring.models.scoring import RiskGrades, ScoreScaler
from credit_scoring.models.threshold import (
    bayes_threshold,
    cost_optimal_threshold,
    expected_cost,
    strategy_table,
)
from credit_scoring.monitoring.profile import build_profile

log = logging.getLogger(__name__)

EXPERIMENT = "credit-scoring"


class QualityGateError(RuntimeError):
    """The trained model doesn't meet the minimum standard and must not be shipped."""


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_splits(settings: Settings) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(settings.processed_dir / f"{name}.parquet")
        for name in ("train", "valid", "test")
    }


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flatten(v, f"{key}.")
        else:
            out[key] = v
    return out


def train(settings: Settings, cfg: TrainConfig, *, tracking_uri: str | None = None) -> Path:
    splits = _load_splits(settings)
    X_train, y_train = split_xy(splits["train"])
    X_valid, y_valid = split_xy(splits["valid"])
    X_test, y_test = split_xy(splits["test"])
    manifest = json.loads((settings.processed_dir / "manifest.json").read_text())
    reports = settings.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        mlflow.set_tags({"git_commit": _git_commit(), "data_sha256": manifest["raw_sha256"]})
        mlflow.log_params(_flatten(cfg.model_dump()))

        # 1. candidates -------------------------------------------------------------
        log.info("Fitting scorecard (WoE + logistic regression)")
        scorecard, dropped = build_scorecard(X_train, y_train, cfg)
        log.info("Tuning LightGBM (%d trials)", cfg.lightgbm.n_trials)
        lightgbm, gbm_info = build_lightgbm(X_train, y_train, cfg)

        candidates = {"scorecard": scorecard, "lightgbm": lightgbm}
        valid_metrics = {
            name: classification_summary(y_valid, est.predict_proba(X_valid)[:, 1])
            for name, est in candidates.items()
        }
        comparison = pd.DataFrame(valid_metrics).T
        log.info("Validation comparison:\n%s", comparison.to_string())
        for name, metrics in valid_metrics.items():
            mlflow.log_metrics({f"valid.{name}.{k}": v for k, v in metrics.items()})
        mlflow.log_metric("lightgbm.cv_auc", gbm_info["cv_auc"])

        # 2. champion selection -----------------------------------------------------
        gain = valid_metrics["lightgbm"]["auc"] - valid_metrics["scorecard"]["auc"]
        champion = "lightgbm" if gain >= cfg.selection.min_auc_gain_for_challenger else "scorecard"
        log.info("Champion: %s (LightGBM AUC gain %+.4f)", champion, gain)

        # 3. calibration ------------------------------------------------------------
        estimator, calib_info = calibrate_if_needed(
            candidates[champion], X_valid, y_valid, cfg.calibration.max_ece
        )
        pd_valid = estimator.predict_proba(X_valid)[:, 1]

        # 4. decision threshold -----------------------------------------------------
        dc = cfg.decision
        c_fn, c_fp = dc.cost_false_negative, dc.cost_false_positive
        threshold = bayes_threshold(c_fn, c_fp)
        cost = expected_cost(y_valid, pd_valid, threshold, c_fn, c_fp)
        empirical_t, empirical_cost = cost_optimal_threshold(y_valid, pd_valid, c_fn, c_fp)
        strategy = strategy_table(y_valid, pd_valid, c_fn, c_fp)
        log.info(
            "PD cut-off %.4f (expected cost %.4f / applicant); empirical optimum %.4f (%.4f)",
            threshold,
            cost,
            empirical_t,
            empirical_cost,
        )
        mlflow.log_metrics(
            {
                "threshold": threshold,
                "valid.expected_cost": cost,
                "valid.approval_rate": float((pd_valid < threshold).mean()),
            }
        )

        # 5. quality gate -----------------------------------------------------------
        final_valid = classification_summary(y_valid, pd_valid)
        qg = cfg.quality_gate
        failures = []
        if final_valid["auc"] < qg.min_valid_auc:
            failures.append(f"valid AUC {final_valid['auc']:.4f} < {qg.min_valid_auc}")
        if final_valid["ece"] > qg.max_valid_ece:
            failures.append(f"valid ECE {final_valid['ece']:.4f} > {qg.max_valid_ece}")
        mlflow.set_tag("quality_gate", "failed" if failures else "passed")
        if failures:
            raise QualityGateError("; ".join(failures))

        # 6. test (reported once, never used for decisions) -------------------------
        test_metrics = {
            name: classification_summary(y_test, est.predict_proba(X_test)[:, 1])
            for name, est in {**candidates, "champion": estimator}.items()
        }
        mlflow.log_metrics({f"test.champion.{k}": v for k, v in test_metrics["champion"].items()})

        # 7. package ----------------------------------------------------------------
        scaler = ScoreScaler(**cfg.score.model_dump())
        model_version = f"{datetime.now(UTC):%Y.%m.%d}-{run.info.run_id[:8]}"
        bundle = CreditRiskModel(
            estimator=estimator,
            scaler=scaler,
            grades=RiskGrades(dict(cfg.risk_grades)),
            threshold=threshold,
            # Only a linear champion needs a background sample for explanations; otherwise
            # don't ship training rows inside the model artefact.
            background=(
                X_train.sample(1000, random_state=cfg.random_state)
                if champion == "scorecard"
                else None
            ),
            metadata={
                "model_version": model_version,
                "champion": champion,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "code_version": __version__,
                "git_commit": _git_commit(),
                "mlflow_run_id": run.info.run_id,
                "data": {k: manifest[k] for k in ("raw_file", "raw_sha256", "splits")},
                "input_features": list(INPUT_COLUMNS),
                "threshold": {
                    "pd": round(threshold, 6),
                    "method": "bayes: cost_fp / (cost_fp + cost_fn)",
                    "expected_cost_valid": round(cost, 6),
                    "empirical_optimum_valid": round(empirical_t, 6),
                    "approval_rate_valid": round(float((pd_valid < threshold).mean()), 4),
                },
                "calibration": calib_info,
                "metrics": {"valid": valid_metrics, "test": test_metrics},
                "scorecard_dropped_features": dropped,
                "lightgbm": {k: v for k, v in gbm_info.items() if k != "monotone_constraints"},
                "config": cfg.model_dump(),
            },
        )
        model_path = bundle.save(settings.models_dir)
        train_pred = bundle.predict(X_train)
        build_profile(
            X_train,
            train_pred["score"].to_numpy(),
            train_pred["pd"].to_numpy(),
            threshold,
            model_version,
        ).save(settings.models_dir)

        comparison.to_csv(reports / "model_comparison.csv")
        points_table(scorecard, scaler).to_csv(reports / "scorecard_points.csv", index=False)
        strategy.to_csv(reports / "strategy_table.csv", index=False)
        calibration_table(y_valid, pd_valid).to_csv(reports / "calibration_valid.csv", index=False)
        for f in (
            "model_comparison.csv",
            "scorecard_points.csv",
            "strategy_table.csv",
            "calibration_valid.csv",
        ):
            mlflow.log_artifact(str(reports / f), artifact_path="reports")
        mlflow.log_artifact(str(settings.models_dir / "credit_model.json"), artifact_path="model")

        log.info("Saved %s (version %s)", model_path, model_version)
        log.info("Test metrics (champion): %s", test_metrics["champion"])
        return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the credit risk model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-trials", type=int, help="override lightgbm.n_trials")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = TrainConfig.from_yaml(args.config)
    if args.n_trials is not None:
        cfg = cfg.model_copy(
            update={"lightgbm": cfg.lightgbm.model_copy(update={"n_trials": args.n_trials})}
        )
    train(get_settings(), cfg)


if __name__ == "__main__":
    main()
