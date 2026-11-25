"""Challenger: LightGBM tuned with Optuna, using monotonic constraints.

* **No class reweighting, no SMOTE.** Training on the natural 6.7% default rate gives
  probabilities that are already calibrated. Reweighting left ranking (AUC) unchanged in
  our experiments but inflated the mean PD from 6.6% to 18%.
* **Monotonic constraints** (configs/train.yaml) guarantee that, all else equal, more
  delinquency or higher utilization can never *lower* predicted risk. This costs about
  0.001 AUC and makes every explanation given to a customer consistent.
* Hyperparameters are tuned with 5-fold stratified CV **on the training split only**, with
  early stopping picking the number of trees. The validation split stays clean for model
  selection and calibration.
"""

import logging
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_scoring.features.transformers import CreditFeatureEngineer
from credit_scoring.models.config import TrainConfig

log = logging.getLogger(__name__)


def _base_params(cfg: TrainConfig, columns: list[str]) -> dict[str, Any]:
    gc = cfg.lightgbm
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": gc.learning_rate,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": cfg.random_state,
        "deterministic": True,
        "force_row_wise": True,
    }
    if gc.monotone:
        params["monotone_constraints"] = [gc.monotone_constraints.get(c, 0) for c in columns]
        params["monotone_constraints_method"] = "advanced"
    return params


def tune(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> tuple[dict[str, Any], int, float]:
    """Return (best params, best number of trees, best CV AUC)."""
    gc = cfg.lightgbm
    base = _base_params(cfg, list(X.columns))
    # feature_pre_filter=False: trials vary min_child_samples on the same Dataset
    data = lgb.Dataset(X, y, params={"feature_pre_filter": False}, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base,
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        }
        cv = lgb.cv(
            params,
            data,
            num_boost_round=3000,
            nfold=gc.cv_folds,
            stratified=True,
            seed=cfg.random_state,
            callbacks=[lgb.early_stopping(gc.early_stopping_rounds, verbose=False)],
        )
        scores: list[float] = cv["valid auc-mean"]  # type: ignore[assignment]
        trial.set_user_attr("n_trees", len(scores))
        return float(max(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_state)
    )
    study.optimize(objective, n_trials=gc.n_trials, show_progress_bar=False)

    best = study.best_trial
    log.info("LightGBM best CV AUC %.4f with %s", best.value, best.params)
    return {**base, **best.params}, int(best.user_attrs["n_trees"]), float(best.value or 0.0)


def build_lightgbm(
    X: pd.DataFrame, y: pd.Series, cfg: TrainConfig
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune, then fit the final pipeline (engineer -> LightGBM) on all training data."""
    engineer = CreditFeatureEngineer(cap_quantile=cfg.features.cap_quantile).set_output(
        transform="pandas"
    )
    X_eng = engineer.fit_transform(X)
    params, n_trees, cv_auc = tune(X_eng, y, cfg)

    sk_params = {k: v for k, v in params.items() if k not in ("objective", "metric")}
    model = lgb.LGBMClassifier(n_estimators=n_trees, **sk_params).fit(X_eng, y)
    pipe = Pipeline([("engineer", engineer), ("model", model)])
    return pipe, {"cv_auc": cv_auc, "n_trees": n_trees, **params}
