"""Typed, validated training configuration (loaded from configs/train.yaml)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_scoring.config import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureConfig(_Strict):
    cap_quantile: float = Field(0.99, gt=0.5, le=1)
    woe_max_bins: int = Field(6, ge=2, le=20)
    woe_min_bin_frac: float = Field(0.03, gt=0, lt=0.5)


class ScorecardConfig(_Strict):
    min_iv: float = Field(0.02, ge=0)
    max_correlation: float = Field(0.7, gt=0, le=1)
    C: float = Field(1.0, gt=0)


class LightGBMConfig(_Strict):
    n_trials: int = Field(30, ge=1)
    cv_folds: int = Field(5, ge=2)
    learning_rate: float = Field(0.05, gt=0, le=1)
    early_stopping_rounds: int = Field(100, ge=1)
    monotone: bool = True
    monotone_constraints: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _directions(self) -> "LightGBMConfig":
        bad = {k: v for k, v in self.monotone_constraints.items() if v not in (-1, 0, 1)}
        if bad:
            raise ValueError(f"monotone constraints must be -1, 0 or 1: {bad}")
        return self


class SelectionConfig(_Strict):
    min_auc_gain_for_challenger: float = Field(0.005, ge=0)


class CalibrationConfig(_Strict):
    max_ece: float = Field(0.01, gt=0)


class ScoreConfig(_Strict):
    base_score: float = 650
    base_odds: float = Field(30, gt=0)
    pdo: float = Field(40, gt=0)
    min_score: int = 300
    max_score: int = 850


class DecisionConfig(_Strict):
    cost_false_negative: float = Field(5.0, gt=0)
    cost_false_positive: float = Field(1.0, gt=0)


class QualityGateConfig(_Strict):
    min_valid_auc: float = Field(0.84, gt=0.5, le=1)
    max_valid_ece: float = Field(0.02, gt=0)


class TrainConfig(_Strict):
    features: FeatureConfig = FeatureConfig()
    scorecard: ScorecardConfig = ScorecardConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()
    selection: SelectionConfig = SelectionConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    score: ScoreConfig = ScoreConfig()
    risk_grades: dict[str, float] = Field(
        default_factory=lambda: {"A": 0.02, "B": 0.05, "C": 0.10, "D": 0.20, "E": 1.0}
    )
    decision: DecisionConfig = DecisionConfig()
    quality_gate: QualityGateConfig = QualityGateConfig()
    random_state: int = 42

    @model_validator(mode="after")
    def _grades_increasing(self) -> "TrainConfig":
        bounds = list(self.risk_grades.values())
        if bounds != sorted(bounds) or bounds[-1] != 1.0:
            raise ValueError("risk_grades must have increasing PD bounds ending at 1.0")
        return self

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "TrainConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})
