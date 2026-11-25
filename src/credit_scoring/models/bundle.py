"""The deployable model: one object from input features to a lending decision.

Everything needed to score an applicant travels together: the feature pipeline, the
estimator (plus calibrator), score scaling, risk grades, the decision threshold and
lineage metadata. Training and the API load the same object, so there's no
hand-copied logic to drift apart.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from credit_scoring.models.scoring import RiskGrades, ScoreScaler, decide

if TYPE_CHECKING:
    from credit_scoring.explain.explainer import Explanation, ReasonCodeExplainer

MODEL_FILE = "credit_model.joblib"
METADATA_FILE = "credit_model.json"


@dataclass
class CreditRiskModel:
    estimator: Any  # sklearn-compatible, with predict_proba
    scaler: ScoreScaler
    grades: RiskGrades
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # Small sample of training inputs; linear-model explanations are relative to it
    background: pd.DataFrame | None = None
    _explainer: Any = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_explainer"] = None  # rebuilt on demand, never pickled
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("background", None)  # models saved before this field existed
        state.setdefault("_explainer", None)
        self.__dict__.update(state)

    @property
    def explainer(self) -> "ReasonCodeExplainer":
        if self._explainer is None:
            from credit_scoring.explain.explainer import ReasonCodeExplainer

            self._explainer = ReasonCodeExplainer(self.estimator, self.background)
        explainer: ReasonCodeExplainer = self._explainer
        return explainer

    def explain(self, X: pd.DataFrame, max_reasons: int = 4) -> list["Explanation"]:
        return self.explainer.explain(X, max_reasons=max_reasons)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X)[:, 1])

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pd_ = self.predict_pd(X)
        return pd.DataFrame(
            {
                "pd": pd_,
                "score": self.scaler.score(pd_),
                "grade": self.grades.grade(pd_),
                "decision": decide(pd_, self.threshold),
            },
            index=X.index,
        )

    # ------------------------------------------------------------------ persistence

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_FILE
        joblib.dump(self, path, compress=3)
        (directory / METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> "CreditRiskModel":
        obj = joblib.load(directory / MODEL_FILE)
        if not isinstance(obj, cls):
            raise TypeError(f"{directory / MODEL_FILE} doesn't contain a CreditRiskModel")
        return obj
