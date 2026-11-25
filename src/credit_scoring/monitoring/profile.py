"""Reference profile: a frozen snapshot of the training population.

Saved next to the model at training time (``models/reference_profile.json``). Production
monitoring compares live traffic against it, with no access to the training data
needed (in a bank, the serving environment usually can't read the modelling data).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_scoring.data import columns as c
from credit_scoring.monitoring.psi import bin_shares, quantile_edges

PROFILE_FILE = "reference_profile.json"
MONITORED_FEATURES = (*c.RAW_FEATURES, c.DELINQUENCY_CODE_FLAG)


@dataclass
class VariableProfile:
    edges: list[float]
    shares: list[float]  # len(edges) + 1 numeric bins, then the missing bin
    missing_rate: float
    mean: float | None


@dataclass
class ReferenceProfile:
    model_version: str
    rows: int
    score: VariableProfile
    approval_rate: float
    mean_pd: float
    features: dict[str, VariableProfile] = field(default_factory=dict)

    def save(self, directory: Path) -> Path:
        path = directory / PROFILE_FILE
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> "ReferenceProfile":
        raw: dict[str, Any] = json.loads((directory / PROFILE_FILE).read_text(encoding="utf-8"))
        return cls(
            model_version=raw["model_version"],
            rows=raw["rows"],
            score=VariableProfile(**raw["score"]),
            approval_rate=raw["approval_rate"],
            mean_pd=raw["mean_pd"],
            features={k: VariableProfile(**v) for k, v in raw["features"].items()},
        )


def profile_variable(values: np.ndarray, n_bins: int = 10) -> VariableProfile:
    v = np.asarray(values, dtype=float)
    edges = quantile_edges(v, n_bins)
    present = v[~np.isnan(v)]
    return VariableProfile(
        edges=edges,
        shares=[round(float(s), 6) for s in bin_shares(v, edges)],
        missing_rate=round(float(np.isnan(v).mean()), 6),
        mean=round(float(present.mean()), 6) if present.size else None,
    )


def build_profile(
    X: pd.DataFrame, scores: np.ndarray, pd_: np.ndarray, threshold: float, model_version: str
) -> ReferenceProfile:
    return ReferenceProfile(
        model_version=model_version,
        rows=len(X),
        score=profile_variable(scores.astype(float)),
        approval_rate=round(float((pd_ < threshold).mean()), 6),
        mean_pd=round(float(pd_.mean()), 6),
        features={f: profile_variable(X[f].to_numpy(dtype=float)) for f in MONITORED_FEATURES},
    )
