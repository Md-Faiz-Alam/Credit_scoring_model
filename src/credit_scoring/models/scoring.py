"""PD -> credit score -> risk grade -> decision.

Score scaling uses the industry-standard "points to double the odds" (PDO) method:

    score = offset + factor * ln(odds_good),   odds_good = (1 - PD) / PD
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

With the default config, an applicant at 30:1 odds (PD ~3.2%) scores 650. Every
+40 points doubles the odds of repayment. Scores are clipped to 300-850.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Decision = Literal["approve", "decline"]

PD_EPS = 1e-6


@dataclass(frozen=True)
class ScoreScaler:
    base_score: float = 650
    base_odds: float = 30
    pdo: float = 40
    min_score: int = 300
    max_score: int = 850

    @property
    def factor(self) -> float:
        return float(self.pdo / np.log(2))

    @property
    def offset(self) -> float:
        return float(self.base_score - self.factor * np.log(self.base_odds))

    def score(self, pd_: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(pd_, dtype=float), PD_EPS, 1 - PD_EPS)
        raw = self.offset + self.factor * np.log((1 - p) / p)
        return np.asarray(np.clip(np.rint(raw), self.min_score, self.max_score).astype(int))


@dataclass(frozen=True)
class RiskGrades:
    """Ordered grades with upper PD bounds, e.g. {"A": 0.02, ..., "E": 1.0}."""

    bounds: dict[str, float] = field(
        default_factory=lambda: {"A": 0.02, "B": 0.05, "C": 0.10, "D": 0.20, "E": 1.0}
    )

    def grade(self, pd_: np.ndarray) -> np.ndarray:
        labels = np.array(list(self.bounds))
        edges = np.array(list(self.bounds.values()))
        idx = np.searchsorted(edges, np.asarray(pd_, dtype=float), side="left")
        return np.asarray(labels[np.minimum(idx, len(labels) - 1)])


def decide(pd_: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.asarray(pd_) >= threshold, "decline", "approve")
