"""Scoring service: model in, decisions out. It holds no web framework code, so it's easy to test.

Every scored application is appended to a JSON-lines **prediction log** (model inputs,
outputs, model version). The drift monitor compares this live population against
the training data for drift. Only model features are logged, never the application id.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from credit_scoring.api.schemas import (
    Applicant,
    ModelInfo,
    ReasonOut,
    ScoreResponse,
    to_model_frame,
)
from credit_scoring.explain.reasons import REASON_GROUPS
from credit_scoring.models.bundle import CreditRiskModel


class PredictionLog:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: list[dict[str, object]]) -> None:
        if self.path is None or not records:
            return
        # Line-buffered append: each record reaches the OS as one small O_APPEND write,
        # so lines from several uvicorn worker processes never interleave mid-record.
        with self._lock, self.path.open("a", encoding="utf-8", buffering=1) as fh:
            for record in records:
                fh.write(json.dumps(record, default=str) + "\n")


class ScoringService:
    def __init__(self, model: CreditRiskModel, log: PredictionLog | None = None) -> None:
        self.model = model
        self.log = log or PredictionLog(None)
        self.version = str(model.metadata.get("model_version", "unknown"))

    def score(self, applicants: list[Applicant], request_id: str = "") -> list[ScoreResponse]:
        X = to_model_frame(applicants)
        preds = self.model.predict(X)
        explanations = self.model.explain(X)
        now = datetime.now(UTC)

        results: list[ScoreResponse] = []
        records: list[dict[str, object]] = []
        for i, (applicant, expl) in enumerate(zip(applicants, explanations, strict=True)):
            row = preds.iloc[i]
            results.append(
                ScoreResponse(
                    application_id=applicant.application_id,
                    probability_of_default=round(float(row["pd"]), 6),
                    score=int(row["score"]),
                    risk_grade=str(row["grade"]),
                    decision=row["decision"],
                    reasons=[
                        ReasonOut(code=r.code, message=r.message, impact=r.impact)
                        for r in expl.reasons
                    ],
                    model_version=self.version,
                    scored_at=now,
                )
            )
            records.append(
                {
                    "ts": now.isoformat(timespec="seconds"),
                    "request_id": request_id,
                    "model_version": self.version,
                    "features": {
                        k: (None if v != v else v)  # NaN -> null
                        for k, v in X.iloc[i].to_dict().items()
                    },
                    "pd": round(float(row["pd"]), 6),
                    "score": int(row["score"]),
                    "decision": row["decision"],
                }
            )
        self.log.write(records)
        return results

    def info(self) -> ModelInfo:
        meta = self.model.metadata
        test = meta.get("metrics", {}).get("test", {}).get("champion", {})
        return ModelInfo(
            model_version=self.version,
            champion=str(meta.get("champion", "unknown")),
            created_at=str(meta.get("created_at", "")),
            decision_threshold_pd=round(self.model.threshold, 6),
            decision_threshold_score=int(
                self.model.scaler.score(np.array([self.model.threshold]))[0]
            ),
            score_range=(self.model.scaler.min_score, self.model.scaler.max_score),
            risk_grades=dict(self.model.grades.bounds),
            test_metrics={k: float(v) for k, v in test.items()},
            reason_codes={g.code: g.message for g in REASON_GROUPS},
        )
