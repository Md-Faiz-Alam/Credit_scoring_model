"""Probability calibration, applied only when the evidence says it's needed.

Expected-loss, pricing and capital calculations take the PD at face value, so a
predicted 5% must mean about 5 in 100 applicants default. Models trained on the natural
class balance usually meet this already. Recalibrating them anyway (isotonic in particular)
made results slightly worse in our experiments. So we measure ECE on the validation
split and fit a Platt (sigmoid) recalibration only when it's above tolerance.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from credit_scoring.evaluation.metrics import expected_calibration_error

log = logging.getLogger(__name__)


def calibrate_if_needed(
    estimator: Any, X_valid: pd.DataFrame, y_valid: pd.Series, max_ece: float
) -> tuple[Any, dict[str, Any]]:
    ece_before = expected_calibration_error(y_valid, estimator.predict_proba(X_valid)[:, 1])
    if ece_before <= max_ece:
        log.info("ECE %.4f <= %.4f: no recalibration needed", ece_before, max_ece)
        return estimator, {"calibrated": False, "ece_before": round(ece_before, 6)}

    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    calibrated.fit(X_valid, y_valid)
    ece_after = expected_calibration_error(y_valid, calibrated.predict_proba(X_valid)[:, 1])
    log.info("ECE %.4f > %.4f: Platt-recalibrated -> %.4f", ece_before, max_ece, ece_after)
    return calibrated, {
        "calibrated": True,
        "method": "sigmoid",
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
    }
