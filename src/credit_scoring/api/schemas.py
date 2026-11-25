"""Public API contract.

Field names are written for API clients, not copied from the dataset. In particular,
clients send their monthly debt *payments* and income separately. The training data's
``DebtRatio`` column, which is a ratio when income is known but a dollar amount when it
isn't, is rebuilt from those two fields in exactly one place, ``to_model_frame``.
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from credit_scoring.data import columns as c

MAX_BATCH = 1000


class Applicant(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "application_id": "APP-1001",
                    "age": 42,
                    "monthly_income": 5400,
                    "monthly_debt_payments": 1600,
                    "revolving_utilization": 0.35,
                    "open_credit_lines": 8,
                    "real_estate_loans": 1,
                    "dependents": 2,
                    "past_due_30_59": 0,
                    "past_due_60_89": 0,
                    "past_due_90_plus": 0,
                }
            ]
        },
    )

    application_id: str | None = Field(None, max_length=64, description="Echoed back")
    age: int = Field(..., ge=18, le=120, description="Applicant age in years")
    monthly_income: float | None = Field(
        None, ge=0, le=10_000_000, description="Gross monthly income (USD); null if unknown"
    )
    monthly_debt_payments: float = Field(
        ..., ge=0, le=10_000_000, description="Total monthly debt payments incl. mortgage (USD)"
    )
    revolving_utilization: float = Field(
        ...,
        ge=0,
        le=100,
        description="Revolving balances / credit limits as a ratio (0.35 = 35%)",
    )
    open_credit_lines: int = Field(..., ge=0, le=100)
    real_estate_loans: int = Field(..., ge=0, le=100)
    dependents: int | None = Field(None, ge=0, le=30)
    past_due_30_59: int = Field(0, ge=0, le=50, description="Times 30-59 days late (2 years)")
    past_due_60_89: int = Field(0, ge=0, le=50, description="Times 60-89 days late (2 years)")
    past_due_90_plus: int = Field(0, ge=0, le=50, description="Times 90+ days late")
    delinquency_code_reported: bool = Field(
        False, description="Bureau reported a special delinquency status instead of counts"
    )


def to_model_frame(applicants: list[Applicant]) -> pd.DataFrame:
    """Convert API inputs to the model's input columns (the cleaned-dataset format)."""
    rows = []
    for a in applicants:
        has_income = a.monthly_income is not None and a.monthly_income > 0
        debt_ratio = (
            a.monthly_debt_payments / a.monthly_income  # type: ignore[operator]
            if has_income
            else a.monthly_debt_payments  # dataset convention: amount when no income
        )
        coded = a.delinquency_code_reported
        rows.append(
            {
                "revolving_utilization": a.revolving_utilization,
                "age": a.age,
                # with a special code the bureau gives no usable counts (as in cleaning)
                "past_due_30_59": None if coded else a.past_due_30_59,
                "debt_ratio": debt_ratio,
                "monthly_income": a.monthly_income,
                "open_credit_lines": a.open_credit_lines,
                "past_due_90_plus": None if coded else a.past_due_90_plus,
                "real_estate_loans": a.real_estate_loans,
                "past_due_60_89": None if coded else a.past_due_60_89,
                "dependents": a.dependents,
                c.DELINQUENCY_CODE_FLAG: int(coded),
            }
        )
    return pd.DataFrame(rows, columns=[*c.RAW_FEATURES, c.DELINQUENCY_CODE_FLAG]).astype("float64")


class ReasonOut(BaseModel):
    code: str = Field(..., examples=["R02"])
    message: str
    impact: float = Field(..., description="Contribution to log-odds of default (> 0 = riskier)")


class ScoreResponse(BaseModel):
    application_id: str | None
    probability_of_default: float = Field(..., ge=0, le=1)
    score: int = Field(..., ge=300, le=850)
    risk_grade: str
    decision: Literal["approve", "decline"]
    reasons: list[ReasonOut] = Field(
        ..., description="Main factors increasing risk, strongest first (max 4)"
    )
    model_version: str
    scored_at: datetime


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applicants: list[Applicant] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    count: int
    approved: int
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    champion: str
    created_at: str
    decision_threshold_pd: float
    decision_threshold_score: int
    score_range: tuple[int, int]
    risk_grades: dict[str, float]
    test_metrics: dict[str, float]
    reason_codes: dict[str, str]
