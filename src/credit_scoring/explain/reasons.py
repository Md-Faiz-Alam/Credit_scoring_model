"""Customer-facing reason codes.

When a lender declines an applicant, it generally has to state the main reasons
("adverse action" notices under the US ECOA / Regulation B; similar fair-practice rules
exist elsewhere). Up to four reasons are typical.

Several engineered features describe the same underlying fact (four columns describe
late payments). Listing "past_due_30_59" and "total_past_due" as two separate reasons
would be confusing, so features are first grouped into **reason groups**, their
contributions summed, and each group gets a single stable code and plain-language text.
"""

from dataclasses import dataclass

from credit_scoring.data import columns as c


@dataclass(frozen=True)
class ReasonGroup:
    code: str
    message: str
    features: tuple[str, ...]


REASON_GROUPS: tuple[ReasonGroup, ...] = (
    ReasonGroup(
        "R01",
        "Serious delinquency: late payments on credit accounts",
        (
            "past_due_30_59",
            "past_due_60_89",
            "past_due_90_plus",
            "total_past_due",
            c.DELINQUENCY_CODE_FLAG,
        ),
    ),
    ReasonGroup(
        "R02",
        "Balances on revolving credit are high relative to credit limits",
        ("revolving_utilization", "utilization_implausible"),
    ),
    ReasonGroup(
        "R03",
        "Debt obligations are high relative to income",
        ("debt_ratio", "monthly_debt"),
    ),
    ReasonGroup(
        "R04",
        "Income is low, not reported, or low relative to household size",
        ("monthly_income", "no_income_reported", "income_per_person"),
    ),
    ReasonGroup(
        "R05",
        "Age of applicant (proxy for length of credit history)",
        ("age",),
    ),
    ReasonGroup(
        "R06",
        "Number of open credit lines and loans",
        ("open_credit_lines",),
    ),
    ReasonGroup(
        "R07",
        "Number of real-estate loans or lines",
        ("real_estate_loans",),
    ),
    ReasonGroup(
        "R08",
        "Number of dependents",
        ("dependents",),
    ),
)

FEATURE_TO_GROUP: dict[str, ReasonGroup] = {f: g for g in REASON_GROUPS for f in g.features}
