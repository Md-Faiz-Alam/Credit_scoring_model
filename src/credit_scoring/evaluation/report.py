"""Evaluate the trained model and generate its model card.

Outputs
  docs/model_card.md           human-readable model card (tracked in git)
  docs/figures/*.png           charts referenced by the card and README
  reports/lift_table.csv       decile lift / gains
  reports/feature_importance.csv
  reports/fairness_age.csv     per age band, with and without the age feature
  reports/evaluation.json      every number in the card, machine-readable

All headline numbers come from the **test** split, which training never touched.

Usage:  uv run poe evaluate
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_scoring.config import PROJECT_ROOT, Settings, get_settings
from credit_scoring.evaluation import plots
from credit_scoring.evaluation.fairness import age_band, group_metrics, without_feature
from credit_scoring.evaluation.lift import decile_table
from credit_scoring.evaluation.metrics import calibration_table, classification_summary
from credit_scoring.features.pipeline import split_xy
from credit_scoring.models.bundle import CreditRiskModel
from credit_scoring.models.config import TrainConfig
from credit_scoring.models.scorecard import build_scorecard
from credit_scoring.models.threshold import strategy_table

log = logging.getLogger(__name__)

DOCS_DIR = PROJECT_ROOT / "docs"


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _md_table(df: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            v = row[col]
            fmt = formats.get(col)
            if fmt and isinstance(v, int | float | np.floating | np.integer) and not pd.isna(v):
                cells.append(fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def evaluate(settings: Settings, docs_dir: Path = DOCS_DIR) -> dict[str, Any]:
    model = CreditRiskModel.load(settings.models_dir)
    meta = model.metadata
    cfg = TrainConfig.model_validate(meta["config"])
    train = pd.read_parquet(settings.processed_dir / "train.parquet")
    test = pd.read_parquet(settings.processed_dir / "test.parquet")
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    y = y_test.to_numpy()
    figs = docs_dir / "figures"
    reports = settings.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    # predictions --------------------------------------------------------------
    pd_test = model.predict_pd(X_test)
    scores = model.scaler.score(pd_test)
    approved = pd_test < model.threshold
    # The deployed bundle holds only the champion. The scorecard is cheap and
    # deterministic, so it's refit here for the comparison chart.
    scorecard, _ = build_scorecard(X_train, y_train, cfg)
    pd_scorecard = scorecard.predict_proba(X_test)[:, 1]
    champion_name = f"{meta['champion']} (champion)"
    curves = {champion_name: pd_test}
    if meta["champion"] != "scorecard":
        curves["scorecard"] = pd_scorecard

    metrics = classification_summary(y, pd_test)
    deciles = decile_table(y, pd_test)
    calib = calibration_table(y, pd_test)
    c_fn, c_fp = cfg.decision.cost_false_negative, cfg.decision.cost_false_positive
    strategy = strategy_table(y, pd_test, c_fn, c_fp)
    decision = {
        "pd_cutoff": model.threshold,
        "score_cutoff": int(model.scaler.score(np.array([model.threshold]))[0]),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y[approved].mean()),
        "bad_rate_all": float(y.mean()),
        "defaults_declined": float((~approved & (y == 1)).sum() / (y == 1).sum()),
        "goods_declined": float((~approved & (y == 0)).sum() / (y == 0).sum()),
    }

    # explainability -----------------------------------------------------------
    importance = model.explainer.global_importance(X_test)
    declined_idx = X_test.index[~approved][:3]
    examples = []
    for idx, expl in zip(declined_idx, model.explain(X_test.loc[declined_idx]), strict=True):
        examples.append(
            {
                "pd": float(pd_test[X_test.index.get_loc(idx)]),
                "score": int(scores[X_test.index.get_loc(idx)]),
                "inputs": X_test.loc[idx].to_dict(),
                "reasons": [(r.code, r.message, r.impact) for r in expl.reasons],
            }
        )

    # fairness -----------------------------------------------------------------
    bands = age_band(X_test["age"])
    fair_with = group_metrics(y, pd_test, model.threshold, bands)
    no_age = without_feature(
        X_train,
        y_train,
        meta["lightgbm"],
        "age",
        cfg.features.cap_quantile,
        cfg.lightgbm.monotone_constraints,
    )
    pd_no_age = no_age.predict_proba(X_test)[:, 1]
    fair_without = group_metrics(y, pd_no_age, model.threshold, bands)
    auc_no_age = classification_summary(y, pd_no_age)["auc"]

    # write artefacts ----------------------------------------------------------
    deciles.to_csv(reports / "lift_table.csv", index=False)
    importance.to_csv(reports / "feature_importance.csv", index=False)
    pd.concat(
        [fair_with.assign(model="with_age"), fair_without.assign(model="without_age")]
    ).to_csv(reports / "fairness_age.csv", index=False)

    plots.roc_chart(y, curves, figs / "roc.png")
    plots.calibration_chart(calib, figs / "calibration.png")
    plots.gains_chart(deciles, figs / "gains.png")
    plots.importance_chart(importance, figs / "feature_importance.png")
    plots.score_distribution_chart(
        scores, y, decision["score_cutoff"], figs / "score_distribution.png"
    )
    plots.fairness_chart(fair_with, fair_without, figs / "fairness_age.png")

    result = {
        "model_version": meta["model_version"],
        "test_metrics": metrics,
        "scorecard_test_metrics": classification_summary(y, pd_scorecard),
        "decision": decision,
        "fairness": {
            "auc_without_age": auc_no_age,
            "with_age": fair_with.to_dict(orient="records"),
            "without_age": fair_without.to_dict(orient="records"),
        },
    }
    (reports / "evaluation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    card = _model_card(
        meta, cfg, metrics, result, deciles, strategy, decision, importance, examples,
        fair_with, fair_without, auc_no_age,
    )  # fmt: skip
    (docs_dir / "model_card.md").write_text(card, encoding="utf-8")
    log.info("Wrote %s", docs_dir / "model_card.md")
    return result


def _fairness_finding(fair_with: pd.DataFrame, fair_without: pd.DataFrame) -> str:
    """Narrative generated from the numbers, so it stays true after retraining."""
    w = fair_with.set_index("group")
    wo = fair_without.set_index("group")
    oldest, youngest = w.index[-1], w.index[0]
    return (
        f"* With age, predicted PD tracks the actual default rate in every band "
        f"(largest gap {w['calibration_gap'].abs().max() * 100:.1f} pp). "
        f"Approval-rate differences follow real differences in default rates "
        f"({_pct(w.loc[youngest, 'default_rate'])} for {youngest} vs "
        f"{_pct(w.loc[oldest, 'default_rate'])} for {oldest}). Lowest adverse impact "
        f"ratio: {w['adverse_impact_ratio'].min():.2f} "
        f"({'passes' if w['passes_four_fifths'].all() else 'fails'} the four-fifths rule).\n"
        f"* Removing age raises the lowest adverse impact ratio to "
        f"{wo['adverse_impact_ratio'].min():.2f}, but the model loses calibration by group. It "
        f"understates risk for {youngest} by "
        f"{-wo.loc[youngest, 'calibration_gap'] * 100:.1f} pp and **overstates it for "
        f"{oldest} by {wo.loc[oldest, 'calibration_gap'] * 100:+.1f} pp**. Good payers aged "
        f"{oldest} who are declined rise from {_pct(w.loc[oldest, 'goods_declined'])} to "
        f"{_pct(wo.loc[oldest, 'goods_declined'])}.\n"
        f'* "Fairness through unawareness" therefore shifts the cost onto older applicants, '
        f"the group that age-discrimination rules protect most explicitly. The champion "
        f"**keeps age**. It's monotonically constrained so that, all else equal, greater age "
        f"never increases risk, and the decision should be reviewed with compliance."
    )


def _model_card(
    meta: dict[str, Any],
    cfg: TrainConfig,
    metrics: dict[str, Any],
    result: dict[str, Any],
    deciles: pd.DataFrame,
    strategy: pd.DataFrame,
    decision: dict[str, Any],
    importance: pd.DataFrame,
    examples: list[dict[str, Any]],
    fair_with: pd.DataFrame,
    fair_without: pd.DataFrame,
    auc_no_age: float,
) -> str:
    sc = result["scorecard_test_metrics"]
    perf = pd.DataFrame(
        [
            {"model": f"{meta['champion']} (champion)", **metrics},
            {"model": "WoE logistic scorecard", **sc},
        ]
    )[["model", "auc", "gini", "ks", "pr_auc", "brier", "ece"]]
    fair_cols = [
        "group",
        "count",
        "default_rate",
        "mean_pd",
        "auc",
        "approval_rate",
        "adverse_impact_ratio",
        "goods_declined",
    ]
    pct = "{:.1%}"
    fair_fmt = {
        "default_rate": pct,
        "mean_pd": pct,
        "approval_rate": pct,
        "goods_declined": pct,
        "auc": "{:.3f}",
        "adverse_impact_ratio": "{:.2f}",
        "count": "{:,.0f}",
    }
    ex_lines = []
    for i, ex in enumerate(examples, 1):
        reasons = "\n".join(f"   - `{code}` {msg}" for code, msg, _ in ex["reasons"])
        ex_lines.append(f"{i}. Score **{ex['score']}**, PD {_pct(ex['pd'])} -> decline\n{reasons}")
    d = decision
    splits = meta["data"]["splits"]

    return f"""# Model Card: Credit Default Risk (PD) Model

> Generated by `uv run poe evaluate` from the trained model. Don't edit by hand.

| | |
|---|---|
| Model version | `{meta["model_version"]}` |
| Champion | **{meta["champion"]}** (challenger: WoE logistic scorecard) |
| Trained | {meta["created_at"]} (git `{meta["git_commit"]}`) |
| Training data | `{meta["data"]["raw_file"]}`, sha256 `{meta["data"]["raw_sha256"][:16]}...` |
| Split (stratified) | train {splits["train"]["rows"]:,} / valid {splits["valid"]["rows"]:,} / test {splits["test"]["rows"]:,} |

## Intended use

Estimates the **probability that an applicant becomes 90+ days delinquent within two
years** (PD), for consumer credit decisioning. It returns the PD, a 300-850 score, a risk
grade (A-E), an approve/decline decision and up to four reason codes.

**Not intended for:** loan pricing without recalibrating to the target portfolio,
populations unlike the training data (see Limitations), or fully automated decisions
without a human review route for declined applicants.

## Performance (held-out test split)

{_md_table(perf, {k: "{:.3f}" for k in ["auc", "gini", "ks", "pr_auc"]} | {"brier": "{:.4f}", "ece": "{:.4f}"})}

The champion is the challenger LightGBM only if it beats the interpretable scorecard by
at least {cfg.selection.min_auc_gain_for_challenger} AUC on validation. Selection, calibration and threshold
choices used the validation split. These test numbers were computed once.

| ROC | Calibration |
|---|---|
| ![ROC](figures/roc.png) | ![Calibration](figures/calibration.png) |

Calibration: mean predicted PD {_pct(metrics["mean_pd"], 2)} vs observed default rate
{_pct(metrics["default_rate"], 2)}, expected calibration error {metrics["ece"]:.4f}.
No class reweighting or oversampling was used, because they inflate PDs without improving ranking.

## Business impact

Decision rule: **decline if PD >= {_pct(d["pd_cutoff"])}** (score below {d["score_cutoff"]}). The
cut-off minimises expected cost, assuming that approving a borrower who defaults costs
{cfg.decision.cost_false_negative:g}x as much as declining one who would have repaid:
cut-off = cost_FP / (cost_FP + cost_FN).

| Metric (test) | Value |
|---|---|
| Approval rate | {_pct(d["approval_rate"])} |
| Bad rate of approved book | **{_pct(d["bad_rate_approved"])}** (vs {_pct(d["bad_rate_all"])} if everyone were approved) |
| Defaults declined | {_pct(d["defaults_declined"])} |
| Good payers declined | {_pct(d["goods_declined"])} |
| Riskiest 10% of applicants | contain {_pct(float(deciles["cum_capture"].iloc[0]))} of all defaults |

| Gains | Score distribution |
|---|---|
| ![Gains](figures/gains.png) | ![Scores](figures/score_distribution.png) |

<details><summary>Strategy table: approval rate vs bad rate</summary>

{_md_table(strategy, {"approval_rate": pct, "pd_cutoff": "{:.3f}", "bad_rate_approved": "{:.2%}", "defaults_caught": pct, "expected_cost": "{:.3f}"})}

</details>

## Explainability

Each prediction comes with SHAP contributions (exact TreeSHAP), which are grouped into stable,
customer-facing **reason codes**. Several features that describe one fact (e.g. four
late-payment columns) form one reason.

![Feature importance](figures/feature_importance.png)

Example declined applicants (test split):

{chr(10).join(ex_lines)}

## Fairness: age

Age is a protected characteristic. Per age band on the test split:

**Champion (uses age)**

{_md_table(fair_with[fair_cols], fair_fmt)}

**Same model refit without age** (AUC {auc_no_age:.4f} vs {metrics["auc"]:.4f})

{_md_table(fair_without[fair_cols], fair_fmt)}

![Fairness by age](figures/fairness_age.png)

**Finding**

{_fairness_finding(fair_with, fair_without)}

## Limitations

* **No out-of-time validation.** The dataset has no application dates, so the test split
  is random rather than later in time. Real deployment needs an out-of-time holdout and
  population-stability monitoring.
* **Population.** Kaggle "Give Me Some Credit" (US consumer credit, ~2008 era). Base
  rates and relationships will differ for other markets and periods. Recalibrate before use.
* **Cost assumptions.** The {cfg.decision.cost_false_negative:g}:{cfg.decision.cost_false_positive:g} cost ratio is an assumption. The cut-off should be set
  from the lender's actual loss-given-default and margin.
* **Data quality.** Self-reported income (about 20% missing) and utilization values far above
  100% are treated as described in `src/credit_scoring/features/transformers.py`.
* **Proxy risk.** Features other than age (e.g. income, dependents) may correlate with
  protected characteristics that aren't in the data and can't be tested here.

## Reproduce

```bash
uv run poe data       # validate, clean, split
uv run poe train      # train, compare, calibrate, package
uv run poe evaluate   # this card
```
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(get_settings())


if __name__ == "__main__":
    main()
