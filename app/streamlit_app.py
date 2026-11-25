"""Credit scoring web UI: a client of the scoring API.

Run (API must be running):
    uv run poe api      # terminal 1
    uv run poe ui       # terminal 2
"""

import io
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from credit_scoring.ui.client import ApiClient, ApiError

API_URL = os.environ.get("CREDIT_API_URL", "http://localhost:8000")

# Palette (validated): blue series, status colours only with icon + label.
# Neutrals are translucent grey so they read on both Streamlit light and dark themes;
# text colours are left to the theme.
BLUE = "#2a78d6"
NEUTRAL = "rgba(128,128,128,0.9)"
GRID = "rgba(128,128,128,0.25)"
ZONE_DECLINE = "rgba(128,128,128,0.32)"
ZONE_APPROVE = "rgba(128,128,128,0.10)"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

SAMPLES: dict[str, dict[str, Any]] = {
    "Established borrower (low risk)": {
        "age": 58, "monthly_income": 9200.0, "monthly_debt_payments": 1800.0,
        "revolving_utilization": 0.08, "open_credit_lines": 9, "real_estate_loans": 1,
        "dependents": 1, "past_due_30_59": 0, "past_due_60_89": 0, "past_due_90_plus": 0,
    },
    "Young professional (borderline)": {
        "age": 27, "monthly_income": 3800.0, "monthly_debt_payments": 1500.0,
        "revolving_utilization": 0.72, "open_credit_lines": 6, "real_estate_loans": 0,
        "dependents": 0, "past_due_30_59": 1, "past_due_60_89": 0, "past_due_90_plus": 0,
    },
    "Stretched borrower (high risk)": {
        "age": 36, "monthly_income": 4100.0, "monthly_debt_payments": 3300.0,
        "revolving_utilization": 0.97, "open_credit_lines": 12, "real_estate_loans": 1,
        "dependents": 3, "past_due_30_59": 2, "past_due_60_89": 1, "past_due_90_plus": 1,
    },
}  # fmt: skip

FIELDS = list(next(iter(SAMPLES.values())))
# Deep links for demos and sharing a case: ?profile=high-risk preselects and scores it
PROFILE_LINKS = dict(zip(("low-risk", "borderline", "high-risk"), SAMPLES, strict=True))

st.set_page_config(page_title="Credit Scoring", page_icon="📊", layout="wide")


@st.cache_resource
def get_client(url: str) -> ApiClient:
    return ApiClient(url)


@st.cache_data(ttl=60)
def get_model_info(url: str) -> dict[str, Any]:
    return get_client(url).model_info()


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Scoring API")
    api_url = st.text_input("API URL", API_URL)
    client = get_client(api_url)
    try:
        health = client.health()
        info = get_model_info(api_url)
        st.success(f"Connected, model `{health['model_version']}`", icon="✅")
    except ApiError as err:
        st.error(str(err), icon="⚠️")
        st.info("Start the API with `uv run poe api`, then reload this page.")
        st.stop()
    st.caption(
        f"Champion: **{info['champion']}**  \n"
        f"Decline if PD ≥ {info['decision_threshold_pd']:.1%} "
        f"(score < {info['decision_threshold_score']})"
    )

cutoff_score = int(info["decision_threshold_score"])


# ------------------------------------------------------------------ helpers


def score_gauge(score: int) -> go.Figure:
    lo, hi = info["score_range"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 44}},
            gauge={
                "axis": {"range": [lo, hi], "tickcolor": NEUTRAL},
                "bar": {"color": BLUE, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [lo, cutoff_score], "color": ZONE_DECLINE},
                    {"range": [cutoff_score, hi], "color": ZONE_APPROVE},
                ],
                "threshold": {
                    "line": {"color": NEUTRAL, "width": 3},
                    "thickness": 0.9,
                    "value": cutoff_score,
                },
            },
        )
    )
    fig.update_layout(height=230, margin={"t": 10, "b": 0, "l": 30, "r": 30})
    return fig


def reasons_chart(reasons: list[dict[str, Any]]) -> go.Figure:
    data = list(reversed(reasons))
    fig = go.Figure(
        go.Bar(
            x=[r["impact"] for r in data],
            y=[r["code"] for r in data],
            orientation="h",
            marker={"color": BLUE},
            hovertemplate="%{y}<br>+%{x:.2f} log-odds<extra></extra>",
        )
    )
    fig.update_layout(
        height=60 + 46 * len(data),
        margin={"t": 10, "b": 30, "l": 10, "r": 10},
        xaxis={"title": "Impact on risk (log-odds)", "gridcolor": GRID, "zeroline": False},
        yaxis={"automargin": True},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def decision_badge(decision: str) -> None:
    if decision == "approve":
        st.markdown(f"<h3 style='color:{GOOD};margin:0'>✓ Approve</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:{CRITICAL};margin:0'>✕ Decline</h3>", unsafe_allow_html=True)


def show_result(result: dict[str, Any]) -> None:
    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(score_gauge(result["score"]), width="stretch")
    with right:
        decision_badge(result["decision"])
        a, b = st.columns(2)
        a.metric("Probability of default", f"{result['probability_of_default']:.1%}")
        b.metric("Risk grade", result["risk_grade"])
        st.caption(
            f"Model `{result['model_version']}`. Shaded zone = decline (score < {cutoff_score})."
        )
    if result["reasons"]:
        title = "Reasons for decline" if result["decision"] == "decline" else "Main risk factors"
        st.subheader(title)
        text_col, chart_col = st.columns([1.2, 1])
        with text_col:
            for r in result["reasons"]:
                st.markdown(f"**{r['code']}** &nbsp; {r['message']}")
        with chart_col:
            st.plotly_chart(reasons_chart(result["reasons"]), width="stretch")
    else:
        st.info("No factor raises this applicant's risk above average.")


# ------------------------------------------------------------------ tabs

tab_single, tab_batch, tab_model = st.tabs(["Score an applicant", "Batch scoring", "Model"])

with tab_single:
    st.title("Credit risk assessment")
    linked = PROFILE_LINKS.get(st.query_params.get("profile", ""))
    options = ["(blank)", *SAMPLES]
    sample = st.selectbox(
        "Start from a sample profile", options, index=options.index(linked) if linked else 0
    )
    defaults = SAMPLES.get(sample, {})

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 120, int(defaults.get("age", 35)))
        income = c1.number_input(
            "Monthly income (USD)", 0.0, 1e7, float(defaults.get("monthly_income", 5000.0)), 100.0
        )
        income_known = c1.checkbox("Income verified / reported", True)
        debt = c2.number_input(
            "Monthly debt payments (USD)", 0.0, 1e7,
            float(defaults.get("monthly_debt_payments", 1500.0)), 50.0,
        )  # fmt: skip
        util = c2.slider(
            "Revolving credit utilization", 0.0, 1.5,
            float(defaults.get("revolving_utilization", 0.3)), 0.01,
            help="Card/revolving balances ÷ credit limits. Above 100% = over limit.",
        )  # fmt: skip
        lines = c2.number_input(
            "Open credit lines & loans", 0, 100, int(defaults.get("open_credit_lines", 6))
        )
        re_loans = c3.number_input(
            "Real-estate loans", 0, 100, int(defaults.get("real_estate_loans", 1))
        )
        deps = c3.number_input("Dependents", 0, 30, int(defaults.get("dependents", 0)))
        st.markdown("**Payment history (last 2 years)**")
        d1, d2, d3 = st.columns(3)
        late30 = d1.number_input("30-59 days late", 0, 50, int(defaults.get("past_due_30_59", 0)))
        late60 = d2.number_input("60-89 days late", 0, 50, int(defaults.get("past_due_60_89", 0)))
        late90 = d3.number_input("90+ days late", 0, 50, int(defaults.get("past_due_90_plus", 0)))
        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        st.session_state["scored_applicant"] = {
            "age": int(age),
            "monthly_income": float(income) if income_known else None,
            "monthly_debt_payments": float(debt),
            "revolving_utilization": float(util),
            "open_credit_lines": int(lines),
            "real_estate_loans": int(re_loans),
            "dependents": int(deps),
            "past_due_30_59": int(late30),
            "past_due_60_89": int(late60),
            "past_due_90_plus": int(late90),
        }

    if linked and "scored_applicant" not in st.session_state:
        st.session_state["scored_applicant"] = dict(SAMPLES[linked])

    applicant = st.session_state.get("scored_applicant")
    if applicant:
        try:
            result = client.predict(applicant)
        except ApiError as err:
            st.error(str(err))
            st.stop()
        show_result(result)

        st.divider()
        st.subheader("What if…?")
        st.caption("Change one lever and see how the decision moves (the model is monotonic).")
        w1, w2, w3 = st.columns(3)
        new_util = w1.slider(
            "Utilization", 0.0, 1.5, min(applicant["revolving_utilization"], 1.5), 0.01
        )
        new_debt = w2.slider(
            "Monthly debt payments", 0.0, max(10_000.0, applicant["monthly_debt_payments"]),
            applicant["monthly_debt_payments"], 50.0,
        )  # fmt: skip
        new_late = w3.slider("30-59 days late", 0, 10, min(applicant["past_due_30_59"], 10))
        what_if = {
            **applicant,
            "revolving_utilization": new_util,
            "monthly_debt_payments": new_debt,
            "past_due_30_59": new_late,
        }
        if what_if != applicant:
            alt = client.predict(what_if)
            m1, m2, m3 = st.columns(3)
            m1.metric("Score", alt["score"], alt["score"] - result["score"])
            pd_change = alt["probability_of_default"] - result["probability_of_default"]
            m2.metric(
                "Probability of default",
                f"{alt['probability_of_default']:.1%}",
                f"{pd_change * 100:+.1f} pp",
                delta_color="inverse",
            )
            m3.metric("Decision", alt["decision"].title())


with tab_batch:
    st.title("Batch scoring")
    st.write(
        "Upload a CSV with one applicant per row (max 1,000). Columns: "
        + ", ".join(f"`{f}`" for f in ["application_id", *FIELDS])
        + ". Leave `monthly_income` / `dependents` empty if unknown."
    )
    template = pd.DataFrame(
        [{"application_id": f"S{i + 1}", **v} for i, v in enumerate(SAMPLES.values())]
    )
    st.download_button(
        "Download template", template.to_csv(index=False), "applicants_template.csv", "text/csv"
    )
    upload = st.file_uploader("Applicants CSV", type="csv")
    if upload is not None:
        frame = pd.read_csv(upload)
        missing = set(FIELDS) - set(frame.columns) - {"monthly_income", "dependents"}
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
        else:
            records = [
                {k: (None if pd.isna(v) else v) for k, v in row.items()}
                for row in frame.to_dict(orient="records")
            ]
            try:
                out = client.predict_batch(records)
            except ApiError as err:
                st.error(str(err))
            else:
                results = pd.DataFrame(
                    [
                        {
                            "application_id": r["application_id"],
                            "score": r["score"],
                            "grade": r["risk_grade"],
                            "pd": r["probability_of_default"],
                            "decision": r["decision"],
                            "reasons": "; ".join(x["code"] for x in r["reasons"]),
                        }
                        for r in out["results"]
                    ]
                )
                k1, k2, k3 = st.columns(3)
                k1.metric("Applicants", out["count"])
                k2.metric("Approved", f"{out['approved'] / out['count']:.0%}")
                k3.metric("Average score", f"{results['score'].mean():.0f}")
                st.dataframe(
                    results,
                    width="stretch",
                    column_config={"pd": st.column_config.NumberColumn("PD", format="%.2%")},
                )
                buf = io.StringIO()
                results.to_csv(buf, index=False)
                st.download_button("Download results", buf.getvalue(), "scored.csv", "text/csv")


with tab_model:
    st.title("Model")
    m = info["test_metrics"]
    cols = st.columns(4)
    for col, (label, key) in zip(
        cols,
        [("ROC-AUC", "auc"), ("Gini", "gini"), ("KS", "ks"), ("Calibration error", "ece")],
        strict=True,
    ):
        col.metric(label, f"{m.get(key, float('nan')):.3f}")
    st.caption(
        f"Held-out test split. Version `{info['model_version']}`, trained {info['created_at']}."
    )
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Risk grades")
        bounds = list(info["risk_grades"].items())
        rows, lo = [], 0.0
        for grade, hi in bounds:
            rows.append({"grade": grade, "PD range": f"{lo:.0%} - {hi:.0%}"})
            lo = hi
        st.table(pd.DataFrame(rows).set_index("grade"))
    with g2:
        st.subheader("Reason codes")
        st.table(
            pd.DataFrame(info["reason_codes"].items(), columns=["code", "meaning"]).set_index(
                "code"
            )
        )
    st.markdown(
        "Full methodology, fairness analysis and limitations: "
        "[model card](https://github.com/Md-Faiz-Alam/Credit_scoring_model/blob/main/docs/model_card.md)"
    )
