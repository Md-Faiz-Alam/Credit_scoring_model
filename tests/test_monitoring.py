import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.api.schemas import to_model_frame
from credit_scoring.data import columns as c
from credit_scoring.data.clean import clean
from credit_scoring.features.pipeline import split_xy
from credit_scoring.models.config import TrainConfig
from credit_scoring.models.train import train
from credit_scoring.monitoring.monitor import compare, read_log, run
from credit_scoring.monitoring.profile import (
    PROFILE_FILE,
    ReferenceProfile,
    build_profile,
    profile_variable,
)
from credit_scoring.monitoring.psi import (
    bin_shares,
    psi,
    quantile_edges,
    status_for,
)
from credit_scoring.monitoring.simulate import apply_scenario, simulate, to_applicants

from .conftest import make_raw
from .test_models import FAST, _prepared_settings

# ------------------------------------------------------------------ PSI


def test_psi_zero_for_identical_population() -> None:
    v = np.random.default_rng(0).normal(size=10_000)
    edges = quantile_edges(v)
    result = psi(bin_shares(v, edges), bin_shares(v, edges))
    assert result.psi == pytest.approx(0.0, abs=1e-9)
    assert result.status == "stable"


def test_psi_grows_with_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=20_000)
    edges = quantile_edges(ref)
    small = psi(bin_shares(ref, edges), bin_shares(rng.normal(0.1, 1, 20_000), edges)).psi
    large = psi(bin_shares(ref, edges), bin_shares(rng.normal(1.0, 1, 20_000), edges)).psi
    assert small < 0.1 < 0.25 < large


@pytest.mark.parametrize(
    ("value", "status"), [(0.05, "stable"), (0.1, "monitor"), (0.2, "monitor"), (0.3, "action")]
)
def test_status_thresholds(value: float, status: str) -> None:
    assert status_for(value) == status


def test_bin_shares_has_missing_bin_and_sums_to_one() -> None:
    shares = bin_shares(np.array([1.0, 2.0, np.nan, 5.0]), [1.5, 3.0])
    assert len(shares) == 4  # 3 numeric bins + missing
    assert shares[-1] == pytest.approx(0.25)
    assert shares.sum() == pytest.approx(1.0)


def test_quantile_edges_collapse_for_count_data() -> None:
    counts = np.array([0] * 900 + [1] * 80 + [2] * 20, dtype=float)
    assert quantile_edges(counts) == sorted(set(quantile_edges(counts)))
    assert len(quantile_edges(counts)) <= 2


def test_psi_finite_with_empty_bins() -> None:
    assert np.isfinite(psi(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5])).psi)


# ------------------------------------------------------------------ profile


def test_profile_roundtrip(tmp_path: Path) -> None:
    df, _ = clean(make_raw(n=500, seed=2).rename(columns=c.RAW_TO_CLEAN))
    X, _ = split_xy(df)
    rng = np.random.default_rng(0)
    p = rng.random(len(X)) * 0.3
    profile = build_profile(X, rng.integers(300, 850, len(X)), p, 0.2, "v1")
    profile.save(tmp_path)
    loaded = ReferenceProfile.load(tmp_path)
    assert loaded == profile
    assert set(loaded.features) >= set(c.RAW_FEATURES)


def test_profile_variable_records_missing() -> None:
    prof = profile_variable(np.array([1.0, np.nan, 3.0, np.nan]))
    assert prof.missing_rate == pytest.approx(0.5)
    assert sum(prof.shares) == pytest.approx(1.0)


# ------------------------------------------------------------------ simulator


@pytest.fixture(scope="module")
def clean_x() -> pd.DataFrame:
    df, _ = clean(make_raw(n=3000, seed=4, signal=True).rename(columns=c.RAW_TO_CLEAN))
    return split_xy(df)[0]


def test_baseline_scenario_is_identity(clean_x: pd.DataFrame) -> None:
    out = apply_scenario(clean_x, "baseline", np.random.default_rng(0))
    pd.testing.assert_frame_equal(out, clean_x)


def test_downturn_stresses_borrowers(clean_x: pd.DataFrame) -> None:
    data = clean_x.copy()
    data.loc[0, "revolving_utilization"] = 5000.0  # data-error value must survive untouched
    out = apply_scenario(data, "downturn", np.random.default_rng(0))
    assert out["monthly_income"].mean() < data["monthly_income"].mean()
    assert (
        out["revolving_utilization"].iloc[1:].mean() > data["revolving_utilization"].iloc[1:].mean()
    )
    assert out.loc[0, "revolving_utilization"] == 5000.0
    assert out["past_due_30_59"].sum() > data["past_due_30_59"].sum()


def test_young_segment_shifts_age(clean_x: pd.DataFrame) -> None:
    out = apply_scenario(clean_x, "young_segment", np.random.default_rng(0))
    assert (out["age"] < 35).mean() > (clean_x["age"] < 35).mean() + 0.2


def test_unknown_scenario(clean_x: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_scenario(clean_x, "meteor", np.random.default_rng(0))


def test_applicant_conversion_roundtrips_to_model_frame(clean_x: pd.DataFrame) -> None:
    """Dataset row -> API request -> model frame must give back the same model inputs."""
    sample = clean_x.head(200)
    applicants, rejected = to_applicants(sample)
    assert rejected == 0
    back = to_model_frame(applicants)
    for col in ("debt_ratio", "revolving_utilization", "monthly_income", "age"):
        np.testing.assert_allclose(back[col], sample[col].to_numpy(), rtol=1e-9, equal_nan=True)


def test_conversion_skips_rows_api_would_reject(clean_x: pd.DataFrame) -> None:
    bad = clean_x.head(3).copy()
    bad.loc[bad.index[0], "age"] = np.nan
    bad.loc[bad.index[1], "revolving_utilization"] = 50_000.0
    applicants, rejected = to_applicants(bad)
    assert (len(applicants), rejected) == (1, 2)


# ------------------------------------------------------------------ monitor


def _profile_and_log(n: int = 2000) -> tuple[ReferenceProfile, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f: rng.normal(10, 2, n) for f in (*c.RAW_FEATURES, c.DELINQUENCY_CODE_FLAG)})
    scores = rng.normal(650, 60, n).round()
    pd_ = np.clip(rng.beta(1, 12, n), 0, 1)
    profile = build_profile(X, scores, pd_, 0.2, "v1")
    log = X.assign(
        score=scores,
        pd=pd_,
        decision=np.where(pd_ >= 0.2, "decline", "approve"),
        model_version="v1",
        ts=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    return profile, log


def test_compare_stable_when_nothing_changes() -> None:
    profile, log = _profile_and_log()
    report = compare(profile, log, "log.jsonl")
    assert report.status == "stable"
    assert report.alerts == []


def test_compare_flags_feature_drift_and_missing_data() -> None:
    profile, log = _profile_and_log()
    log["age"] = log["age"] + 4  # large shift
    log.loc[: len(log) // 5, "monthly_income"] = np.nan  # 20% missing suddenly
    report = compare(profile, log, "log.jsonl")
    by_name = {d.name: d for d in report.features}
    assert by_name["age"].status == "action"
    assert report.status == "action"
    assert any("missing rate" in a and "monthly_income" in a for a in report.alerts)


def test_compare_kpi_catches_what_psi_misses() -> None:
    """A uniform PD increase can leave PSI low, but the mean-PD KPI must alert."""
    profile, log = _profile_and_log()
    log["pd"] = log["pd"] * 1.5
    report = compare(profile, log, "log.jsonl")
    assert report.mean_pd_status == "action"
    assert report.status == "action"


def test_compare_warns_on_model_version_mismatch() -> None:
    profile, log = _profile_and_log()
    log["model_version"] = "v2"
    assert any("model versions" in a for a in compare(profile, log, "x").alerts)


def test_read_log_filters_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    rows = [
        {"ts": ts, "model_version": "v", "features": {"age": 30.0}, "pd": 0.1, "score": 600,
         "decision": "approve"}
        for ts in ("2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00")
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert len(read_log(path)) == 2
    assert len(read_log(path, since=datetime(2026, 1, 15, tzinfo=UTC))) == 1
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_log(tmp_path / "empty.jsonl")
    with pytest.raises(FileNotFoundError):
        read_log(tmp_path / "missing.jsonl")


# ------------------------------------------------------------------ end to end


def test_train_simulate_monitor(tmp_path: Path) -> None:
    settings = _prepared_settings(tmp_path, signal=True)
    cfg = TrainConfig.model_validate(
        {**FAST, "quality_gate": {"min_valid_auc": 0.6, "max_valid_ece": 0.1}}
    )
    train(settings, cfg, tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    assert (settings.models_dir / PROFILE_FILE).exists()  # saved at training time

    reports = {}
    for scenario in ("baseline", "downturn"):
        log = tmp_path / f"{scenario}.jsonl"
        assert simulate(settings, scenario, n=600, out=log) > 0
        out = tmp_path / "out" / scenario
        reports[scenario] = run(settings, log, [out])
        assert (out / "report.md").exists()
        assert (out / "figures" / "stability.png").exists()

    assert reports["downturn"].mean_pd_now > reports["baseline"].mean_pd_now
    assert reports["baseline"].score.psi < reports["downturn"].score.psi
