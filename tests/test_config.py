from pathlib import Path

import pytest

from credit_scoring import __version__
from credit_scoring.config import PROJECT_ROOT, Settings


def test_package_is_installed() -> None:
    assert __version__ != "0.0.0"


def test_paths_are_relative_to_project_root() -> None:
    settings = Settings()
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.raw_data_path == PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CREDIT_MODELS_DIR", str(tmp_path))
    assert Settings().models_dir == tmp_path


def test_empty_prediction_log_disables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_PREDICTION_LOG", "")
    assert Settings().prediction_log is None
