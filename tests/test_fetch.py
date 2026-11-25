from pathlib import Path

import pytest

from credit_scoring.config import Settings
from credit_scoring.models.bundle import METADATA_FILE, MODEL_FILE
from credit_scoring.models.fetch import ModelIntegrityError, ensure_model, sha256_of
from credit_scoring.monitoring.profile import PROFILE_FILE


@pytest.fixture
def release(tmp_path: Path) -> Path:
    """A fake release folder served over file:// (no network in tests)."""
    folder = tmp_path / "release"
    folder.mkdir()
    (folder / MODEL_FILE).write_bytes(b"model-bytes")
    (folder / METADATA_FILE).write_text("{}", encoding="utf-8")
    (folder / PROFILE_FILE).write_text("{}", encoding="utf-8")
    return folder


def _settings(tmp_path: Path, release: Path, sha: str | None) -> Settings:
    return Settings(models_dir=tmp_path / "models", model_url=release.as_uri(), model_sha256=sha)


def test_downloads_and_verifies(tmp_path: Path, release: Path) -> None:
    sha = sha256_of(release / MODEL_FILE)
    settings = _settings(tmp_path, release, sha)
    assert ensure_model(settings) is True
    for name in (MODEL_FILE, METADATA_FILE, PROFILE_FILE):
        assert (settings.models_dir / name).exists()
    assert not list(settings.models_dir.glob("*.download"))


def test_hash_mismatch_refuses_model(tmp_path: Path, release: Path) -> None:
    settings = _settings(tmp_path, release, "0" * 64)
    with pytest.raises(ModelIntegrityError, match="mismatch"):
        ensure_model(settings)
    assert not (settings.models_dir / MODEL_FILE).exists()
    assert not list(settings.models_dir.iterdir())  # nothing half-written left behind


def test_url_without_hash_is_refused(tmp_path: Path, release: Path) -> None:
    with pytest.raises(ModelIntegrityError, match="unverified"):
        ensure_model(_settings(tmp_path, release, None))


def test_local_model_wins(tmp_path: Path, release: Path) -> None:
    settings = _settings(tmp_path, release, "irrelevant")
    settings.models_dir.mkdir()
    (settings.models_dir / MODEL_FILE).write_bytes(b"local")
    assert ensure_model(settings) is False
    assert (settings.models_dir / MODEL_FILE).read_bytes() == b"local"


def test_no_url_is_a_no_op(tmp_path: Path) -> None:
    assert ensure_model(Settings(models_dir=tmp_path / "m")) is False


def test_optional_files_may_be_missing(tmp_path: Path, release: Path) -> None:
    (release / PROFILE_FILE).unlink()
    settings = _settings(tmp_path, release, sha256_of(release / MODEL_FILE))
    assert ensure_model(settings) is True
    assert (settings.models_dir / MODEL_FILE).exists()
    assert not (settings.models_dir / PROFILE_FILE).exists()


def test_unreachable_url_fails_cleanly(tmp_path: Path) -> None:
    settings = Settings(
        models_dir=tmp_path / "models",
        model_url=(tmp_path / "nowhere").as_uri(),
        model_sha256="a" * 64,
    )
    with pytest.raises(OSError):
        ensure_model(settings)
    assert not list((tmp_path / "models").iterdir())
