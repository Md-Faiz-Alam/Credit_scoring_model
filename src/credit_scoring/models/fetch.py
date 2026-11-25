"""Fetch released model artefacts when the model isn't on local disk.

Hosting platforms build images from git, and model files deliberately aren't in git. A
trained model is published as versioned release assets instead, and the service downloads
them at startup.

``credit_model.joblib`` is a pickle, and unpickling runs code, so a downloaded model is
loaded only if its SHA-256 matches the expected hash from configuration. A wrong or
missing hash stops startup.
"""

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

from credit_scoring.config import Settings
from credit_scoring.models.bundle import METADATA_FILE, MODEL_FILE
from credit_scoring.monitoring.profile import PROFILE_FILE

log = logging.getLogger(__name__)

OPTIONAL_FILES = (METADATA_FILE, PROFILE_FILE)


class ModelIntegrityError(RuntimeError):
    """The downloaded model doesn't match the expected SHA-256."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, timeout: float) -> None:
    """Download to a temp file, then move into place, so a failed download leaves nothing."""
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                shutil.copyfileobj(resp, tmp)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(dest)


def ensure_model(settings: Settings, timeout: float = 60.0) -> bool:
    """Make sure the model exists locally. Returns True if it was downloaded."""
    model_path = settings.models_dir / MODEL_FILE
    if model_path.exists() or not settings.model_url:
        return False
    if not settings.model_sha256:
        raise ModelIntegrityError(
            "CREDIT_MODEL_URL is set but CREDIT_MODEL_SHA256 is not: refusing to load an "
            "unverified remote model"
        )

    base = settings.model_url.rstrip("/")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    staged = settings.models_dir / f"{MODEL_FILE}.download"
    log.info("Downloading model from %s", base)
    _download(f"{base}/{MODEL_FILE}", staged, timeout)

    actual = sha256_of(staged)
    if actual != settings.model_sha256.strip().lower():
        staged.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"SHA-256 mismatch for {MODEL_FILE}: expected {settings.model_sha256}, got {actual}"
        )
    staged.replace(model_path)

    for name in OPTIONAL_FILES:  # metadata copy and monitoring profile: nice to have
        try:
            _download(f"{base}/{name}", settings.models_dir / name, timeout)
        except OSError as exc:
            log.warning("Optional artefact %s not downloaded: %s", name, exc)
    log.info("Model verified (sha256 %s...)", actual[:12])
    return True
