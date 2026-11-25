"""Project-wide settings.

Every path is resolved from the project root, never hardcoded, so the code runs the
same on any machine, in CI and inside Docker. Any value can be overridden with an
environment variable prefixed ``CREDIT_`` (for example ``CREDIT_MODELS_DIR=/mnt/models``)
or a ``.env`` file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CREDIT_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    # JSON-lines log of every scored application (inputs for drift monitoring).
    # Set CREDIT_PREDICTION_LOG="" to disable.
    prediction_log: Path | None = PROJECT_ROOT / "logs" / "predictions.jsonl"

    @field_validator("prediction_log", mode="before")
    @classmethod
    def _empty_disables_log(cls, v: object) -> object:
        # pydantic would turn "" into Path(".") (the current directory). Treat it as "off".
        return None if isinstance(v, str) and not v.strip() else v

    raw_data_file: str = "cs-training.csv"

    # Split: train / validation (model selection, calibration, threshold) / test (final, once)
    valid_size: float = Field(0.15, gt=0, lt=0.5)
    test_size: float = Field(0.15, gt=0, lt=0.5)
    random_state: int = 42

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_data_path(self) -> Path:
        return self.raw_dir / self.raw_data_file


@lru_cache
def get_settings() -> Settings:
    return Settings()
