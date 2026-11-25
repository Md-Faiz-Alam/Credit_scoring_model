"""Download the Kaggle "Give Me Some Credit" training file.

Prerequisites (one time):
  1. Create a Kaggle API token (kaggle.com -> Settings -> API -> Create New Token) and place
     ``kaggle.json`` in ``~/.kaggle/`` (on Windows ``%USERPROFILE%\\.kaggle\\``).
  2. Accept the competition rules at https://www.kaggle.com/c/GiveMeSomeCredit/rules

Usage:  uv run poe download
"""

import logging
import zipfile
from pathlib import Path

from credit_scoring.config import get_settings

COMPETITION = "GiveMeSomeCredit"
REMOTE_FILE = "cs-training.csv"

log = logging.getLogger(__name__)


def download(dest_dir: Path, *, force: bool = False) -> Path:
    target = dest_dir / REMOTE_FILE
    if target.exists() and not force:
        log.info("Dataset already present at %s (use --force to re-download)", target)
        return target

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("Kaggle client missing. Install it with: uv sync --extra data") from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_file(COMPETITION, REMOTE_FILE, path=str(dest_dir), force=force)

    # Kaggle may deliver the file zipped
    zipped = dest_dir / f"{REMOTE_FILE}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extract(REMOTE_FILE, dest_dir)
        zipped.unlink()

    log.info("Downloaded %s", target)
    return target


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download(get_settings().raw_dir, force=args.force)


if __name__ == "__main__":
    main()
