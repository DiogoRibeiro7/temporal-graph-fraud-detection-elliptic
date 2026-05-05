"""Download the Elliptic dataset from Kaggle."""

from __future__ import annotations

import shutil
import subprocess

from graph_fraud.config import RAW_DATA_DIR

KAGGLE_SLUG = "ellipticco/elliptic-data-set"


def main() -> None:
    """Download and unzip the Kaggle dataset."""
    if shutil.which("kaggle") is None:
        raise SystemExit("Kaggle CLI not found. Run `poetry install` and configure Kaggle credentials.")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_SLUG,
        "-p",
        str(RAW_DATA_DIR),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Kaggle download failed. Check ~/.kaggle/kaggle.json.") from exc
    print(f"Downloaded {KAGGLE_SLUG} to {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
