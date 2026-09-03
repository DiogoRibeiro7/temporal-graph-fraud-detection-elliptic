"""Data loading utilities for the Elliptic transaction graph."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dataexcept import DataLoadingError, DataValidationError, MissingColumnError

from graph_fraud.config import (
    CLASSES_FILE,
    EDGES_FILE,
    FEATURES_FILE,
    LABEL_COL,
    RAW_DATA_DIR,
    TIME_COL,
    TX_ID_COL,
)


def require_columns(frame: pd.DataFrame, required: Iterable[str], *, frame_name: str = "frame") -> None:
    """Validate that a DataFrame contains all required columns."""
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise MissingColumnError(column=missing[0], dataframe=frame_name)


def map_class_label(value: object) -> float:
    """Map Elliptic labels to binary values."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "illicit"}:
        return 1.0
    if text in {"2", "0", "licit"}:
        return 0.0
    if text in {"unknown", "", "nan"}:
        return np.nan
    raise DataValidationError(
        field="class",
        value=value,
        message=f"Unknown Elliptic class label: {value!r}",
    )


def _resolve(data_dir: Path, name: str) -> Path:
    """Resolve a required data file path."""
    path = data_dir / name
    if not path.exists():
        raise DataLoadingError(
            source=str(path),
            original=FileNotFoundError(
                f"Missing {path}. Run `make data` or `make synthetic`."
            ),
        )
    return path


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV file and normalize pandas I/O/parsing failures."""
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadingError(source=str(path), original=exc) from exc


def load_classes(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load transaction labels."""
    path = _resolve(data_dir, CLASSES_FILE)
    frame = _read_csv(path).rename(columns={"txId": TX_ID_COL, "class": "raw_class"})
    require_columns(frame, [TX_ID_COL, "raw_class"], frame_name="classes")
    frame[LABEL_COL] = frame["raw_class"].map(map_class_label)
    return frame[[TX_ID_COL, LABEL_COL]]


def load_edges(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load directed transaction edges."""
    path = _resolve(data_dir, EDGES_FILE)
    frame = _read_csv(path).rename(columns={"txId1": "source", "txId2": "target"})
    require_columns(frame, ["source", "target"], frame_name="edges")
    return frame[["source", "target"]].drop_duplicates().reset_index(drop=True)


def load_features(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load transaction features, supporting files with or without headers."""
    path = _resolve(data_dir, FEATURES_FILE)
    peek = _read_csv(path, nrows=1)
    if "txId" in peek.columns or TX_ID_COL in peek.columns:
        frame = _read_csv(path).rename(columns={"txId": TX_ID_COL})
    else:
        frame = _read_csv(path, header=None)
        if frame.shape[1] < 2:
            raise DataValidationError(
                field="features",
                value=frame.shape[1],
                message="Feature data must contain transaction id and time-step columns.",
            )
        frame.columns = [TX_ID_COL, TIME_COL, *[f"x_{i}" for i in range(1, frame.shape[1] - 1)]]
    if TIME_COL not in frame.columns:
        if frame.shape[1] < 2:
            raise MissingColumnError(column=TIME_COL, dataframe="features")
        frame = frame.rename(columns={frame.columns[1]: TIME_COL})
    require_columns(frame, [TX_ID_COL, TIME_COL], frame_name="features")
    return frame


def load_elliptic_dataset(data_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load nodes and edges from raw data."""
    nodes = load_features(data_dir).merge(load_classes(data_dir), on=TX_ID_COL, how="left")
    return nodes, load_edges(data_dir)


def save_tables(nodes: pd.DataFrame, edges: pd.DataFrame, output_dir: Path) -> None:
    """Save node and edge tables."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes.to_csv(output_dir / "nodes.csv", index=False)
        edges.to_csv(output_dir / "edges.csv", index=False)
    except OSError as exc:
        raise DataLoadingError(source=str(output_dir), original=exc) from exc
