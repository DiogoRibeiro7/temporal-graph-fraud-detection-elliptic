"""Temporal validation utilities."""

from __future__ import annotations

import pandas as pd

from graph_fraud.config import LABEL_COL, TIME_COL
from graph_fraud.data import require_columns


def temporal_train_test_split(
    frame: pd.DataFrame,
    *,
    test_time_steps: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split labelled observations by time step."""
    require_columns(frame, [TIME_COL, LABEL_COL], frame_name="frame")
    labelled = frame[frame[LABEL_COL].notna()].copy()
    steps = sorted(labelled[TIME_COL].astype(int).unique().tolist())
    if len(steps) <= test_time_steps:
        raise ValueError("Not enough time steps for temporal split")
    cutoff = steps[-test_time_steps - 1]
    train = labelled[labelled[TIME_COL] <= cutoff].copy()
    test = labelled[labelled[TIME_COL] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced empty train or test")
    return train, test, int(cutoff)


def assert_no_temporal_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Ensure all train observations are earlier than test observations."""
    if int(train[TIME_COL].max()) >= int(test[TIME_COL].min()):
        raise ValueError("Temporal leakage detected")


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    min_train_steps: int = 4,
    test_window: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Create rolling-origin temporal validation splits."""
    require_columns(frame, [TIME_COL, LABEL_COL], frame_name="frame")
    labelled = frame[frame[LABEL_COL].notna()].copy()
    steps = sorted(labelled[TIME_COL].astype(int).unique().tolist())
    out = []
    for i in range(min_train_steps, len(steps) - test_window + 1):
        cutoff = steps[i - 1]
        train = labelled[labelled[TIME_COL] <= cutoff].copy()
        test = labelled[labelled[TIME_COL].isin(steps[i : i + test_window])].copy()
        if not train.empty and not test.empty:
            out.append((train, test, int(cutoff)))
    if not out:
        raise ValueError("No rolling-origin splits could be created")
    return out
