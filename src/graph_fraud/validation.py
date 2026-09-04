"""Temporal validation utilities."""

from __future__ import annotations

import pandas as pd
from dataexcept import DataLeakageError, DataValidationError

from graph_fraud.config import LABEL_COL, TIME_COL
from graph_fraud.data import require_columns


def temporal_train_test_split(
    frame: pd.DataFrame,
    *,
    test_time_steps: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split labelled observations by time step."""
    if test_time_steps <= 0:
        raise ValueError("test_time_steps must be positive")

    require_columns(frame, [TIME_COL, LABEL_COL], frame_name="frame")
    labelled = frame[frame[LABEL_COL].notna()].copy()
    steps = sorted(labelled[TIME_COL].astype(int).unique().tolist())
    if len(steps) <= test_time_steps:
        raise DataValidationError(
            field=TIME_COL,
            value=len(steps),
            message=(
                "Not enough labelled time steps for temporal split: "
                f"found {len(steps)}, need more than {test_time_steps}."
            ),
        )
    cutoff = steps[-test_time_steps - 1]
    train = labelled[labelled[TIME_COL] <= cutoff].copy()
    test = labelled[labelled[TIME_COL] > cutoff].copy()
    if train.empty or test.empty:
        raise DataValidationError(
            field=TIME_COL,
            value={"train_rows": len(train), "test_rows": len(test)},
            message="Temporal split produced an empty train or test partition.",
        )
    return train, test, int(cutoff)


def temporal_train_calibration_test_split(
    frame: pd.DataFrame,
    *,
    calibration_time_steps: int = 2,
    test_time_steps: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """Create ordered train, calibration, and test partitions.

    The calibration slice is strictly later than training and strictly earlier
    than the final test slice. This prevents probability calibration from using
    the same observations on which final performance is reported.
    """
    if calibration_time_steps <= 0:
        raise ValueError("calibration_time_steps must be positive")
    if test_time_steps <= 0:
        raise ValueError("test_time_steps must be positive")

    require_columns(frame, [TIME_COL, LABEL_COL], frame_name="frame")
    labelled = frame[frame[LABEL_COL].notna()].copy()
    steps = sorted(labelled[TIME_COL].astype(int).unique().tolist())
    reserved_steps = calibration_time_steps + test_time_steps
    if len(steps) <= reserved_steps:
        raise DataValidationError(
            field=TIME_COL,
            value=len(steps),
            message=(
                "Not enough labelled time steps for train/calibration/test split: "
                f"found {len(steps)}, need more than {reserved_steps}."
            ),
        )

    train_cutoff = int(steps[-reserved_steps - 1])
    calibration_cutoff = int(steps[-test_time_steps - 1])
    train = labelled[labelled[TIME_COL] <= train_cutoff].copy()
    calibration = labelled[
        (labelled[TIME_COL] > train_cutoff)
        & (labelled[TIME_COL] <= calibration_cutoff)
    ].copy()
    test = labelled[labelled[TIME_COL] > calibration_cutoff].copy()

    if train.empty or calibration.empty or test.empty:
        raise DataValidationError(
            field=TIME_COL,
            value={
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
            },
            message="Temporal calibration split produced an empty partition.",
        )

    assert_no_temporal_leakage(train, calibration)
    assert_no_temporal_leakage(calibration, test)
    return train, calibration, test, train_cutoff, calibration_cutoff


def assert_no_temporal_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Ensure all train observations are earlier than test observations."""
    require_columns(train, [TIME_COL], frame_name="train")
    require_columns(test, [TIME_COL], frame_name="test")
    if train.empty or test.empty:
        raise DataValidationError(
            field=TIME_COL,
            value={"train_rows": len(train), "test_rows": len(test)},
            message="Temporal leakage validation requires non-empty train and test data.",
        )

    train_max = int(train[TIME_COL].max())
    test_min = int(test[TIME_COL].min())
    if train_max >= test_min:
        raise DataLeakageError(
            feature=TIME_COL,
            stage="temporal validation",
            message=(
                "Temporal leakage detected: "
                f"max(train.{TIME_COL})={train_max} >= min(test.{TIME_COL})={test_min}."
            ),
        )


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    min_train_steps: int = 4,
    test_window: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Create rolling-origin temporal validation splits."""
    if min_train_steps <= 0:
        raise ValueError("min_train_steps must be positive")
    if test_window <= 0:
        raise ValueError("test_window must be positive")

    require_columns(frame, [TIME_COL, LABEL_COL], frame_name="frame")
    labelled = frame[frame[LABEL_COL].notna()].copy()
    steps = sorted(labelled[TIME_COL].astype(int).unique().tolist())
    out: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    for i in range(min_train_steps, len(steps) - test_window + 1):
        cutoff = steps[i - 1]
        train = labelled[labelled[TIME_COL] <= cutoff].copy()
        test = labelled[labelled[TIME_COL].isin(steps[i : i + test_window])].copy()
        if not train.empty and not test.empty:
            assert_no_temporal_leakage(train, test)
            out.append((train, test, int(cutoff)))
    if not out:
        raise DataValidationError(
            field=TIME_COL,
            value=len(steps),
            message=(
                "No rolling-origin splits could be created with "
                f"min_train_steps={min_train_steps} and test_window={test_window}."
            ),
        )
    return out
