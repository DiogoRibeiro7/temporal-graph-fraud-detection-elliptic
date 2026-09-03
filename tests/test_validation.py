from __future__ import annotations

import pandas as pd
import pytest
from dataexcept import DataLeakageError, DataValidationError

from graph_fraud.config import LABEL_COL, TIME_COL
from graph_fraud.validation import assert_no_temporal_leakage, temporal_train_test_split


def test_temporal_split(synthetic_tables) -> None:
    nodes, _ = synthetic_tables
    train, test, _ = temporal_train_test_split(nodes, test_time_steps=2)
    assert train[TIME_COL].max() < test[TIME_COL].min()
    assert_no_temporal_leakage(train, test)


def test_leakage_detection_raises_structured_error(synthetic_tables) -> None:
    nodes, _ = synthetic_tables
    train = nodes.iloc[:10].copy()
    test = nodes.iloc[10:20].copy()
    train[TIME_COL] = 5
    test[TIME_COL] = 5

    with pytest.raises(DataLeakageError, match="Temporal leakage detected"):
        assert_no_temporal_leakage(train, test)


def test_temporal_split_rejects_insufficient_history() -> None:
    frame = pd.DataFrame(
        {
            TIME_COL: [1, 1, 2, 2],
            LABEL_COL: [0.0, 1.0, 0.0, 1.0],
        }
    )

    with pytest.raises(DataValidationError, match="Not enough labelled time steps"):
        temporal_train_test_split(frame, test_time_steps=2)
