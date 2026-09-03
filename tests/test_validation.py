from __future__ import annotations

import pandas as pd
import pytest
from dataexcept import DataLeakageError, DataValidationError

from graph_fraud.validation import assert_no_temporal_leakage, temporal_train_test_split


def test_temporal_split(synthetic_tables) -> None:
    nodes, _ = synthetic_tables
    train, test, _ = temporal_train_test_split(nodes, test_time_steps=2)
    assert train["time_step"].max() < test["time_step"].min()
    assert_no_temporal_leakage(train, test)


def test_leakage_detection_raises_structured_error(synthetic_tables) -> None:
    nodes, _ = synthetic_tables
    train = nodes.iloc[:10].copy()
    test = nodes.iloc[10:20].copy()
    train["time_step"] = 5
    test["time_step"] = 5

    with pytest.raises(DataLeakageError, match="Temporal leakage detected"):
        assert_no_temporal_leakage(train, test)


def test_temporal_split_rejects_insufficient_history() -> None:
    frame = pd.DataFrame(
        {
            "time_step": [1, 1, 2, 2],
            "is_illicit": [0.0, 1.0, 0.0, 1.0],
        }
    )

    with pytest.raises(DataValidationError, match="Not enough labelled time steps"):
        temporal_train_test_split(frame, test_time_steps=2)
