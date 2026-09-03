from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dataexcept import DataLoadingError, DataValidationError, MissingColumnError

from graph_fraud.data import load_edges, map_class_label, require_columns


def test_map_class_label() -> None:
    assert map_class_label("1") == 1.0
    assert map_class_label("2") == 0.0
    assert map_class_label("illicit") == 1.0
    assert map_class_label("licit") == 0.0
    assert np.isnan(map_class_label("unknown"))


def test_map_class_label_rejects_unknown_value() -> None:
    with pytest.raises(DataValidationError, match="Unknown Elliptic class label"):
        map_class_label("suspicious")


def test_require_columns_raises_structured_error() -> None:
    with pytest.raises(MissingColumnError) as exc_info:
        require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], frame_name="example")

    assert exc_info.value.column == "b"
    assert exc_info.value.dataframe == "example"


def test_load_edges_wraps_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataLoadingError) as exc_info:
        load_edges(tmp_path)

    assert "elliptic_txs_edgelist.csv" in exc_info.value.source
    assert isinstance(exc_info.value.original, FileNotFoundError)
