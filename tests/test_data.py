from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_fraud.data import map_class_label, require_columns


def test_map_class_label() -> None:
    assert map_class_label("1") == 1.0
    assert map_class_label("2") == 0.0
    assert map_class_label("illicit") == 1.0
    assert map_class_label("licit") == 0.0
    assert np.isnan(map_class_label("unknown"))


def test_require_columns_raises() -> None:
    with pytest.raises(ValueError):
        require_columns(pd.DataFrame({"a": [1]}), ["a", "b"])
