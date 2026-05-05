from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_fraud.synthetic import make_synthetic_transaction_graph  # noqa: E402


@pytest.fixture()
def synthetic_tables():
    """Return a small synthetic graph."""
    return make_synthetic_transaction_graph(n_nodes=200, n_features=8, n_time_steps=8, random_state=7)
