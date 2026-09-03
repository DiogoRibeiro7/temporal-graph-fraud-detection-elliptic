"""Synthetic graph generator for tests and local demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_transaction_graph(
    *,
    n_nodes: int = 500,
    n_features: int = 12,
    n_time_steps: int = 10,
    illicit_rate: float = 0.08,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a synthetic transaction graph with feature and graph signal."""
    if n_nodes <= 20:
        raise ValueError("n_nodes must be greater than 20")
    rng = np.random.default_rng(random_state)
    tx_ids = np.arange(1, n_nodes + 1)
    time_steps = rng.integers(1, n_time_steps + 1, size=n_nodes)
    x = rng.normal(size=(n_nodes, n_features))
    score = 1.3 * x[:, 0] - 0.9 * x[:, 1] + rng.normal(scale=0.5, size=n_nodes)
    labels = (score >= np.quantile(score, 1 - illicit_rate)).astype(float)
    labels[rng.random(n_nodes) < 0.25] = np.nan

    nodes = pd.DataFrame({"tx_id": tx_ids, "time_step": time_steps, "label": labels})
    for i in range(n_features):
        nodes[f"x_{i + 1}"] = x[:, i]

    edges: list[tuple[int, int]] = []
    for source in tx_ids:
        n_targets = min(int(rng.poisson(1.5)), n_nodes - 1)
        targets = rng.choice(tx_ids[tx_ids != source], size=n_targets, replace=False)
        edges.extend((int(source), int(target)) for target in targets)

    return (
        nodes.sort_values("time_step").reset_index(drop=True),
        pd.DataFrame(edges, columns=["source", "target"])
        .drop_duplicates()
        .reset_index(drop=True),
    )
