from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from graph_fraud.features import build_progressive_graph_augmented_nodes


def test_future_only_edge_does_not_change_training_graph_features() -> None:
    nodes = pd.DataFrame(
        {
            "tx_id": [1, 2, 3],
            "time_step": [1, 2, 3],
            "label": [0.0, 1.0, 0.0],
            "x_1": [0.1, 0.2, 0.3],
        }
    )
    base_edges = pd.DataFrame({"source": [1], "target": [2]})
    future_edge = pd.DataFrame({"source": [1, 1], "target": [2, 3]})

    base = build_progressive_graph_augmented_nodes(
        nodes,
        base_edges,
        max_known_time_step=2,
        training_cutoff=2,
    )
    with_future_edge = build_progressive_graph_augmented_nodes(
        nodes,
        future_edge,
        max_known_time_step=2,
        training_cutoff=2,
    )

    feature_cols = [
        "tx_id",
        "in_degree",
        "out_degree",
        "total_degree",
        "component_size",
        "known_neighbour_count",
        "illicit_neighbour_ratio",
    ]
    base_training = base.loc[base["time_step"] <= 2, feature_cols].reset_index(drop=True)
    future_training = with_future_edge.loc[
        with_future_edge["time_step"] <= 2,
        feature_cols,
    ].reset_index(drop=True)

    assert_frame_equal(base_training, future_training)


def test_future_edge_appears_only_when_future_node_is_visible() -> None:
    nodes = pd.DataFrame(
        {
            "tx_id": [1, 2, 3],
            "time_step": [1, 2, 3],
            "label": [0.0, 1.0, 0.0],
            "x_1": [0.1, 0.2, 0.3],
        }
    )
    edges = pd.DataFrame({"source": [1, 1], "target": [2, 3]})

    result = build_progressive_graph_augmented_nodes(
        nodes,
        edges,
        max_known_time_step=2,
        training_cutoff=2,
    )

    row_one = result.loc[result["tx_id"] == 1].iloc[0]
    row_three = result.loc[result["tx_id"] == 3].iloc[0]

    assert int(row_one["out_degree"]) == 1
    assert int(row_three["in_degree"]) == 1
    assert int(row_three["known_neighbour_count"]) == 1
