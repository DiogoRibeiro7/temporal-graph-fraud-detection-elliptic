from __future__ import annotations

from graph_fraud.graph import degree_features, graph_feature_table


def test_degree_features(synthetic_tables) -> None:
    nodes, edges = synthetic_tables
    features = degree_features(nodes, edges)
    assert "in_degree" in features.columns
    assert "out_degree" in features.columns
    assert len(features) == len(nodes)


def test_graph_feature_table(synthetic_tables) -> None:
    nodes, edges = synthetic_tables
    table = graph_feature_table(nodes, edges, max_known_time_step=4)
    assert "known_neighbour_count" in table.columns
    assert "illicit_neighbour_ratio" in table.columns
    assert "component_size" in table.columns
