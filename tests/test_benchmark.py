from __future__ import annotations

from graph_fraud.benchmark import run_baseline_benchmark


def test_baseline_benchmark_has_expected_matrix(synthetic_tables) -> None:
    nodes, edges = synthetic_tables

    result = run_baseline_benchmark(
        nodes,
        edges,
        test_time_steps=2,
        investigator_capacity=10,
    )

    assert len(result) == 4
    assert set(result["model"]) == {"logistic", "random_forest"}
    assert set(result["feature_set"]) == {"tabular", "graph_augmented"}
    assert result["cutoff"].nunique() == 1
    assert result["n_train"].nunique() == 1
    assert result["n_test"].nunique() == 1
    assert result["precision_at_capacity"].between(0.0, 1.0).all()
    assert result["recall_at_capacity"].between(0.0, 1.0).all()


def test_graph_augmented_benchmark_adds_features(synthetic_tables) -> None:
    nodes, edges = synthetic_tables

    result = run_baseline_benchmark(nodes, edges, test_time_steps=2)
    tabular_features = int(
        result.loc[result["feature_set"] == "tabular", "n_features"].iloc[0]
    )
    graph_features = int(
        result.loc[result["feature_set"] == "graph_augmented", "n_features"].iloc[0]
    )

    assert graph_features > tabular_features
