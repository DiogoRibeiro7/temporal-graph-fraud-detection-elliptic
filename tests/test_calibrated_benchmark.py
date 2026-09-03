from __future__ import annotations

from graph_fraud.benchmark import run_calibrated_baseline_benchmark


def test_calibrated_benchmark_matrix(synthetic_tables) -> None:
    nodes, edges = synthetic_tables
    result = run_calibrated_baseline_benchmark(
        nodes,
        edges,
        calibration_time_steps=2,
        test_time_steps=2,
        investigator_capacity=10,
        n_reliability_bins=5,
    )

    assert len(result) == 4
    assert set(result["model"]) == {"logistic", "random_forest"}
    assert set(result["feature_set"]) == {"tabular", "graph_augmented"}
    assert result["train_cutoff"].nunique() == 1
    assert result["calibration_cutoff"].nunique() == 1
    assert result["n_train"].nunique() == 1
    assert result["n_calibration"].nunique() == 1
    assert result["n_test"].nunique() == 1
    assert (result["raw_brier"] >= 0.0).all()
    assert (result["calibrated_brier"] >= 0.0).all()
    assert (result["raw_ece"] >= 0.0).all()
    assert (result["calibrated_ece"] >= 0.0).all()

    tabular_features = result.loc[
        result["feature_set"] == "tabular",
        "n_features",
    ].unique()
    graph_features = result.loc[
        result["feature_set"] == "graph_augmented",
        "n_features",
    ].unique()
    assert len(tabular_features) == 1
    assert len(graph_features) == 1
    assert int(graph_features[0]) > int(tabular_features[0])
