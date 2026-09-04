from __future__ import annotations

import pytest

from graph_fraud.gnn_benchmark import (
    compare_static_gnn_to_baselines,
    fit_static_graphsage,
    run_calibrated_static_gnn_benchmark,
    static_gnn_hurdle_summary,
)
from graph_fraud.validation import temporal_train_calibration_test_split

torch = pytest.importorskip("torch")


def test_static_gnn_training_uses_only_history_visible_by_cutoff(synthetic_tables) -> None:
    nodes, edges = synthetic_tables
    _, _, _, train_cutoff, _ = temporal_train_calibration_test_split(
        nodes,
        calibration_time_steps=2,
        test_time_steps=2,
    )

    artifact = fit_static_graphsage(
        nodes,
        edges,
        train_cutoff=train_cutoff,
        hidden_dim=8,
        epochs=3,
        dropout=0.0,
        random_state=7,
    )

    visible_ids = set(
        nodes.loc[nodes["time_step"].astype(int) <= train_cutoff, "tx_id"].tolist()
    )
    visible_edges = edges[
        edges["source"].isin(visible_ids) & edges["target"].isin(visible_ids)
    ]

    assert artifact.train_cutoff == train_cutoff
    assert artifact.train_graph_nodes == len(visible_ids)
    assert artifact.train_graph_edges == len(visible_edges)
    assert artifact.final_train_loss >= 0.0


def test_static_gnn_benchmark_matches_temporal_partition(synthetic_tables) -> None:
    nodes, edges = synthetic_tables
    _, calibration, test, train_cutoff, calibration_cutoff = (
        temporal_train_calibration_test_split(
            nodes,
            calibration_time_steps=2,
            test_time_steps=2,
        )
    )

    result = run_calibrated_static_gnn_benchmark(
        nodes,
        edges,
        calibration_time_steps=2,
        test_time_steps=2,
        investigator_capacity=10,
        n_reliability_bins=5,
        hidden_dim=8,
        epochs=3,
        dropout=0.0,
        random_state=7,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["model"] == "graphsage"
    assert row["feature_set"] == "directed_gnn"
    assert int(row["train_cutoff"]) == train_cutoff
    assert int(row["calibration_cutoff"]) == calibration_cutoff
    assert int(row["n_calibration"]) == len(calibration)
    assert int(row["n_test"]) == len(test)
    assert 0.0 <= float(row["pr_auc"]) <= 1.0
    assert float(row["calibrated_policy_loss"]) >= 0.0


def test_static_gnn_comparison_has_empirical_hurdle(synthetic_tables) -> None:
    nodes, edges = synthetic_tables

    comparison = compare_static_gnn_to_baselines(
        nodes,
        edges,
        calibration_time_steps=2,
        test_time_steps=2,
        investigator_capacity=10,
        n_reliability_bins=5,
        hidden_dim=8,
        epochs=3,
        dropout=0.0,
        random_state=7,
    )
    hurdle = static_gnn_hurdle_summary(comparison)

    assert len(comparison) == 5
    assert (comparison["model"] == "graphsage").sum() == 1
    assert set(comparison.loc[comparison["model"] != "graphsage", "model"]) == {
        "logistic",
        "random_forest",
    }
    assert "pr_auc_delta" in hurdle.index
    assert "policy_loss_improvement" in hurdle.index
    assert isinstance(hurdle["beats_best_baseline_pr_auc"], bool)
