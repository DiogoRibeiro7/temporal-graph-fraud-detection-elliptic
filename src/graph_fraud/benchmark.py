"""Leakage-aware baseline benchmarking for temporal graph fraud models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from graph_fraud.config import RANDOM_SEED, TX_ID_COL
from graph_fraud.evaluation import (
    binary_classification_metrics,
    precision_at_k,
    recall_at_k,
)
from graph_fraud.features import build_graph_augmented_nodes, feature_columns, make_xy
from graph_fraud.models import (
    fit_model,
    make_logistic_model,
    make_random_forest_model,
    predict_risk,
)
from graph_fraud.validation import assert_no_temporal_leakage, temporal_train_test_split

ModelFactory = Callable[[list[str], int], Pipeline]


@dataclass(frozen=True)
class BenchmarkSpec:
    """One model/feature-set combination in the benchmark matrix."""

    model_name: str
    feature_set: str
    model_factory: ModelFactory


def _logistic_factory(feature_cols: list[str], random_state: int) -> Pipeline:
    """Build the logistic-regression benchmark model."""
    return make_logistic_model(feature_cols, random_state=random_state)


def _random_forest_factory(feature_cols: list[str], random_state: int) -> Pipeline:
    """Build the random-forest benchmark model."""
    return make_random_forest_model(feature_cols, random_state=random_state)


DEFAULT_BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("logistic", "tabular", _logistic_factory),
    BenchmarkSpec("random_forest", "tabular", _random_forest_factory),
    BenchmarkSpec("logistic", "graph_augmented", _logistic_factory),
    BenchmarkSpec("random_forest", "graph_augmented", _random_forest_factory),
)


def _feature_frame(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    feature_set: str,
    cutoff: int,
) -> pd.DataFrame:
    """Construct one benchmark feature frame using only information known by cutoff."""
    if feature_set == "tabular":
        return nodes.copy()
    if feature_set == "graph_augmented":
        return build_graph_augmented_nodes(
            nodes,
            edges,
            max_known_time_step=cutoff,
        )
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def run_baseline_benchmark(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    test_time_steps: int = 3,
    investigator_capacity: int = 50,
    random_state: int = RANDOM_SEED,
    specs: tuple[BenchmarkSpec, ...] = DEFAULT_BENCHMARK_SPECS,
) -> pd.DataFrame:
    """Compare tabular and graph-augmented baselines on one temporal holdout.

    The train/test partition is defined once on the original node table. Feature
    frames are then joined back to those partitions by transaction ID, avoiding
    positional alignment assumptions after graph feature construction.
    """
    if investigator_capacity <= 0:
        raise ValueError("investigator_capacity must be positive")
    if not specs:
        raise ValueError("At least one benchmark specification is required")

    train_base, test_base, cutoff = temporal_train_test_split(
        nodes,
        test_time_steps=test_time_steps,
    )
    assert_no_temporal_leakage(train_base, test_base)

    train_ids = set(train_base[TX_ID_COL].tolist())
    test_ids = set(test_base[TX_ID_COL].tolist())
    rows: list[dict[str, object]] = []

    for spec in specs:
        frame = _feature_frame(
            nodes,
            edges,
            feature_set=spec.feature_set,
            cutoff=cutoff,
        )
        columns = feature_columns(frame)
        train = frame[frame[TX_ID_COL].isin(train_ids)].copy()
        test = frame[frame[TX_ID_COL].isin(test_ids)].copy()
        x_train, y_train = make_xy(train, columns)
        x_test, y_test = make_xy(test, columns)

        model = fit_model(
            spec.model_factory(columns, random_state),
            x_train,
            y_train,
        )
        scores = predict_risk(model, x_test)
        metrics = binary_classification_metrics(y_test, scores)
        k = min(investigator_capacity, len(scores))

        rows.append(
            {
                "model": spec.model_name,
                "feature_set": spec.feature_set,
                "cutoff": cutoff,
                "n_features": len(columns),
                "n_train": len(x_train),
                "n_test": len(x_test),
                **metrics,
                "precision_at_capacity": precision_at_k(y_test, scores, k=k),
                "recall_at_capacity": recall_at_k(y_test, scores, k=k),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["feature_set", "model"],
        kind="stable",
    ).reset_index(drop=True)
