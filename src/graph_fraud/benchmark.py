"""Leakage-aware baseline benchmarking for temporal graph fraud models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from graph_fraud.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
)
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
from graph_fraud.policy import (
    DEFAULT_INVESTIGATION_COSTS,
    InvestigationCosts,
    cost_threshold_policy_metrics,
)
from graph_fraud.validation import (
    assert_no_temporal_leakage,
    temporal_train_calibration_test_split,
    temporal_train_test_split,
)

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
    """Compare tabular and graph-augmented baselines on one temporal holdout."""
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


def run_calibrated_baseline_benchmark(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    calibration_time_steps: int = 2,
    test_time_steps: int = 2,
    investigator_capacity: int = 50,
    investigation_costs: InvestigationCosts = DEFAULT_INVESTIGATION_COSTS,
    n_reliability_bins: int = 10,
    random_state: int = RANDOM_SEED,
    specs: tuple[BenchmarkSpec, ...] = DEFAULT_BENCHMARK_SPECS,
) -> pd.DataFrame:
    """Compare raw and calibrated probabilities on a strict temporal split.

    Graph-derived label features are frozen at the training cutoff. Raw and
    calibrated probabilities are evaluated under identical cost and capacity
    assumptions using a probability-sensitive investigation rule.
    """
    if investigator_capacity <= 0:
        raise ValueError("investigator_capacity must be positive")
    if n_reliability_bins <= 1:
        raise ValueError("n_reliability_bins must be greater than one")
    if not specs:
        raise ValueError("At least one benchmark specification is required")

    train_base, calibration_base, test_base, train_cutoff, calibration_cutoff = (
        temporal_train_calibration_test_split(
            nodes,
            calibration_time_steps=calibration_time_steps,
            test_time_steps=test_time_steps,
        )
    )
    train_ids = set(train_base[TX_ID_COL].tolist())
    calibration_ids = set(calibration_base[TX_ID_COL].tolist())
    test_ids = set(test_base[TX_ID_COL].tolist())
    rows: list[dict[str, object]] = []

    for spec in specs:
        frame = _feature_frame(
            nodes,
            edges,
            feature_set=spec.feature_set,
            cutoff=train_cutoff,
        )
        columns = feature_columns(frame)
        train = frame[frame[TX_ID_COL].isin(train_ids)].copy()
        calibration = frame[frame[TX_ID_COL].isin(calibration_ids)].copy()
        test = frame[frame[TX_ID_COL].isin(test_ids)].copy()
        x_train, y_train = make_xy(train, columns)
        x_calibration, y_calibration = make_xy(calibration, columns)
        x_test, y_test = make_xy(test, columns)

        model = fit_model(
            spec.model_factory(columns, random_state),
            x_train,
            y_train,
        )
        calibration_scores = predict_risk(model, x_calibration)
        raw_test_scores = predict_risk(model, x_test)
        calibrated_test_scores = PlattCalibrator(random_state=random_state).fit(
            y_calibration,
            calibration_scores,
        ).transform(raw_test_scores)
        metrics = binary_classification_metrics(y_test, calibrated_test_scores)
        k = min(investigator_capacity, len(calibrated_test_scores))
        raw_brier = brier_score(y_test, raw_test_scores)
        calibrated_brier = brier_score(y_test, calibrated_test_scores)
        raw_ece = expected_calibration_error(
            y_test,
            raw_test_scores,
            n_bins=n_reliability_bins,
        )
        calibrated_ece = expected_calibration_error(
            y_test,
            calibrated_test_scores,
            n_bins=n_reliability_bins,
        )
        raw_policy = cost_threshold_policy_metrics(
            y_test,
            raw_test_scores,
            capacity=investigator_capacity,
            costs=investigation_costs,
        )
        calibrated_policy = cost_threshold_policy_metrics(
            y_test,
            calibrated_test_scores,
            capacity=investigator_capacity,
            costs=investigation_costs,
        )

        rows.append(
            {
                "model": spec.model_name,
                "feature_set": spec.feature_set,
                "train_cutoff": train_cutoff,
                "calibration_cutoff": calibration_cutoff,
                "n_features": len(columns),
                "n_train": len(x_train),
                "n_calibration": len(x_calibration),
                "n_test": len(x_test),
                **metrics,
                "raw_brier": raw_brier,
                "calibrated_brier": calibrated_brier,
                "brier_improvement": raw_brier - calibrated_brier,
                "raw_ece": raw_ece,
                "calibrated_ece": calibrated_ece,
                "ece_improvement": raw_ece - calibrated_ece,
                "precision_at_capacity": precision_at_k(
                    y_test,
                    calibrated_test_scores,
                    k=k,
                ),
                "recall_at_capacity": recall_at_k(
                    y_test,
                    calibrated_test_scores,
                    k=k,
                ),
                "raw_policy_reviews": raw_policy["reviews"],
                "calibrated_policy_reviews": calibrated_policy["reviews"],
                "raw_policy_loss": raw_policy["total_expected_loss"],
                "calibrated_policy_loss": calibrated_policy["total_expected_loss"],
                "policy_loss_improvement": (
                    raw_policy["total_expected_loss"]
                    - calibrated_policy["total_expected_loss"]
                ),
                "raw_policy_recall": raw_policy["recall_at_capacity"],
                "calibrated_policy_recall": calibrated_policy["recall_at_capacity"],
                "cost_threshold": calibrated_policy["cost_threshold"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["feature_set", "model"],
        kind="stable",
    ).reset_index(drop=True)
