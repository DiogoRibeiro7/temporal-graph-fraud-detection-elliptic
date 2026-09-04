from __future__ import annotations

import pandas as pd

from graph_fraud.policy import (
    InvestigationCosts,
    cost_threshold_policy_metrics,
    investigator_policy_metrics,
    optimal_capacity,
    policy_curve,
)


def test_investigator_policy_metrics_accounting() -> None:
    labels = pd.Series([1, 0, 1, 0, 1], index=[10, 11, 12, 13, 14])
    scores = pd.Series([0.9, 0.8, 0.7, 0.2, 0.1], index=labels.index)
    costs = InvestigationCosts(review_cost=2.0, missed_illicit_cost=10.0)

    metrics = investigator_policy_metrics(
        labels,
        scores,
        capacity=2,
        costs=costs,
    )

    assert metrics["captures"] == 1.0
    assert metrics["false_reviews"] == 1.0
    assert metrics["missed_illicit"] == 2.0
    assert metrics["precision_at_capacity"] == 0.5
    assert metrics["recall_at_capacity"] == 1.0 / 3.0
    assert metrics["review_cost"] == 4.0
    assert metrics["missed_illicit_cost"] == 20.0
    assert metrics["total_expected_loss"] == 24.0


def test_capacity_is_capped_by_available_transactions() -> None:
    labels = pd.Series([1, 0, 1])
    scores = pd.Series([0.9, 0.5, 0.1])

    metrics = investigator_policy_metrics(labels, scores, capacity=10)

    assert metrics["capacity"] == 3.0
    assert metrics["captures"] == 2.0
    assert metrics["missed_illicit"] == 0.0


def test_cost_threshold_policy_is_probability_sensitive() -> None:
    labels = pd.Series([1, 0, 1, 0], index=[10, 11, 12, 13])
    probabilities = pd.Series([0.9, 0.4, 0.3, 0.1], index=labels.index)
    costs = InvestigationCosts(review_cost=2.0, missed_illicit_cost=10.0)

    metrics = cost_threshold_policy_metrics(
        labels,
        probabilities,
        capacity=4,
        costs=costs,
    )

    assert metrics["cost_threshold"] == 0.2
    assert metrics["reviews"] == 3.0
    assert metrics["captures"] == 2.0
    assert metrics["false_reviews"] == 1.0
    assert metrics["missed_illicit"] == 0.0
    assert metrics["total_expected_loss"] == 6.0


def test_cost_threshold_policy_respects_capacity() -> None:
    labels = pd.Series([1, 0, 1, 0])
    probabilities = pd.Series([0.9, 0.8, 0.7, 0.6])
    costs = InvestigationCosts(review_cost=1.0, missed_illicit_cost=10.0)

    metrics = cost_threshold_policy_metrics(
        labels,
        probabilities,
        capacity=2,
        costs=costs,
    )

    assert metrics["reviews"] == 2.0
    assert metrics["captures"] == 1.0
    assert metrics["missed_illicit"] == 1.0
    assert metrics["total_expected_loss"] == 12.0


def test_policy_curve_and_optimal_capacity() -> None:
    labels = pd.Series([1, 0, 1, 0, 1])
    scores = pd.Series([0.9, 0.8, 0.7, 0.2, 0.1])
    costs = InvestigationCosts(review_cost=2.0, missed_illicit_cost=10.0)

    curve = policy_curve(
        labels,
        scores,
        capacities=[1, 2, 3, 5],
        costs=costs,
    )
    best = optimal_capacity(
        labels,
        scores,
        capacities=[1, 2, 3, 5],
        costs=costs,
    )

    assert curve["capacity"].tolist() == [1.0, 2.0, 3.0, 5.0]
    assert best["total_expected_loss"] == curve["total_expected_loss"].min()
    assert best["capacity"] == 5.0
