"""Operational policy metrics for finite-capacity fraud investigation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InvestigationCosts:
    """Economic assumptions for one investigation policy."""

    review_cost: float = 1.0
    missed_illicit_cost: float = 20.0

    def __post_init__(self) -> None:
        if self.review_cost < 0.0:
            raise ValueError("review_cost cannot be negative")
        if self.missed_illicit_cost < 0.0:
            raise ValueError("missed_illicit_cost cannot be negative")


DEFAULT_INVESTIGATION_COSTS = InvestigationCosts()


def investigator_policy_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    capacity: int,
    costs: InvestigationCosts = DEFAULT_INVESTIGATION_COSTS,
) -> dict[str, float]:
    """Evaluate a top-capacity investigator policy."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if y_true.empty:
        raise ValueError("y_true and y_score cannot be empty")

    labels = y_true.astype(int)
    k = min(capacity, len(y_score))
    reviewed_idx = y_score.sort_values(ascending=False, kind="stable").head(k).index
    reviewed = labels.loc[reviewed_idx]

    captures = int(reviewed.sum())
    false_reviews = int(k - captures)
    total_illicit = int(labels.sum())
    missed_illicit = int(total_illicit - captures)

    review_loss = float(k) * costs.review_cost
    missed_loss = float(missed_illicit) * costs.missed_illicit_cost
    total_loss = review_loss + missed_loss

    precision = float(captures / k) if k else 0.0
    recall = float(captures / total_illicit) if total_illicit else 0.0

    return {
        "capacity": float(k),
        "reviews": float(k),
        "captures": float(captures),
        "false_reviews": float(false_reviews),
        "missed_illicit": float(missed_illicit),
        "precision_at_capacity": precision,
        "recall_at_capacity": recall,
        "review_cost": review_loss,
        "missed_illicit_cost": missed_loss,
        "total_expected_loss": total_loss,
    }


def cost_threshold_policy_metrics(
    y_true: pd.Series,
    y_probability: pd.Series,
    *,
    capacity: int,
    costs: InvestigationCosts = DEFAULT_INVESTIGATION_COSTS,
) -> dict[str, float]:
    """Evaluate a probability-sensitive review policy under finite capacity.

    A transaction is economically reviewable when the expected loss from
    missing an illicit case exceeds the cost of review::

        p(illicit) * missed_illicit_cost > review_cost

    If more transactions satisfy that rule than investigators can review, only
    the highest-probability observations up to ``capacity`` are selected.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if costs.missed_illicit_cost <= 0.0:
        raise ValueError("missed_illicit_cost must be positive for threshold policy")
    if len(y_true) != len(y_probability):
        raise ValueError("y_true and y_probability must have the same length")
    if y_true.empty:
        raise ValueError("y_true and y_probability cannot be empty")

    probabilities = y_probability.astype(float).clip(0.0, 1.0)
    threshold = costs.review_cost / costs.missed_illicit_cost
    eligible = probabilities[probabilities > threshold]
    reviewed_idx = eligible.sort_values(ascending=False, kind="stable").head(capacity).index

    labels = y_true.astype(int)
    reviewed = labels.loc[reviewed_idx]
    reviews = len(reviewed_idx)
    captures = int(reviewed.sum()) if reviews else 0
    false_reviews = int(reviews - captures)
    total_illicit = int(labels.sum())
    missed_illicit = int(total_illicit - captures)

    review_loss = float(reviews) * costs.review_cost
    missed_loss = float(missed_illicit) * costs.missed_illicit_cost
    total_loss = review_loss + missed_loss
    precision = float(captures / reviews) if reviews else 0.0
    recall = float(captures / total_illicit) if total_illicit else 0.0

    return {
        "capacity": float(min(capacity, len(y_probability))),
        "reviews": float(reviews),
        "cost_threshold": float(threshold),
        "captures": float(captures),
        "false_reviews": float(false_reviews),
        "missed_illicit": float(missed_illicit),
        "precision_at_capacity": precision,
        "recall_at_capacity": recall,
        "review_cost": review_loss,
        "missed_illicit_cost": missed_loss,
        "total_expected_loss": total_loss,
    }


def policy_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    capacities: list[int],
    costs: InvestigationCosts = DEFAULT_INVESTIGATION_COSTS,
) -> pd.DataFrame:
    """Evaluate one score vector over multiple investigator capacities."""
    if not capacities:
        raise ValueError("capacities cannot be empty")
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("capacities must contain only positive values")

    rows = [
        investigator_policy_metrics(
            y_true,
            y_score,
            capacity=capacity,
            costs=costs,
        )
        for capacity in capacities
    ]
    return pd.DataFrame(rows).sort_values("capacity", kind="stable").reset_index(drop=True)


def optimal_capacity(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    capacities: list[int],
    costs: InvestigationCosts = DEFAULT_INVESTIGATION_COSTS,
) -> pd.Series:
    """Return the tested capacity with minimum expected policy loss."""
    curve = policy_curve(y_true, y_score, capacities=capacities, costs=costs)
    best_position = int(curve["total_expected_loss"].to_numpy().argmin())
    return curve.iloc[best_position]
