"""Investigation-oriented risk tier mapping."""

from __future__ import annotations

import pandas as pd

RISK_ACTIONS = {
    "low": "No immediate review. Continue normal monitoring.",
    "medium": "Review if analyst capacity allows or if linked to active cases.",
    "high": "Prioritize for analyst review and inspect transaction neighbourhood.",
    "very_high": "Escalate to senior analyst review. Do not treat score as proof.",
}


def assign_risk_tiers(
    scores: pd.Series,
    *,
    high_quantile: float = 0.90,
    very_high_quantile: float = 0.98,
) -> pd.Series:
    """Assign low, medium, high, and very-high tiers from model scores."""
    if scores.empty:
        raise ValueError("scores cannot be empty")
    high = float(scores.quantile(high_quantile))
    very_high = float(scores.quantile(very_high_quantile))
    median = float(scores.median())

    def _tier(score: float) -> str:
        if score >= very_high:
            return "very_high"
        if score >= high:
            return "high"
        if score >= median:
            return "medium"
        return "low"

    return scores.map(_tier).rename("risk_tier")


def action_table(frame: pd.DataFrame, *, tier_col: str = "risk_tier") -> pd.DataFrame:
    """Create risk-tier action table."""
    if tier_col not in frame.columns:
        raise ValueError(f"Missing tier column: {tier_col}")
    table = frame.groupby(tier_col, observed=True).size().rename("n_transactions").reset_index()
    table["suggested_action"] = table[tier_col].map(RISK_ACTIONS)
    table["caution"] = "Risk score supports review; it is not evidence of illicit activity."
    order = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
    return (
        table.assign(_order=table[tier_col].map(order))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
