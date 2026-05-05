from __future__ import annotations

import pandas as pd

from graph_fraud.investigation import action_table, assign_risk_tiers


def test_risk_tiers_and_action_table() -> None:
    scores = pd.Series([0.01, 0.2, 0.5, 0.8, 0.99], name="risk_score")
    tiers = assign_risk_tiers(scores, high_quantile=0.7, very_high_quantile=0.9)
    assert set(tiers.unique()).issubset({"low", "medium", "high", "very_high"})

    table = action_table(pd.DataFrame({"risk_tier": tiers}))
    assert "suggested_action" in table.columns
    assert "caution" in table.columns
