"""End-to-end graph fraud pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from graph_fraud.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, RAW_DATA_DIR
from graph_fraud.data import load_elliptic_dataset, save_tables
from graph_fraud.evaluation import binary_classification_metrics, precision_at_k, recall_at_k
from graph_fraud.features import build_graph_augmented_nodes, feature_columns, make_xy
from graph_fraud.investigation import action_table, assign_risk_tiers
from graph_fraud.models import fit_model, make_logistic_model, predict_risk
from graph_fraud.synthetic import make_synthetic_transaction_graph
from graph_fraud.validation import assert_no_temporal_leakage, temporal_train_test_split


def load_or_make_data(*, use_synthetic_if_missing: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle data or create synthetic fallback."""
    try:
        return load_elliptic_dataset(RAW_DATA_DIR)
    except FileNotFoundError:
        if not use_synthetic_if_missing:
            raise
        return make_synthetic_transaction_graph(
            n_nodes=800,
            n_features=16,
            random_state=RANDOM_SEED,
        )


def run_pipeline(
    *,
    use_synthetic_if_missing: bool = False,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, float]:
    """Run temporal graph fraud pipeline."""
    nodes, edges = load_or_make_data(use_synthetic_if_missing=use_synthetic_if_missing)
    save_tables(nodes, edges, INTERIM_DATA_DIR)

    train_base, test_base, cutoff = temporal_train_test_split(nodes, test_time_steps=3)
    assert_no_temporal_leakage(train_base, test_base)

    augmented = build_graph_augmented_nodes(nodes, edges, max_known_time_step=cutoff)
    train = augmented.loc[train_base.index]
    test = augmented.loc[test_base.index]

    cols = feature_columns(augmented)
    x_train, y_train = make_xy(train, cols)
    x_test, y_test = make_xy(test, cols)

    model = fit_model(make_logistic_model(cols, random_state=RANDOM_SEED), x_train, y_train)
    scores = predict_risk(model, x_test)

    metrics = binary_classification_metrics(y_test, scores)
    metrics["precision_at_50"] = precision_at_k(y_test, scores, k=50)
    metrics["recall_at_50"] = recall_at_k(y_test, scores, k=50)

    scored = test.copy()
    scored.loc[x_test.index, "risk_score"] = scores
    scored.loc[x_test.index, "risk_tier"] = assign_risk_tiers(scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "scored_transactions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    action_table(scored.dropna(subset=["risk_tier"])).to_csv(
        output_dir / "action_table.csv",
        index=False,
    )
    return metrics
