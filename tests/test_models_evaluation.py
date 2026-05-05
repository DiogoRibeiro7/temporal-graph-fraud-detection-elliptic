from __future__ import annotations

from graph_fraud.evaluation import binary_classification_metrics, precision_at_k, recall_at_k
from graph_fraud.features import build_graph_augmented_nodes, feature_columns, make_xy
from graph_fraud.models import fit_model, make_logistic_model, predict_risk
from graph_fraud.validation import temporal_train_test_split


def test_model_training_and_metrics(synthetic_tables) -> None:
    nodes, edges = synthetic_tables
    train_base, test_base, cutoff = temporal_train_test_split(nodes, test_time_steps=2)
    augmented = build_graph_augmented_nodes(nodes, edges, max_known_time_step=cutoff)
    cols = feature_columns(augmented)

    x_train, y_train = make_xy(augmented.loc[train_base.index], cols)
    x_test, y_test = make_xy(augmented.loc[test_base.index], cols)

    model = fit_model(make_logistic_model(cols), x_train, y_train)
    scores = predict_risk(model, x_test)
    metrics = binary_classification_metrics(y_test, scores)

    assert "pr_auc" in metrics
    assert precision_at_k(y_test, scores, k=10) >= 0.0
    assert recall_at_k(y_test, scores, k=10) >= 0.0
