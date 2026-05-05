"""Evaluation utilities for imbalanced fraud models."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    precision_score,
    roc_auc_score,
)

from graph_fraud.config import LABEL_COL


def binary_classification_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    y_true = y_true.astype(int)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan"),
        "pr_auc": (
            float(average_precision_score(y_true, y_score))
            if y_true.nunique() > 1
            else float("nan")
        ),
    }


def precision_at_k(y_true: pd.Series, y_score: pd.Series, *, k: int) -> float:
    """Precision among top-k transactions."""
    if k <= 0:
        raise ValueError("k must be positive")
    idx = y_score.sort_values(ascending=False).head(min(k, len(y_score))).index
    return float(precision_score(y_true.loc[idx].astype(int), [1] * len(idx), zero_division=0))


def recall_at_k(y_true: pd.Series, y_score: pd.Series, *, k: int) -> float:
    """Recall captured in top-k transactions."""
    if k <= 0:
        raise ValueError("k must be positive")
    positives = int(y_true.astype(int).sum())
    if positives == 0:
        return 0.0
    idx = y_score.sort_values(ascending=False).head(min(k, len(y_score))).index
    return float(y_true.loc[idx].astype(int).sum() / positives)


def risk_decile_summary(frame: pd.DataFrame, *, score_col: str = "risk_score") -> pd.DataFrame:
    """Summarize labels by risk decile."""
    labelled = frame[frame[LABEL_COL].notna()].copy()
    labelled["risk_decile"] = pd.qcut(
        labelled[score_col].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )
    return (
        labelled.groupby("risk_decile", observed=True)
        .agg(
            n_transactions=(LABEL_COL, "size"),
            observed_illicit_rate=(LABEL_COL, "mean"),
            mean_score=(score_col, "mean"),
        )
        .reset_index()
        .sort_values("risk_decile", ascending=False)
    )
