"""Model factories and training helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from graph_fraud.features import make_preprocessor


def make_logistic_model(feature_cols: list[str], *, random_state: int = 42) -> Pipeline:
    """Create a balanced logistic regression baseline."""
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(feature_cols)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest_model(feature_cols: list[str], *, random_state: int = 42) -> Pipeline:
    """Create a random forest comparison model."""
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(feature_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_model(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit a model pipeline."""
    if x_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty")
    return model.fit(x_train, y_train)


def predict_risk(model: Pipeline, x: pd.DataFrame) -> pd.Series:
    """Predict positive-class risk scores."""
    if x.empty:
        raise ValueError("x cannot be empty")
    return pd.Series(model.predict_proba(x)[:, 1], index=x.index, name="risk_score")
