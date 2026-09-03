"""Probability calibration utilities for fraud risk scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from dataexcept import DataValidationError
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    """Fit a logistic calibration map on a dedicated holdout slice."""

    random_state: int = 42
    _model: Any = None

    def fit(self, y_true: pd.Series, y_score: pd.Series) -> PlattCalibrator:
        """Fit the calibrator using calibration labels and raw risk scores."""
        if len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length")
        if y_true.empty:
            raise DataValidationError(
                field="calibration_labels",
                value=0,
                message="Calibration data cannot be empty.",
            )

        labels = y_true.astype(int)
        if labels.nunique() < 2:
            raise DataValidationError(
                field="calibration_labels",
                value=sorted(labels.unique().tolist()),
                message="Probability calibration requires both binary classes.",
            )

        x = np.asarray(y_score, dtype=float).reshape(-1, 1)
        model = LogisticRegression(random_state=self.random_state)
        model.fit(x, labels.to_numpy())
        self._model = model
        return self

    def transform(self, y_score: pd.Series) -> pd.Series:
        """Transform raw risk scores into calibrated probabilities."""
        if self._model is None:
            raise RuntimeError("Calibrator must be fitted before transform")
        x = np.asarray(y_score, dtype=float).reshape(-1, 1)
        probabilities = self._model.predict_proba(x)[:, 1]
        return pd.Series(probabilities, index=y_score.index, name="calibrated_risk_score")


def brier_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """Return the mean squared probability error for binary outcomes."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if y_true.empty:
        raise ValueError("y_true and y_score cannot be empty")
    truth = y_true.astype(float).to_numpy()
    score = y_score.astype(float).to_numpy()
    return float(np.mean((score - truth) ** 2))


def reliability_table(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed risk by equal-width probability bins."""
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than one")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if y_true.empty:
        raise ValueError("y_true and y_score cannot be empty")

    frame = pd.DataFrame(
        {
            "label": y_true.astype(int),
            "score": y_score.astype(float).clip(0.0, 1.0),
        }
    )
    frame["bin"] = pd.cut(
        frame["score"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        labels=False,
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            n=("label", "size"),
            mean_predicted_risk=("score", "mean"),
            observed_illicit_rate=("label", "mean"),
        )
        .reset_index()
    )
