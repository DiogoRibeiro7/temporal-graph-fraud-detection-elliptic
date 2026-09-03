from __future__ import annotations

import pandas as pd
import pytest
from dataexcept import DataValidationError

from graph_fraud.calibration import PlattCalibrator, brier_score, reliability_table
from graph_fraud.config import LABEL_COL, TIME_COL
from graph_fraud.models import fit_model, make_logistic_model, predict_risk
from graph_fraud.features import feature_columns, make_xy
from graph_fraud.validation import temporal_train_calibration_test_split


def test_temporal_calibration_split_is_strictly_ordered(synthetic_tables) -> None:
    nodes, _ = synthetic_tables
    train, calibration, test, train_cutoff, calibration_cutoff = (
        temporal_train_calibration_test_split(
            nodes,
            calibration_time_steps=2,
            test_time_steps=2,
        )
    )

    assert train[TIME_COL].max() <= train_cutoff
    assert calibration[TIME_COL].min() > train_cutoff
    assert calibration[TIME_COL].max() <= calibration_cutoff
    assert test[TIME_COL].min() > calibration_cutoff


def test_platt_calibrator_and_reliability_table(synthetic_tables) -> None:
    nodes, _ = synthetic_tables
    train, calibration, test, _, _ = temporal_train_calibration_test_split(
        nodes,
        calibration_time_steps=2,
        test_time_steps=2,
    )
    columns = feature_columns(nodes)
    x_train, y_train = make_xy(train, columns)
    x_calibration, y_calibration = make_xy(calibration, columns)
    x_test, y_test = make_xy(test, columns)

    model = fit_model(make_logistic_model(columns), x_train, y_train)
    calibration_scores = predict_risk(model, x_calibration)
    test_scores = predict_risk(model, x_test)
    calibrated_scores = PlattCalibrator().fit(
        y_calibration,
        calibration_scores,
    ).transform(test_scores)

    assert calibrated_scores.between(0.0, 1.0).all()
    assert brier_score(y_test, calibrated_scores) >= 0.0

    table = reliability_table(y_test, calibrated_scores, n_bins=5)
    assert int(table["n"].sum()) == len(y_test)
    assert set(table.columns) == {
        "bin",
        "n",
        "mean_predicted_risk",
        "observed_illicit_rate",
    }


def test_platt_calibrator_rejects_single_class() -> None:
    labels = pd.Series([0, 0, 0], name=LABEL_COL)
    scores = pd.Series([0.1, 0.2, 0.3], name="risk_score")

    with pytest.raises(DataValidationError, match="requires both binary classes"):
        PlattCalibrator().fit(labels, scores)
