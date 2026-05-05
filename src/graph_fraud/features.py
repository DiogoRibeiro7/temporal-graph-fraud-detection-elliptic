"""Feature preparation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from graph_fraud.config import LABEL_COL, TIME_COL, TX_ID_COL
from graph_fraud.data import require_columns
from graph_fraud.graph import graph_feature_table


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric model features excluding ids, time and label."""
    excluded = {TX_ID_COL, TIME_COL, LABEL_COL}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def build_graph_augmented_nodes(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_known_time_step: int | None = None,
) -> pd.DataFrame:
    """Merge transaction features with graph features."""
    return nodes.merge(
        graph_feature_table(nodes, edges, max_known_time_step=max_known_time_step),
        on=TX_ID_COL,
        how="left",
    )


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Create a numeric preprocessing transformer."""
    if not columns:
        raise ValueError("At least one feature column is required")
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                columns,
            )
        ],
        remainder="drop",
    )


def make_xy(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Extract labelled features and target."""
    require_columns(frame, [*columns, LABEL_COL], frame_name="model frame")
    labelled = frame[frame[LABEL_COL].notna()].copy()
    return labelled[columns], labelled[LABEL_COL].astype(int)
