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
    max_topology_time_step: int | None = None,
) -> pd.DataFrame:
    """Merge transaction features with cutoff-aware graph features."""
    return nodes.merge(
        graph_feature_table(
            nodes,
            edges,
            max_known_time_step=max_known_time_step,
            max_topology_time_step=max_topology_time_step,
        ),
        on=TX_ID_COL,
        how="left",
    )


def build_progressive_graph_augmented_nodes(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_known_time_step: int,
    training_cutoff: int,
) -> pd.DataFrame:
    """Build graph features with training-snapshot and progressive holdout topology.

    Rows at or before ``training_cutoff`` share the graph snapshot available at
    that cutoff. Later rows are recomputed at their own time step so a row at
    time ``t`` never uses nodes or edges that only become visible after ``t``.
    Label-derived neighbour features remain frozen at ``max_known_time_step``.
    """
    require_columns(nodes, [TX_ID_COL, TIME_COL], frame_name="nodes")
    history = nodes[nodes[TIME_COL].astype(int) <= training_cutoff].copy()
    history_augmented = build_graph_augmented_nodes(
        history,
        edges,
        max_known_time_step=max_known_time_step,
        max_topology_time_step=training_cutoff,
    )
    frames = [history_augmented]

    future_steps = sorted(
        nodes.loc[nodes[TIME_COL].astype(int) > training_cutoff, TIME_COL]
        .astype(int)
        .unique()
        .tolist()
    )
    for time_step in future_steps:
        visible_nodes = nodes[nodes[TIME_COL].astype(int) <= time_step].copy()
        augmented = build_graph_augmented_nodes(
            visible_nodes,
            edges,
            max_known_time_step=max_known_time_step,
            max_topology_time_step=time_step,
        )
        frames.append(augmented[augmented[TIME_COL].astype(int) == time_step].copy())

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([TIME_COL, TX_ID_COL], kind="stable")
        .reset_index(drop=True)
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
