"""Leakage-safe training and evaluation for the optional static GNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from dataexcept import DataValidationError

from graph_fraud.benchmark import run_calibrated_baseline_benchmark
from graph_fraud.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
)
from graph_fraud.config import LABEL_COL, RANDOM_SEED, TIME_COL, TX_ID_COL
from graph_fraud.data import require_columns
from graph_fraud.evaluation import (
    binary_classification_metrics,
    precision_at_k,
    recall_at_k,
)
from graph_fraud.features import feature_columns
from graph_fraud.gnn import StaticGraphSAGE, import_torch
from graph_fraud.policy import (
    DEFAULT_INVESTIGATION_COSTS,
    InvestigationCosts,
    cost_threshold_policy_metrics,
)
from graph_fraud.validation import temporal_train_calibration_test_split


@dataclass
class StaticGNNArtifact:
    """Trained static GNN plus train-only preprocessing state."""

    model: StaticGraphSAGE
    feature_cols: tuple[str, ...]
    medians: pd.Series
    means: pd.Series
    scales: pd.Series
    train_cutoff: int
    train_graph_nodes: int
    train_graph_edges: int
    final_train_loss: float


def _validate_graph_tables(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Validate the minimal node/edge contract required by the GNN."""
    require_columns(nodes, [TX_ID_COL, TIME_COL, LABEL_COL], frame_name="nodes")
    require_columns(edges, ["source", "target"], frame_name="edges")
    duplicate_count = int(nodes[TX_ID_COL].duplicated().sum())
    if duplicate_count:
        raise DataValidationError(
            field=TX_ID_COL,
            value=duplicate_count,
            message="Static GNN requires unique transaction identifiers.",
        )


def _fit_feature_scaler(
    history: pd.DataFrame,
) -> tuple[tuple[str, ...], pd.Series, pd.Series, pd.Series]:
    """Fit numeric imputation/scaling state using historical nodes only."""
    columns = tuple(feature_columns(history))
    if not columns:
        raise DataValidationError(
            field="features",
            value=0,
            message="Static GNN requires at least one numeric feature.",
        )

    numeric = history[list(columns)].astype(float)
    medians = numeric.median().fillna(0.0)
    filled = numeric.fillna(medians)
    means = filled.mean().fillna(0.0)
    scales = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return columns, medians, means, scales


def _normalised_feature_array(
    frame: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    medians: pd.Series,
    means: pd.Series,
    scales: pd.Series,
) -> np.ndarray:
    """Apply train-fitted numeric preprocessing to one visible graph."""
    numeric = frame[list(feature_cols)].astype(float).fillna(medians)
    scaled = (numeric - means) / scales
    return scaled.to_numpy(dtype=np.float32)


def _visible_graph_tensors(
    visible_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    artifact: StaticGNNArtifact | None = None,
    *,
    feature_cols: tuple[str, ...] | None = None,
    medians: pd.Series | None = None,
    means: pd.Series | None = None,
    scales: pd.Series | None = None,
) -> tuple[Any, Any, dict[Any, int], int]:
    """Create dense node features and sparse directed edge indices."""
    if artifact is not None:
        feature_cols = artifact.feature_cols
        medians = artifact.medians
        means = artifact.means
        scales = artifact.scales
    if feature_cols is None or medians is None or means is None or scales is None:
        raise ValueError("Feature preprocessing state is required")

    torch = import_torch()
    node_ids = visible_nodes[TX_ID_COL].tolist()
    node_to_position = {node_id: position for position, node_id in enumerate(node_ids)}
    visible_id_set = set(node_ids)

    visible_edges = edges[
        edges["source"].isin(visible_id_set) & edges["target"].isin(visible_id_set)
    ]
    sources: list[int] = []
    destinations: list[int] = []
    for source, target in visible_edges[["source", "target"]].itertuples(
        index=False,
        name=None,
    ):
        sources.append(node_to_position[source])
        destinations.append(node_to_position[target])

    if sources:
        edge_index = torch.tensor([sources, destinations], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    features = _normalised_feature_array(
        visible_nodes,
        feature_cols=feature_cols,
        medians=medians,
        means=means,
        scales=scales,
    )
    x = torch.tensor(features, dtype=torch.float32)
    return x, edge_index, node_to_position, len(sources)


def fit_static_graphsage(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    train_cutoff: int,
    hidden_dim: int = 32,
    epochs: int = 80,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> StaticGNNArtifact:
    """Fit GraphSAGE using only graph information visible by ``train_cutoff``.

    Unlabelled historical nodes may contribute features/messages, but only
    labelled historical nodes contribute to the supervised loss. No node or
    edge involving a later time step enters the training forward pass.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")

    _validate_graph_tables(nodes, edges)
    torch = import_torch()
    torch.manual_seed(random_state)

    history = nodes[nodes[TIME_COL].astype(int) <= train_cutoff].copy()
    if history.empty:
        raise DataValidationError(
            field=TIME_COL,
            value=train_cutoff,
            message="No nodes are visible at the requested GNN training cutoff.",
        )

    feature_cols, medians, means, scales = _fit_feature_scaler(history)
    x, edge_index, node_to_position, n_edges = _visible_graph_tensors(
        history,
        edges,
        feature_cols=feature_cols,
        medians=medians,
        means=means,
        scales=scales,
    )

    labelled = history[history[LABEL_COL].notna()].copy()
    labels = labelled[LABEL_COL].astype(int)
    classes = sorted(labels.unique().tolist())
    if classes != [0, 1]:
        raise DataValidationError(
            field=LABEL_COL,
            value=classes,
            message="Static GNN training requires both binary classes.",
        )

    labelled_positions = torch.tensor(
        [node_to_position[tx_id] for tx_id in labelled[TX_ID_COL]],
        dtype=torch.long,
    )
    y = torch.tensor(labels.to_numpy(), dtype=torch.long)
    class_counts = np.bincount(labels.to_numpy(), minlength=2).astype(float)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    loss_weights = torch.tensor(class_weights, dtype=torch.float32)

    model = StaticGraphSAGE(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        output_dim=2,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)

    final_loss = float("nan")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = loss_fn(logits[labelled_positions], y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    return StaticGNNArtifact(
        model=model,
        feature_cols=feature_cols,
        medians=medians,
        means=means,
        scales=scales,
        train_cutoff=int(train_cutoff),
        train_graph_nodes=len(history),
        train_graph_edges=n_edges,
        final_train_loss=final_loss,
    )


def predict_static_graphsage_by_time(
    artifact: StaticGNNArtifact,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    time_steps: list[int],
) -> pd.Series:
    """Score labelled nodes step-by-step without exposing later graph state."""
    if not time_steps:
        raise ValueError("time_steps cannot be empty")
    _validate_graph_tables(nodes, edges)
    torch = import_torch()
    artifact.model.eval()
    scores: dict[Any, float] = {}

    with torch.no_grad():
        for time_step in sorted(set(int(step) for step in time_steps)):
            visible = nodes[nodes[TIME_COL].astype(int) <= time_step].copy()
            x, edge_index, node_to_position, _ = _visible_graph_tensors(
                visible,
                edges,
                artifact,
            )
            logits = artifact.model(x, edge_index)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            targets = nodes[
                (nodes[TIME_COL].astype(int) == time_step) & nodes[LABEL_COL].notna()
            ]
            for row_index, tx_id in zip(targets.index, targets[TX_ID_COL], strict=True):
                scores[row_index] = float(probabilities[node_to_position[tx_id]])

    if not scores:
        raise DataValidationError(
            field=TIME_COL,
            value=time_steps,
            message="No labelled observations were available for GNN scoring.",
        )
    return pd.Series(scores, dtype=float, name="risk_score").sort_index()


def run_calibrated_static_gnn_benchmark(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    calibration_time_steps: int = 2,
    test_time_steps: int = 2,
    investigator_capacity: int = 50,
    investigation_costs: InvestigationCosts = DEFAULT_INVESTIGATION_COSTS,
    n_reliability_bins: int = 10,
    hidden_dim: int = 32,
    epochs: int = 80,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Train, calibrate and evaluate static GraphSAGE out of time."""
    if investigator_capacity <= 0:
        raise ValueError("investigator_capacity must be positive")
    if n_reliability_bins <= 1:
        raise ValueError("n_reliability_bins must be greater than one")

    train, calibration, test, train_cutoff, calibration_cutoff = (
        temporal_train_calibration_test_split(
            nodes,
            calibration_time_steps=calibration_time_steps,
            test_time_steps=test_time_steps,
        )
    )
    artifact = fit_static_graphsage(
        nodes,
        edges,
        train_cutoff=train_cutoff,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        random_state=random_state,
    )

    calibration_steps = sorted(calibration[TIME_COL].astype(int).unique().tolist())
    test_steps = sorted(test[TIME_COL].astype(int).unique().tolist())
    calibration_scores = predict_static_graphsage_by_time(
        artifact,
        nodes,
        edges,
        time_steps=calibration_steps,
    )
    raw_test_scores = predict_static_graphsage_by_time(
        artifact,
        nodes,
        edges,
        time_steps=test_steps,
    )
    y_calibration = nodes.loc[calibration_scores.index, LABEL_COL].astype(int)
    y_test = nodes.loc[raw_test_scores.index, LABEL_COL].astype(int)
    calibrated_scores = PlattCalibrator(random_state=random_state).fit(
        y_calibration,
        calibration_scores,
    ).transform(raw_test_scores)

    metrics = binary_classification_metrics(y_test, calibrated_scores)
    raw_brier = brier_score(y_test, raw_test_scores)
    calibrated_brier = brier_score(y_test, calibrated_scores)
    raw_ece = expected_calibration_error(
        y_test,
        raw_test_scores,
        n_bins=n_reliability_bins,
    )
    calibrated_ece = expected_calibration_error(
        y_test,
        calibrated_scores,
        n_bins=n_reliability_bins,
    )
    raw_policy = cost_threshold_policy_metrics(
        y_test,
        raw_test_scores,
        capacity=investigator_capacity,
        costs=investigation_costs,
    )
    calibrated_policy = cost_threshold_policy_metrics(
        y_test,
        calibrated_scores,
        capacity=investigator_capacity,
        costs=investigation_costs,
    )
    k = min(investigator_capacity, len(calibrated_scores))

    row: dict[str, object] = {
        "model": "graphsage",
        "feature_set": "directed_gnn",
        "train_cutoff": train_cutoff,
        "calibration_cutoff": calibration_cutoff,
        "n_features": len(artifact.feature_cols),
        "n_train": len(train),
        "n_calibration": len(calibration_scores),
        "n_test": len(raw_test_scores),
        **metrics,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": raw_brier - calibrated_brier,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "ece_improvement": raw_ece - calibrated_ece,
        "precision_at_capacity": precision_at_k(y_test, calibrated_scores, k=k),
        "recall_at_capacity": recall_at_k(y_test, calibrated_scores, k=k),
        "raw_policy_reviews": raw_policy["reviews"],
        "calibrated_policy_reviews": calibrated_policy["reviews"],
        "raw_policy_loss": raw_policy["total_expected_loss"],
        "calibrated_policy_loss": calibrated_policy["total_expected_loss"],
        "policy_loss_improvement": (
            raw_policy["total_expected_loss"] - calibrated_policy["total_expected_loss"]
        ),
        "raw_policy_recall": raw_policy["recall_at_capacity"],
        "calibrated_policy_recall": calibrated_policy["recall_at_capacity"],
        "cost_threshold": calibrated_policy["cost_threshold"],
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "train_graph_nodes": artifact.train_graph_nodes,
        "train_graph_edges": artifact.train_graph_edges,
        "final_train_loss": artifact.final_train_loss,
    }
    return pd.DataFrame([row])


def compare_static_gnn_to_baselines(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    calibration_time_steps: int = 2,
    test_time_steps: int = 2,
    investigator_capacity: int = 50,
    investigation_costs: InvestigationCosts = DEFAULT_INVESTIGATION_COSTS,
    n_reliability_bins: int = 10,
    hidden_dim: int = 32,
    epochs: int = 80,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Append static GraphSAGE to the established calibrated baseline matrix."""
    baselines = run_calibrated_baseline_benchmark(
        nodes,
        edges,
        calibration_time_steps=calibration_time_steps,
        test_time_steps=test_time_steps,
        investigator_capacity=investigator_capacity,
        investigation_costs=investigation_costs,
        n_reliability_bins=n_reliability_bins,
        random_state=random_state,
    )
    gnn = run_calibrated_static_gnn_benchmark(
        nodes,
        edges,
        calibration_time_steps=calibration_time_steps,
        test_time_steps=test_time_steps,
        investigator_capacity=investigator_capacity,
        investigation_costs=investigation_costs,
        n_reliability_bins=n_reliability_bins,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        random_state=random_state,
    )
    return pd.concat([baselines, gnn], ignore_index=True, sort=False)


def static_gnn_hurdle_summary(comparison: pd.DataFrame) -> pd.Series:
    """Quantify whether GraphSAGE clears the best simpler out-of-time baseline."""
    require_columns(
        comparison,
        ["model", "pr_auc", "calibrated_policy_loss"],
        frame_name="comparison",
    )
    gnn_rows = comparison[comparison["model"] == "graphsage"]
    baseline_rows = comparison[comparison["model"] != "graphsage"]
    if len(gnn_rows) != 1 or baseline_rows.empty:
        raise DataValidationError(
            field="model",
            value={"gnn_rows": len(gnn_rows), "baseline_rows": len(baseline_rows)},
            message="Hurdle summary requires one GraphSAGE row and at least one baseline.",
        )

    gnn_row = gnn_rows.iloc[0]
    best_baseline_pr_auc = float(baseline_rows["pr_auc"].max())
    best_baseline_policy_loss = float(baseline_rows["calibrated_policy_loss"].min())
    gnn_pr_auc = float(gnn_row["pr_auc"])
    gnn_policy_loss = float(gnn_row["calibrated_policy_loss"])
    return pd.Series(
        {
            "best_baseline_pr_auc": best_baseline_pr_auc,
            "gnn_pr_auc": gnn_pr_auc,
            "pr_auc_delta": gnn_pr_auc - best_baseline_pr_auc,
            "best_baseline_policy_loss": best_baseline_policy_loss,
            "gnn_policy_loss": gnn_policy_loss,
            "policy_loss_improvement": best_baseline_policy_loss - gnn_policy_loss,
            "beats_best_baseline_pr_auc": gnn_pr_auc > best_baseline_pr_auc,
            "beats_best_baseline_policy_loss": gnn_policy_loss < best_baseline_policy_loss,
        }
    )
