"""Graph construction and graph feature engineering."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from graph_fraud.config import LABEL_COL, TIME_COL, TX_ID_COL
from graph_fraud.data import require_columns


def build_directed_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from nodes and edges."""
    require_columns(nodes, [TX_ID_COL], frame_name="nodes")
    require_columns(edges, ["source", "target"], frame_name="edges")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes[TX_ID_COL].tolist())
    graph.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    return graph


def degree_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute degree-based graph features."""
    graph = build_directed_graph(nodes, edges)
    rows = []
    for tx_id in nodes[TX_ID_COL]:
        in_degree = graph.in_degree(tx_id)
        out_degree = graph.out_degree(tx_id)
        rows.append(
            {
                TX_ID_COL: tx_id,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
            }
        )
    return pd.DataFrame(rows)


def component_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute weakly connected component size."""
    graph = build_directed_graph(nodes, edges).to_undirected()
    sizes = {}
    for comp in nx.connected_components(graph):
        for node in comp:
            sizes[node] = len(comp)
    return pd.DataFrame(
        {
            TX_ID_COL: nodes[TX_ID_COL],
            "component_size": [sizes.get(x, 1) for x in nodes[TX_ID_COL]],
        }
    )


def labelled_neighbour_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_known_time_step: int | None = None,
) -> pd.DataFrame:
    """Compute neighbour-label features without using future labels."""
    require_columns(nodes, [TX_ID_COL, TIME_COL, LABEL_COL], frame_name="nodes")
    labels = {}
    for tx_id, time_step, label in nodes[[TX_ID_COL, TIME_COL, LABEL_COL]].itertuples(
        index=False, name=None
    ):
        if max_known_time_step is not None and int(time_step) > max_known_time_step:
            continue
        if pd.notna(label):
            labels[tx_id] = float(label)

    graph = build_directed_graph(nodes, edges)
    rows = []
    for tx_id in nodes[TX_ID_COL]:
        neighbours = set(graph.predecessors(tx_id)).union(graph.successors(tx_id))
        known = np.array([labels[n] for n in neighbours if n in labels], dtype=float)
        rows.append(
            {
                TX_ID_COL: tx_id,
                "known_neighbour_count": len(known),
                "illicit_neighbour_ratio": float(known.mean()) if len(known) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def graph_feature_table(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_known_time_step: int | None = None,
) -> pd.DataFrame:
    """Create graph-feature table."""
    result = degree_features(nodes, edges)
    for frame in [
        component_features(nodes, edges),
        labelled_neighbour_features(nodes, edges, max_known_time_step=max_known_time_step),
    ]:
        result = result.merge(frame, on=TX_ID_COL, how="left")
    return result
