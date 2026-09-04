"""Graph construction and graph feature engineering."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from graph_fraud.config import LABEL_COL, TIME_COL, TX_ID_COL
from graph_fraud.data import require_columns


def _visible_graph_inputs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_topology_time_step: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict graph topology to nodes visible by a temporal cutoff."""
    require_columns(nodes, [TX_ID_COL], frame_name="nodes")
    require_columns(edges, ["source", "target"], frame_name="edges")
    if max_topology_time_step is None:
        return nodes, edges

    require_columns(nodes, [TIME_COL], frame_name="nodes")
    visible_nodes = nodes[nodes[TIME_COL].astype(int) <= max_topology_time_step].copy()
    visible_ids = set(visible_nodes[TX_ID_COL].tolist())
    visible_edges = edges[
        edges["source"].isin(visible_ids) & edges["target"].isin(visible_ids)
    ].copy()
    return visible_nodes, visible_edges


def build_directed_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from nodes and edges."""
    require_columns(nodes, [TX_ID_COL], frame_name="nodes")
    require_columns(edges, ["source", "target"], frame_name="edges")
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(nodes[TX_ID_COL].tolist())
    graph.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    return graph


def degree_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_topology_time_step: int | None = None,
) -> pd.DataFrame:
    """Compute degree features from topology visible by the requested cutoff."""
    visible_nodes, visible_edges = _visible_graph_inputs(
        nodes,
        edges,
        max_topology_time_step=max_topology_time_step,
    )
    graph = build_directed_graph(visible_nodes, visible_edges)
    rows = []
    for tx_id in visible_nodes[TX_ID_COL]:
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


def component_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_topology_time_step: int | None = None,
) -> pd.DataFrame:
    """Compute weak-component size from topology visible by the cutoff."""
    visible_nodes, visible_edges = _visible_graph_inputs(
        nodes,
        edges,
        max_topology_time_step=max_topology_time_step,
    )
    graph = build_directed_graph(visible_nodes, visible_edges).to_undirected()
    sizes = {}
    for comp in nx.connected_components(graph):
        for node in comp:
            sizes[node] = len(comp)
    return pd.DataFrame(
        {
            TX_ID_COL: visible_nodes[TX_ID_COL],
            "component_size": [sizes.get(x, 1) for x in visible_nodes[TX_ID_COL]],
        }
    )


def labelled_neighbour_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_known_time_step: int | None = None,
    max_topology_time_step: int | None = None,
) -> pd.DataFrame:
    """Compute neighbour-label features without future labels or topology."""
    require_columns(nodes, [TX_ID_COL, TIME_COL, LABEL_COL], frame_name="nodes")
    visible_nodes, visible_edges = _visible_graph_inputs(
        nodes,
        edges,
        max_topology_time_step=max_topology_time_step,
    )
    labels = {}
    for tx_id, time_step, label in visible_nodes[[TX_ID_COL, TIME_COL, LABEL_COL]].itertuples(
        index=False, name=None
    ):
        if max_known_time_step is not None and int(time_step) > max_known_time_step:
            continue
        if pd.notna(label):
            labels[tx_id] = float(label)

    graph = build_directed_graph(visible_nodes, visible_edges)
    rows = []
    for tx_id in visible_nodes[TX_ID_COL]:
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
    max_topology_time_step: int | None = None,
) -> pd.DataFrame:
    """Create graph features using only labels/topology available by cutoffs."""
    result = degree_features(
        nodes,
        edges,
        max_topology_time_step=max_topology_time_step,
    )
    for frame in [
        component_features(
            nodes,
            edges,
            max_topology_time_step=max_topology_time_step,
        ),
        labelled_neighbour_features(
            nodes,
            edges,
            max_known_time_step=max_known_time_step,
            max_topology_time_step=max_topology_time_step,
        ),
    ]:
        result = result.merge(frame, on=TX_ID_COL, how="left")
    return result
