"""Optional static GraphSAGE-style model for directed transaction graphs."""

from __future__ import annotations

from typing import Any


def import_torch() -> Any:
    """Import torch or explain how to install it."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required. Install with `poetry install --with dev,gnn`."
        ) from exc
    return torch


def mean_predecessor_features(x: Any, edge_index: Any) -> Any:
    """Aggregate predecessor features by arithmetic mean.

    ``edge_index[0]`` contains sources and ``edge_index[1]`` destinations, so
    information flows only along the directed transaction edge.
    """
    torch = import_torch()
    src = edge_index[0].long()
    dst = edge_index[1].long()
    aggregated = torch.zeros_like(x)
    aggregated.index_add_(0, dst, x[src])

    degree = torch.zeros(
        x.shape[0],
        dtype=x.dtype,
        device=x.device,
    )
    degree.index_add_(
        0,
        dst,
        torch.ones(dst.shape[0], dtype=x.dtype, device=x.device),
    )
    return aggregated / degree.clamp_min(1.0).unsqueeze(1)


class StaticGraphSAGE:
    """Two-layer mean-aggregation GraphSAGE classifier.

    ``edge_index`` is expected to have shape ``[2, n_edges]`` with source nodes
    in row 0 and destination nodes in row 1. Aggregation is directed: each node
    receives messages from its predecessors only.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("Model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        torch = import_torch()
        nn = torch.nn

        class _GraphSAGE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim * 2, output_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: Any, edge_index: Any) -> Any:
                neighbours = mean_predecessor_features(x, edge_index)
                h = torch.relu(self.linear1(torch.cat([x, neighbours], dim=1)))
                h = self.dropout(h)
                neighbours_h = mean_predecessor_features(h, edge_index)
                return self.linear2(torch.cat([h, neighbours_h], dim=1))

        self.model = _GraphSAGE()

    def __call__(self, x: Any, edge_index: Any) -> Any:
        """Run a forward pass."""
        return self.model(x, edge_index)

    def parameters(self) -> Any:
        """Return trainable model parameters."""
        return self.model.parameters()

    def train(self) -> StaticGraphSAGE:
        """Put the wrapped model in training mode."""
        self.model.train()
        return self

    def eval(self) -> StaticGraphSAGE:
        """Put the wrapped model in evaluation mode."""
        self.model.eval()
        return self

    def state_dict(self) -> Any:
        """Return the wrapped model state for reproducibility checks."""
        return self.model.state_dict()
