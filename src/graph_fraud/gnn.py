"""Optional PyTorch GCN scaffold."""

from __future__ import annotations

from typing import Any


def _import_torch() -> Any:
    """Import torch or explain how to install it."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required. Install with `poetry install --with dev,gnn`.") from exc
    return torch


class SimpleGCN:
    """Small two-layer GCN wrapper for experimentation."""

    def __init__(self, *, input_dim: int, hidden_dim: int = 32, output_dim: int = 2) -> None:
        torch = _import_torch()
        nn = torch.nn

        class _GCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(input_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x: Any, adjacency: Any) -> Any:
                h = torch.relu(self.linear1(adjacency @ x))
                return self.linear2(adjacency @ h)

        self.model = _GCN()

    def __call__(self, x: Any, adjacency: Any) -> Any:
        """Run a forward pass."""
        return self.model(x, adjacency)
