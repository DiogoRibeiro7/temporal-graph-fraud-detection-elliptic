from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from graph_fraud.gnn import StaticGraphSAGE, mean_predecessor_features


def test_mean_predecessor_features_respects_edge_direction() -> None:
    x = torch.tensor([[1.0], [3.0], [9.0]])
    edge_index = torch.tensor([[0, 2], [1, 1]])

    aggregated = mean_predecessor_features(x, edge_index)

    assert torch.allclose(aggregated[0], torch.tensor([0.0]))
    assert torch.allclose(aggregated[1], torch.tensor([5.0]))
    assert torch.allclose(aggregated[2], torch.tensor([0.0]))


def test_static_graphsage_forward_shape_and_gradients() -> None:
    model = StaticGraphSAGE(
        input_dim=3,
        hidden_dim=4,
        output_dim=2,
        dropout=0.0,
    )
    x = torch.randn(5, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

    logits = model(x, edge_index)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (5, 2)
    assert x.grad is not None


def test_static_graphsage_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        StaticGraphSAGE(input_dim=0)
