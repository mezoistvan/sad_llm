"""Tests for steering/hooks.py.

We don't need a real LLM. A toy module whose forward pass returns the input
hidden state lets us verify that the hook adds exactly what we expect to the
residual stream.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from steering.hooks import MultiVectorSteeringHook, steer  # noqa: E402


class IdentityLayer(nn.Module):
    """Stand-in for a Llama transformer layer. Returns (hidden, ...) tuple
    so the hook's tuple-handling code path is exercised."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (x, None)


class ToyModel(nn.Module):
    """Wrapper exposing `model.layers[i]` so `steer(model, i, ...)` works."""

    def __init__(self, n_layers: int = 4, hidden: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([IdentityLayer(hidden) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)[0]
        return x


def test_empty_vector_list_is_baseline():
    """No vectors installed -> output bit-identical to no hook."""
    torch.manual_seed(0)
    inner = ToyModel()
    x = torch.randn(2, 5, 8)
    baseline = inner(x.clone())

    hook = MultiVectorSteeringHook(inner.layers[2], [], layer_norm=10.0)
    with hook:
        out = inner(x.clone())
    assert torch.equal(out, baseline)


def test_zero_coefficient_is_baseline():
    """Vector with coefficient 0 -> output bit-identical to no hook."""
    torch.manual_seed(1)
    inner = ToyModel()
    x = torch.randn(2, 5, 8)
    baseline = inner(x.clone())
    vec = torch.randn(8)

    with steer(inner, 2, [(vec, 0.0)], layer_norm=12.5):
        out = inner(x.clone())
    assert torch.equal(out, baseline)


def test_anti_parallel_vectors_cancel():
    """Two vectors pointing in opposite directions with equal coefficients
    sum to zero and produce baseline output."""
    torch.manual_seed(2)
    inner = ToyModel()
    x = torch.randn(2, 5, 8)
    baseline = inner(x.clone())
    vec = torch.randn(8)

    with steer(inner, 1, [(vec, 0.5), (-vec, 0.5)], layer_norm=10.0):
        out = inner(x.clone())
    assert torch.allclose(out, baseline, atol=1e-6)


def test_single_vector_adds_expected_delta():
    """coef * layer_norm * vector should be added to every position of the
    residual stream output of the steered layer."""
    torch.manual_seed(3)
    inner = ToyModel()
    x = torch.randn(2, 5, 8)
    baseline = inner(x.clone())
    vec = torch.randn(8)
    coef = 0.5
    norm = 4.0
    expected_delta = coef * norm * vec

    with steer(inner, 3, [(vec, coef)], layer_norm=norm):
        out = inner(x.clone())

    diff = out - baseline
    # The delta should be added at the LAST steered layer (index 3, the output
    # layer); it propagates through identity layers unchanged.
    assert diff.shape == (2, 5, 8)
    for b in range(2):
        for s in range(5):
            assert torch.allclose(diff[b, s, :], expected_delta, atol=1e-6)


def test_two_vectors_compose_additively():
    """Two distinct vectors with their coefficients should produce a sum delta."""
    torch.manual_seed(4)
    inner = ToyModel()
    x = torch.randn(2, 5, 8)
    baseline = inner(x.clone())
    v1 = torch.randn(8)
    v2 = torch.randn(8)
    norm = 5.0

    with steer(inner, 0, [(v1, 0.3), (v2, 0.7)], layer_norm=norm):
        out = inner(x.clone())

    expected = 0.3 * norm * v1 + 0.7 * norm * v2
    diff = out - baseline
    for b in range(2):
        for s in range(5):
            assert torch.allclose(diff[b, s, :], expected, atol=1e-6)


def test_hook_removed_after_context():
    """After exiting the context manager, the model behaves exactly like baseline."""
    torch.manual_seed(5)
    inner = ToyModel()
    x = torch.randn(2, 5, 8)
    baseline = inner(x.clone())
    vec = torch.randn(8)

    with steer(inner, 2, [(vec, 0.5)], layer_norm=10.0):
        steered = inner(x.clone())
    after = inner(x.clone())

    assert not torch.allclose(steered, baseline, atol=1e-3)
    assert torch.equal(after, baseline)


def test_rejects_non_1d_vector():
    """The hook should refuse anything other than a 1-D vector."""
    inner = ToyModel()
    bad = torch.randn(8, 8)
    try:
        MultiVectorSteeringHook(inner.layers[0], [(bad, 0.5)], layer_norm=1.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
