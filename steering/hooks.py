"""Activation-steering hook for residual stream injection.

Single source of truth used by 02_layer_sweep, 03_calibrate, and run.py.
Coefficients are in fraction-of-residual-norm units (paper convention) — a
coefficient of 0.5 means "add a vector whose magnitude is half the typical
residual stream norm at this layer." The layer norm is computed once by
`steering/norm.py` and stored in config.yaml.

Each forward pass through the steered layer adds:

    sum(coef_i * layer_norm * vec_i)   for each (vec_i, coef_i) installed

The vectors are expected to already be L2-normalized at extraction time
(unit vectors), so this gives the desired magnitude in residual-norm units.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


VectorWithCoef = tuple[torch.Tensor, float]


class MultiVectorSteeringHook:
    """Add `sum(coef * layer_norm * vec)` to the residual stream output of a
    Llama-style transformer layer on every forward pass while installed.

    Usage:
        hook = MultiVectorSteeringHook(
            layer_module=model.model.layers[21],
            vectors_with_coefs=[(happy_vec, 0.5), (sad_vec, 0.0)],
            layer_norm=14.7,
        )
        with hook:
            output = model.generate(...)
    """

    def __init__(
        self,
        layer_module: torch.nn.Module,
        vectors_with_coefs: list[VectorWithCoef],
        layer_norm: float,
    ) -> None:
        self.layer_module = layer_module
        self.layer_norm = float(layer_norm)
        self._handle = None

        if not vectors_with_coefs:
            self._delta = None
        else:
            device = vectors_with_coefs[0][0].device
            dtype = vectors_with_coefs[0][0].dtype
            delta = torch.zeros_like(vectors_with_coefs[0][0])
            for vec, coef in vectors_with_coefs:
                if vec.device != device:
                    raise ValueError("All steering vectors must be on the same device.")
                if vec.dim() != 1:
                    raise ValueError(f"Steering vector must be 1-D, got shape {tuple(vec.shape)}.")
                delta = delta + (float(coef) * self.layer_norm) * vec.to(dtype)
            # Reshape to broadcast over [batch, seq, hidden].
            self._delta = delta.view(1, 1, -1)

    def _hook(self, _module, _inputs, output):
        if self._delta is None:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            return (hidden + self._delta.to(hidden.dtype),) + output[1:]
        return output + self._delta.to(output.dtype)

    def __enter__(self) -> "MultiVectorSteeringHook":
        self._handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def steer(
    model,
    layer: int,
    vectors_with_coefs: list[VectorWithCoef],
    layer_norm: float,
) -> Iterator[MultiVectorSteeringHook]:
    """Convenience wrapper: `with steer(model, 21, [...], norm): generate(...)`."""
    hook = MultiVectorSteeringHook(
        layer_module=model.model.layers[layer],
        vectors_with_coefs=vectors_with_coefs,
        layer_norm=layer_norm,
    )
    with hook as h:
        yield h
