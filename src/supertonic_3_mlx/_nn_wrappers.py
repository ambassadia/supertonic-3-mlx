"""Small wrapper modules to match Supertonic 3 ONNX submodule nesting.

The s3 checkpoint nests primitives one level deeper than typical MLX modules:
- ``norm.norm.weight``  — LayerNorm wrapped in a Norm container
- ``linear.linear.weight`` — Linear wrapped in a Linear container
- ``W_query.linear.weight`` — attention projection wrapped

Mirroring this nesting lets us load the safetensors with ``model.load_weights(...)``
without any key remapping at load time.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class WrappedNorm(nn.Module):
    """Container with a single nested LayerNorm — produces key ``X.norm.weight``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(x)


class WrappedLinear(nn.Module):
    """Container with a single nested Linear — produces keys ``X.linear.weight/bias``."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class ProjConv1x1(nn.Module):
    """Conv1d k=1 expressed as ``self.net = Linear`` (matches ``proj_in.net.weight``)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


__all__ = ["WrappedNorm", "WrappedLinear", "ProjConv1x1"]
