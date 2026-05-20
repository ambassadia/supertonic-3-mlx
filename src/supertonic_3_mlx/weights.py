"""ONNX → MLX safetensors conversion for Supertonic 3.

Two-stage extraction:
1. **Named initializers** (e.g. ``vector_estimator.tts.ttl.vector_field.main_blocks.0.convnext.0.dwconv.weight``)
   — straight name strip + optional shape transformation.
2. **Anonymous MatMul weights** (e.g. ``onnx::MatMul_3391``) — looked up via the
   MatMul node graph: each MatMul output path is the human-readable name of the
   weight (e.g. ``…/W_query/linear/MatMul_output_0``); we trace the second
   operand initializer and rebind it to the structured name + transpose to
   the MLX Linear layout ``(out, in)``.

Shape transformations:
- depthwise dwconv:   ONNX ``(C, 1, K)``  → MLX ``(C, K, 1)``
- pwconv1/2 k=1:      ONNX ``(out, in, 1)`` → MLX ``(out, in)``
- proj_in/out k=1:    ONNX ``(out, in, 1)`` → MLX ``(out, in)``
- MatMul Linear:      ONNX ``(in, out)``    → MLX ``(out, in)``
- gamma:              ONNX ``(1, dim, 1)``  → MLX ``(dim,)``
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx
import numpy as np


_ONNX_PREFIX = "vector_estimator.tts.ttl."

_DWCONV_SUFFIX = ".dwconv.weight"
_PWCONV_SUFFIXES = (".pwconv1.weight", ".pwconv2.weight")
_GAMMA_SUFFIX = ".gamma"


def _strip_prefix(name: str) -> str:
    if name.startswith(_ONNX_PREFIX):
        return name[len(_ONNX_PREFIX):]
    return name


def _is_named_weight(name: str) -> bool:
    """True if this is a structured weight (vs anonymous graph constant)."""
    if name.startswith(_ONNX_PREFIX):
        return True
    if name.startswith("uncond_masker."):
        return True
    return False


def _convert_named(clean_name: str, arr: np.ndarray) -> np.ndarray:
    """Apply shape transforms to a named initializer based on its key."""
    # Depthwise Conv1d weight: (C, 1, K) → (C, K, 1)
    if clean_name.endswith(_DWCONV_SUFFIX) and arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] != 1:
        arr = np.transpose(arr, (0, 2, 1))

    # Pointwise k=1 / proj net weight: (out, in, 1) → (out, in)
    if (any(clean_name.endswith(s) for s in _PWCONV_SUFFIXES) or clean_name.endswith(".net.weight")) \
            and arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)

    # gamma: (1, C, 1) → (C,)
    if clean_name.endswith(_GAMMA_SUFFIX) and arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 1:
        arr = arr.reshape(arr.shape[1])

    return arr


def _matmul_output_to_clean_name(matmul_output: str) -> str:
    """Map a MatMul node output path to the structured ``.weight`` key.

    Example::

        /vector_estimator/vector_field/main_blocks.3/attn/W_query/linear/MatMul_output_0
        → vector_field.main_blocks.3.attn.W_query.linear.weight
    """
    # Strip prefix slash and the trailing /MatMul_output_0
    path = matmul_output.lstrip("/")
    if path.endswith("/MatMul_output_0"):
        path = path[: -len("/MatMul_output_0")]
    # Drop leading "vector_estimator/" if present
    if path.startswith("vector_estimator/"):
        path = path[len("vector_estimator/"):]
    return path.replace("/", ".") + ".weight"


def convert_onnx_to_mlx(onnx_path: str | Path) -> Dict[str, mx.array]:
    """Load an ONNX model and return all weights as ``{clean_name: mx.array}``.

    Combines named initializers and MatMul-only weights into a single dict ready
    for ``model.load_weights(...)``.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))

    # Build initializer name → numpy array map (in-memory once)
    inits: Dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }

    out: Dict[str, mx.array] = {}

    # Stage 1: named initializers
    for name, arr in inits.items():
        if not _is_named_weight(name):
            continue
        clean = _strip_prefix(name)
        arr = _convert_named(clean, arr)
        out[clean] = mx.array(arr)

    # Stage 2: anonymous MatMul weights, recovered via the graph
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        if len(node.input) < 2:
            continue
        # The weight is conventionally the second operand
        weight_name = node.input[1]
        if weight_name not in inits:
            continue
        # Skip if it's already named structurally (shouldn't happen here)
        if _is_named_weight(weight_name):
            continue
        # Look up the structured name from the MatMul output path
        if len(node.output) < 1:
            continue
        clean = _matmul_output_to_clean_name(node.output[0])
        # ONNX MatMul stores W as (in, out); MLX Linear expects (out, in)
        arr = inits[weight_name]
        if arr.ndim == 2:
            arr = arr.T
        out[clean] = mx.array(arr)

    if not out:
        raise RuntimeError(f"no weights extracted from {onnx_path}")
    return out


def save_safetensors(
    onnx_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Tuple[int, ...]]:
    """Convert an ONNX file to MLX safetensors. Returns a {name: shape} map."""
    weights = convert_onnx_to_mlx(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_path), weights)
    return {k: tuple(v.shape) for k, v in weights.items()}


__all__ = ["convert_onnx_to_mlx", "save_safetensors"]
