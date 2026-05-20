"""Supertonic 3 vector estimator (64 M params) — flow-matching denoiser, MLX port.

Pipeline (operating in channels-last NTC layout):

    noisy_latent  [B, 144, T_lat]  (channels first from ONNX I/O)
      → transpose                  [B, T_lat, 144]
      → proj_in (Linear 144→512)   [B, T_lat, 512]
      → 24 main_blocks (4 cycles × 6 sub-types):
            cycle = [stack4, time_film, cn1, text_attn, cn1, style_attn]
      → last_convnext (4 ConvNeXt) [B, T_lat, 512]
      → proj_out (Linear 512→144)  [B, T_lat, 144]
      → transpose                  [B, 144, T_lat]
      → Euler step:   denoised = noisy + velocity * (1 / total_step)
      → output                     [B, 144, T_lat]

Submodule naming matches the s3 ONNX initializer keys exactly, so loading
the safetensors produced by ``weights.convert_onnx_to_mlx`` requires no
remapping.

The forward path is faithful to ONNX semantics in fp32; ``mx.compile``,
quantisation, and kernel fusion are layered on later in T.3.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from supertonic_3_mlx._config import (
    DIM, LATENT_CH, CONVNEXT_HIDDEN, CONVNEXT_K, STACK4_DILATIONS,
    NUM_MAIN_BLOCKS, BLOCKS_PER_CYCLE, BLOCK_CYCLE,
    TEXT_DIM, TEXT_HEADS, TEXT_HEAD_DIM, ROTARY_BASE, ROTARY_SCALE,
    STYLE_DIM, STYLE_LEN, STYLE_HEADS, STYLE_HEAD_DIM,
    TIME_EMB_DIM, TIME_MLP_HIDDEN,
    EPS_LN,
)
from supertonic_3_mlx._nn_wrappers import (
    WrappedNorm, WrappedLinear, ProjConv1x1,
)


def _pad_sym_edge(x: mx.array, pad: int) -> mx.array:
    """Symmetric replicate-edge pad on the time axis (axis=1 for [B, T, C])."""
    if pad == 0:
        return x
    left = mx.broadcast_to(x[:, :1, :], (x.shape[0], pad, x.shape[2]))
    right = mx.broadcast_to(x[:, -1:, :], (x.shape[0], pad, x.shape[2]))
    return mx.concatenate([left, x, right], axis=1)


def _gelu_exact(x: mx.array) -> mx.array:
    """Exact (non-tanh) GELU: x * 0.5 * (1 + erf(x / sqrt(2)))."""
    return x * 0.5 * (1.0 + mx.erf(x * (2 ** -0.5)))


def _mish(x: mx.array) -> mx.array:
    """Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))."""
    return x * mx.tanh(mx.logaddexp(x, mx.array(0.0, dtype=x.dtype)))


def _load_shared_style_key() -> mx.array:
    """Best-effort load of the fixed conditional style-attention key bank.

    The upstream vector_estimator ONNX graph bakes this tensor in as the
    anonymous initializer ``/vector_estimator/Expand_output_0``. It is the same
    tensor as text_encoder ``tts.ttl.style_encoder.style_token_layer.style_key``.
    """
    candidates: list[Path] = []
    for env_name in ("SUPERTONIC3_STYLE_KEY_ONNX", "SUPERTONIC3_TEXT_ENCODER_WEIGHTS"):
        if value := os.environ.get(env_name):
            candidates.append(Path(value))
    candidates.extend(
        [
            Path("/tmp/supertonic3/model/onnx/vector_estimator.onnx"),
            Path("/tmp/supertonic3/model/onnx/text_encoder.onnx"),
            Path.cwd() / "weights" / "text_encoder.safetensors",
            Path.cwd() / "sub-projects/supertonic3-mlx/hf_release/weights/text_encoder.safetensors",
        ]
    )

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".onnx":
                import onnx
                from onnx import numpy_helper

                model = onnx.load(str(path))
                names = {
                    "/vector_estimator/Expand_output_0",
                    "tts.ttl.style_encoder.style_token_layer.style_key",
                }
                for init in model.graph.initializer:
                    if init.name in names:
                        arr = numpy_helper.to_array(init)
                        if arr.shape == (1, STYLE_LEN, STYLE_DIM):
                            return mx.array(arr.astype("float32", copy=False))
            elif path.suffix == ".safetensors":
                from safetensors import safe_open

                with safe_open(str(path), framework="np") as f:
                    key = "tts.ttl.style_encoder.style_token_layer.style_key"
                    if key in f.keys():
                        arr = f.get_tensor(key)
                        if arr.shape == (1, STYLE_LEN, STYLE_DIM):
                            return mx.array(arr.astype("float32", copy=False))
        except Exception:
            continue

    return mx.zeros((1, STYLE_LEN, STYLE_DIM))


# ──────────────────────────────────────────────────────────────────
# ConvNeXt building blocks
# ──────────────────────────────────────────────────────────────────


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block matching s3 keys: ``dwconv``, ``norm.norm``, ``pwconv1/2``, ``gamma``."""

    def __init__(
        self,
        dim: int = DIM,
        hidden: int = CONVNEXT_HIDDEN,
        kernel: int = CONVNEXT_K,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dilation = dilation
        self.pad = dilation * (kernel - 1) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel, padding=0, dilation=dilation,
            groups=dim, bias=True,
        )
        self.norm = WrappedNorm(dim, eps=EPS_LN)
        self.pwconv1 = nn.Linear(dim, hidden, bias=True)
        self.pwconv2 = nn.Linear(hidden, dim, bias=True)
        # Stored as shape (1, dim, 1) in the ONNX checkpoint — see weights.py for
        # the load-time reshape that flattens it to (dim,) for broadcasting in NTC.
        self.gamma = mx.zeros((dim,))

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # x: (B, T, C)
        residual = x
        y = _pad_sym_edge(x, self.pad)
        y = self.dwconv(y)           # (B, T, C)
        y = self.norm(y)             # LayerNorm last-dim
        y = self.pwconv1(y)          # (B, T, hidden)
        y = _gelu_exact(y)
        y = self.pwconv2(y)          # (B, T, C)
        y = y * self.gamma           # broadcast over (B, T, .)
        out = residual + y
        if mask is not None:
            out = out * mask
        return out


class ConvNeXtStack(nn.Module):
    """List of ConvNeXt blocks. Loaded as ``convnext.[0..N-1].X``."""

    def __init__(self, dilations: tuple, dim: int = DIM, hidden: int = CONVNEXT_HIDDEN) -> None:
        super().__init__()
        self.convnext = [ConvNeXtBlock(dim, hidden, CONVNEXT_K, d) for d in dilations]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for b in self.convnext:
            x = b(x, mask)
        return x


# ──────────────────────────────────────────────────────────────────
# 6 block types per cycle
# ──────────────────────────────────────────────────────────────────


class Stack4Block(nn.Module):
    """Cycle position 0 — 4 ConvNeXt with dilations [1, 2, 4, 8].

    Loaded keys: ``convnext.[0..3].{dwconv,norm.norm,pwconv1,pwconv2,gamma}``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.convnext = [ConvNeXtBlock(DIM, CONVNEXT_HIDDEN, CONVNEXT_K, d) for d in STACK4_DILATIONS]

    def __call__(self, x: mx.array, mask: mx.array | None, **_) -> mx.array:
        for b in self.convnext:
            x = b(x, mask)
        return x


class TimeFiLMBlock(nn.Module):
    """Cycle position 1 — additive time conditioning: ``x + linear(t_emb)``.

    Loaded keys: ``linear.linear.{weight,bias}``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = WrappedLinear(TIME_EMB_DIM, DIM, bias=True)

    def __call__(self, x: mx.array, mask: mx.array | None, t_emb: mx.array, **_) -> mx.array:
        # t_emb: (B, TIME_EMB_DIM) → broadcast across T
        bias = self.linear(t_emb)[:, None, :]     # (B, 1, DIM)
        y = x + bias
        if mask is not None:
            y = y * mask
        return y


class ConvNeXt1Block(nn.Module):
    """Cycle positions 2 and 4 — a single ConvNeXt block.

    Loaded keys: ``convnext.0.{dwconv,norm.norm,pwconv1,pwconv2,gamma}``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.convnext = [ConvNeXtBlock(DIM, CONVNEXT_HIDDEN, CONVNEXT_K, 1)]

    def __call__(self, x: mx.array, mask: mx.array | None, **_) -> mx.array:
        return self.convnext[0](x, mask)


def _build_rope_freqs(head_dim: int, base: int, scale: int, max_len: int = 1024) -> mx.array:
    """Pre-compute RoPE cos/sin table — (max_len, head_dim/2, 2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (mx.arange(half, dtype=mx.float32) / half))
    pos = mx.arange(max_len, dtype=mx.float32) * scale
    angles = pos[:, None] * inv_freq[None, :]   # (max_len, half)
    return mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)  # (max_len, half, 2)


def _apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply RoPE rotation. ``x`` shape (B, H, T, head_dim); ``freqs`` (T, half, 2)."""
    half = x.shape[-1] // 2
    x_even, x_odd = x[..., :half], x[..., half:]
    cos = freqs[..., 0]    # (T, half)
    sin = freqs[..., 1]
    rot_even = x_even * cos[None, None, :, :] - x_odd * sin[None, None, :, :]
    rot_odd = x_even * sin[None, None, :, :] + x_odd * cos[None, None, :, :]
    return mx.concatenate([rot_even, rot_odd], axis=-1)


class TextCrossAttnBlock(nn.Module):
    """Cycle position 3 — text cross-attention with RoPE on Q and K.

    Loaded keys:
        ``attn.W_query.linear.{weight,bias}``
        ``attn.W_key.linear.{weight,bias}``
        ``attn.W_value.linear.{weight,bias}``
        ``attn.out_fc.linear.{weight,bias}``
        ``attn.theta``       — frozen RoPE inv-freq table (1, 1, half)
        ``attn.increments``  — frozen position table (1, 1000, 1) — 0..999
        ``norm.norm.{weight,bias}``
    """

    def __init__(self) -> None:
        super().__init__()
        self.attn = _AttnInner(DIM, TEXT_DIM, TEXT_HEADS, TEXT_HEAD_DIM)
        self.norm = WrappedNorm(DIM, eps=EPS_LN)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None,
        *,
        text_emb: mx.array | None = None,
        text_mask: mx.array | None = None,
        latent_seq_len: mx.array | None = None,
        text_seq_len: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
        **_,
    ) -> mx.array:
        # x: (B, T_lat, DIM); text_emb: (B, T_text, TEXT_DIM) — unused when kv_cache supplied.
        residual = x * mask if mask is not None else x
        h = self.attn(
            residual, text_emb, text_mask=text_mask,
            latent_seq_len=latent_seq_len, text_seq_len=text_seq_len,
            kv_cache=kv_cache,
        )
        if mask is not None:
            h = h * mask
        out = self.norm(residual + h)
        if mask is not None:
            out = out * mask
        return out


class _AttnInner(nn.Module):
    """Multi-head cross-attention with RoPE applied to query and key.

    Holds parameters under ``W_query``, ``W_key``, ``W_value``, ``out_fc`` —
    each is a :class:`WrappedLinear` so its weight is keyed
    ``…W_query.linear.weight`` to match the ONNX checkpoint.

    ``theta`` and ``increments`` come from the ONNX graph as frozen tensors
    (precomputed RoPE table). We rebuild the equivalent table from the
    Supertonic-3 config so the module is self-contained.
    """

    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # ONNX divides attention logits by 16.0 (= sqrt(TEXT_DIM)), not sqrt(head_dim).
        self.scale = ctx_dim ** -0.5

        kv_dim = num_heads * head_dim   # = DIM = 512
        self.W_query = WrappedLinear(in_dim, kv_dim, bias=True)
        self.W_key   = WrappedLinear(ctx_dim, kv_dim, bias=True)
        self.W_value = WrappedLinear(ctx_dim, kv_dim, bias=True)
        self.out_fc  = WrappedLinear(kv_dim, in_dim, bias=True)

        # Frozen RoPE tables — overwritten by checkpoint at load time.
        # ONNX layout:
        #   ``increments`` (1, 1000, 1) holds positions 0..999 (no scale baked in)
        #   ``theta``      (1, 1, half) holds rotary_scale × base^(-i/half)
        # Angle formula: ``angle = (pos / actual_seq_len) × theta``.
        # The division by the actual seq length is critical — it normalises
        # absolute positions into [0, 1] so audio and text are RoPE-aligned
        # regardless of their respective lengths.
        max_len = 1000
        half = head_dim // 2
        idx = mx.arange(half, dtype=mx.float32)
        self.theta = (ROTARY_SCALE * mx.exp(-math.log(ROTARY_BASE) * idx / half))[None, None, :]
        positions = mx.arange(max_len, dtype=mx.int64)
        self.increments = positions[None, :, None]   # (1, max_len, 1)

    def _rope(self, x: mx.array, seq_len: mx.array | int | None = None) -> mx.array:
        """Apply RoPE rotation. ``seq_len`` is the effective (unmasked) length.

        Args:
            x: (B, H, T, head_dim)
            seq_len: scalar or (B,) — actual sequence length for position normalisation.
                If None, defaults to T (no normalisation).
        """
        T = x.shape[-2]
        positions = self.increments[:, :T, :]            # (1, T, 1)
        if seq_len is None:
            seq_len = float(T)
        if isinstance(seq_len, (int, float)):
            divisor = float(seq_len)
        else:
            divisor = seq_len.astype(mx.float32).reshape(-1, 1, 1)
        norm_pos = positions / divisor                   # broadcasts to (B, T, 1) if divisor is (B,1,1)
        angles = norm_pos * self.theta                   # (B, T, half) or (1, T, half)
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        half = self.head_dim // 2
        # Broadcast (?, T, half) → (?, 1, T, half) for head dim
        cos_b = cos[..., None, :, :] if cos.ndim == 3 else cos[None, None, :, :]
        sin_b = sin[..., None, :, :] if sin.ndim == 3 else sin[None, None, :, :]
        # Make sure broadcasts properly
        if cos_b.shape[0] == 1 and x.shape[0] > 1:
            cos_b = mx.broadcast_to(cos_b, (x.shape[0], 1, T, half))
            sin_b = mx.broadcast_to(sin_b, (x.shape[0], 1, T, half))
        # Reshape if needed
        cos_b = cos_b.reshape(-1, 1, T, half)
        sin_b = sin_b.reshape(-1, 1, T, half)
        x_first, x_second = x[..., :half], x[..., half:]
        rot_first = x_first * cos_b - x_second * sin_b
        rot_second = x_first * sin_b + x_second * cos_b
        return mx.concatenate([rot_first, rot_second], axis=-1)

    def project_kv(
        self,
        text_emb: mx.array,
        text_seq_len: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Project text_emb → (K_rope, V) once. Both are constant across the
        Euler steps in a TTS inference call (T.5.3 cache target)."""
        B, T_text, _ = text_emb.shape
        H, D = self.num_heads, self.head_dim
        k = self.W_key(text_emb).reshape(B, T_text, H, D).transpose(0, 2, 1, 3)
        v = self.W_value(text_emb).reshape(B, T_text, H, D).transpose(0, 2, 1, 3)
        k = self._rope(k, seq_len=text_seq_len if text_seq_len is not None else T_text)
        return k, v

    def __call__(
        self,
        x: mx.array,
        text_emb: mx.array | None = None,
        text_mask: mx.array | None = None,
        latent_seq_len: mx.array | None = None,
        text_seq_len: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        B, T_lat, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.W_query(x).reshape(B, T_lat, H, D).transpose(0, 2, 1, 3)   # (B, H, T_lat, D)
        if kv_cache is not None:
            k, v = kv_cache
        else:
            k, v = self.project_kv(text_emb, text_seq_len=text_seq_len)

        # RoPE normalises positions by the effective (unmasked) sequence length.
        q = self._rope(q, seq_len=latent_seq_len if latent_seq_len is not None else T_lat)

        # Attention
        logits = (q @ k.transpose(0, 1, 3, 2)) * self.scale     # (B, H, T_lat, T_text)
        if text_mask is not None:
            neg_inf = mx.array(-1e4, dtype=logits.dtype)
            logits = mx.where(text_mask[:, :, None, :].astype(mx.bool_), logits, neg_inf)
        attn = mx.softmax(logits, axis=-1)
        out = attn @ v                                          # (B, H, T_lat, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T_lat, H * D)
        return self.out_fc(out)


class StyleCrossAttnBlock(nn.Module):
    """Cycle position 5 — style cross-attention to 50 learned style tokens.

    Loaded keys:
        ``attention.W_query.linear.{weight,bias}``
        ``attention.W_key.linear.{weight,bias}``
        ``attention.W_value.linear.{weight,bias}``
        ``attention.out_fc.linear.{weight,bias}``
        ``norm.norm.{weight,bias}``
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention = _StyleAttnInner(DIM, STYLE_DIM, STYLE_HEADS, STYLE_HEAD_DIM)
        self.norm = WrappedNorm(DIM, eps=EPS_LN)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None,
        *,
        style_k: mx.array | None = None,
        style_v: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
        **_,
    ) -> mx.array:
        # style_v defaults to style_k (same tensor for cond path); CFG path supplies
        # different style_v to model the uncond branch.
        if style_v is None and style_k is not None:
            style_v = style_k
        residual = x * mask if mask is not None else x
        h = self.attention(residual, style_k, style_v, kv_cache=kv_cache)
        if mask is not None:
            h = h * mask
        out = self.norm(residual + h)
        if mask is not None:
            out = out * mask
        return out


class _StyleAttnInner(nn.Module):
    def __init__(self, in_dim: int, ctx_dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # ONNX divides attention logits by 16.0 (= sqrt(STYLE_DIM)), not sqrt(head_dim).
        self.scale = ctx_dim ** -0.5
        kv_dim = num_heads * head_dim    # 2 * 128 = 256
        # Q is on DIM (audio), K/V on ctx_dim (style 256)
        self.W_query = WrappedLinear(in_dim, kv_dim, bias=True)
        self.W_key   = WrappedLinear(ctx_dim, kv_dim, bias=True)
        self.W_value = WrappedLinear(ctx_dim, kv_dim, bias=True)
        self.out_fc  = WrappedLinear(kv_dim, in_dim, bias=True)

    def project_kv(
        self, style_k: mx.array, style_v: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Project (style_k, style_v) → (K, V) once. T.5.3 cache target."""
        B, T_style = style_k.shape[0], style_k.shape[1]
        H, D = self.num_heads, self.head_dim
        # Note: ONNX graph applies tanh to the K projection (``attention/tanh/Tanh``
        # node) — the style key bank is bounded into [-1, 1] before softmax dot
        # product, which acts as a soft attention temperature regulariser.
        k = mx.tanh(self.W_key(style_k)).reshape(B, T_style, H, D).transpose(0, 2, 1, 3)
        v = self.W_value(style_v).reshape(B, style_v.shape[1], H, D).transpose(0, 2, 1, 3)
        return k, v

    def __call__(
        self,
        x: mx.array,
        style_k: mx.array | None = None,
        style_v: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        # style_k and style_v can be the same tensor (cond) or distinct (uncond
        # branch in CFG, where K comes from style_key_special_token and V from
        # style_value_special_token).
        B, T_lat, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.W_query(x).reshape(B, T_lat, H, D).transpose(0, 2, 1, 3)
        if kv_cache is not None:
            k, v = kv_cache
        else:
            k, v = self.project_kv(style_k, style_v)
        logits = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(logits, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T_lat, H * D)
        return self.out_fc(out)


# ──────────────────────────────────────────────────────────────────
# Time encoder
# ──────────────────────────────────────────────────────────────────


class _MlpItem(nn.Module):
    """A single MLP layer wrapped to produce keys ``mlp.N.linear.{weight,bias}``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class TimeEncoder(nn.Module):
    """Sinusoidal time embedding + 2-layer MLP. Keys: ``mlp.0.linear``, ``mlp.2.linear``."""

    def __init__(self) -> None:
        super().__init__()
        # ONNX: mlp.0.linear (64→256), mlp.2.linear (256→64). Index 1 is activation.
        self.mlp = [
            _MlpItem(TIME_EMB_DIM, TIME_MLP_HIDDEN),    # mlp.0
            nn.Identity(),                              # mlp.1 (activation; no weights)
            _MlpItem(TIME_MLP_HIDDEN, TIME_EMB_DIM),    # mlp.2
        ]

    def __call__(self, t: mx.array) -> mx.array:
        # t: (B,) — produce sinusoidal embedding then run through MLP.
        # Activation is Mish (not SiLU) to match the ONNX graph
        # (Softplus → Tanh → Mul pattern == x * tanh(softplus(x))).
        emb = self._sinusoidal(t, TIME_EMB_DIM)
        h = self.mlp[0](emb)
        h = _mish(h)
        h = self.mlp[2](h)
        return h

    @staticmethod
    def _sinusoidal(t: mx.array, dim: int) -> mx.array:
        """Time embedding matching ``Supertonic-3`` ONNX exactly.

        ONNX path:  pos = t * 1000;  freqs[i] = 10000^(-i/(half-1));
                    concat[sin(pos*freqs), cos(pos*freqs)].
        """
        half = dim // 2
        denom = max(half - 1, 1)
        freqs = mx.exp(-math.log(10_000) * mx.arange(half, dtype=mx.float32) / denom)
        pos = t.astype(mx.float32)[:, None] * 1000.0
        angles = pos * freqs[None, :]
        return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1).astype(mx.float32)


# ──────────────────────────────────────────────────────────────────
# Top-level VectorEstimator
# ──────────────────────────────────────────────────────────────────


def _build_main_block(idx: int) -> nn.Module:
    """Instantiate the appropriate block class for cycle position ``idx % 6``."""
    pos = idx % BLOCKS_PER_CYCLE
    name = BLOCK_CYCLE[pos]
    if name == "stack4":
        return Stack4Block()
    if name == "time":
        return TimeFiLMBlock()
    if name == "cn1":
        return ConvNeXt1Block()
    if name == "text_attn":
        return TextCrossAttnBlock()
    if name == "style_attn":
        return StyleCrossAttnBlock()
    raise RuntimeError(f"unknown block type for index {idx}: {name}")


class _VectorField(nn.Module):
    """Inner module mirroring ONNX ``vector_estimator.tts.ttl.vector_field.*``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_in = ProjConv1x1(LATENT_CH, DIM, bias=False)
        self.main_blocks = [_build_main_block(i) for i in range(NUM_MAIN_BLOCKS)]
        self.last_convnext = ConvNeXtStack(dilations=(1, 1, 1, 1), dim=DIM, hidden=CONVNEXT_HIDDEN)
        self.proj_out = ProjConv1x1(DIM, LATENT_CH, bias=False)
        self.time_encoder = TimeEncoder()


class _UncondMasker(nn.Module):
    """Holds the style-key bank plus unconditional-token tensors used by CFG.

    Keys:
        ``style_key``                  (1, 50, 256)
        ``text_special_token``        (1, 256, 1)
        ``style_key_special_token``   (1, 50, 256)
        ``style_value_special_token`` (1, 50, 256)
    """

    def __init__(self) -> None:
        super().__init__()
        # Conditional style attention uses the fixed text-encoder style key bank
        # for K and the per-voice ``style_ttl`` for V. The vector_estimator ONNX
        # graph stores this as an anonymous initializer, so load it best-effort.
        self.style_key = _load_shared_style_key()
        # Initialised to zero; checkpoint provides real values.
        self.text_special_token = mx.zeros((1, TEXT_DIM, 1))
        self.style_key_special_token = mx.zeros((1, STYLE_LEN, STYLE_DIM))
        self.style_value_special_token = mx.zeros((1, STYLE_LEN, STYLE_DIM))


class VectorEstimator(nn.Module):
    """Top-level module — matches ONNX root names ``vector_field.*`` and ``uncond_masker.*``.

    Two inference paths:
    - :meth:`velocity`: single forward pass; predicts the velocity from one set
      of conditioning inputs. Conditional style attention uses the fixed
      style key bank for K and ``style_ttl`` for V; CFG uses special-token
      K/V for the unconditional path.
    - :meth:`__call__`: full ONNX-parity forward — applies CFG batch doubling
      (cond + uncond) internally and combines via
      ``final = noisy + (4*cond - 3*uncond) / total_step``.
    """

    # CFG guidance constants — baked into the ONNX graph as ``/Constant_3`` (=4.0)
    # and ``/Constant_4`` (=3.0). Equivalent to guidance_scale = 4 with the
    # standard formula ``v = uncond + g*(cond - uncond) = 4*cond - 3*uncond``.
    CFG_COND_SCALE: float = 4.0
    CFG_UNCOND_SCALE: float = 3.0

    def __init__(self) -> None:
        super().__init__()
        self.vector_field = _VectorField()
        self.uncond_masker = _UncondMasker()

    def _conditional_style_key(self, batch_size: int, dtype: mx.Dtype) -> mx.array:
        key = self.uncond_masker.style_key.astype(dtype)
        return mx.broadcast_to(key, (batch_size, STYLE_LEN, STYLE_DIM))

    def _style_k_for_precompute(self, style_k: mx.array, style_v: mx.array) -> mx.array:
        batch = style_k.shape[0]
        if batch % 2 == 0 and batch > 1:
            half = batch // 2
            uncond_key = mx.broadcast_to(
                self.uncond_masker.style_key_special_token.astype(style_k.dtype),
                (batch - half, STYLE_LEN, STYLE_DIM),
            )
            try:
                mx.eval(uncond_key)
                looks_cfg = bool(mx.all(mx.abs(style_k[half:] - uncond_key) < 1e-5).item())
            except Exception:
                looks_cfg = False
            if looks_cfg:
                cond_key = self._conditional_style_key(half, style_k.dtype)
                return mx.concatenate([cond_key, style_k[half:]], axis=0)
        return self._conditional_style_key(batch, style_k.dtype)

    # ── inference API ─────────────────────────────────────────────
    def velocity(
        self,
        noisy_latent: mx.array,      # (B, 144, T_lat)
        text_emb: mx.array,          # (B, 256, T_text)
        style_k: mx.array,           # (B, 50, 256) — K side of style attention
        style_v: mx.array,           # (B, 50, 256) — V side of style attention
        latent_mask: mx.array,       # (B, 1, T_lat)
        text_mask: mx.array,         # (B, 1, T_text)
        t_norm: mx.array,            # (B,) timestep in [0, 1]
    ) -> mx.array:
        """Predict velocity (B, 144, T_lat) without applying CFG or Euler step."""
        x = noisy_latent.transpose(0, 2, 1)                  # (B, T_lat, 144)
        text = text_emb.transpose(0, 2, 1)                   # (B, T_text, 256)
        lat_mask_ntc = latent_mask.transpose(0, 2, 1)        # (B, T_lat, 1)

        x = self.vector_field.proj_in(x)                     # (B, T_lat, 512)
        t_emb = self.vector_field.time_encoder(t_norm)       # (B, TIME_EMB_DIM)

        # Effective (unmasked) sequence lengths for RoPE normalisation —
        # ONNX uses ``ReduceSum(mask)`` for this so that audio and text are
        # rope-aligned regardless of padding.
        latent_seq_len = mx.sum(latent_mask, axis=(1, 2))    # (B,)
        text_seq_len = mx.sum(text_mask, axis=(1, 2))        # (B,)

        for blk in self.vector_field.main_blocks:
            x = blk(
                x,
                lat_mask_ntc,
                t_emb=t_emb,
                text_emb=text,
                text_mask=text_mask,
                style_k=style_k,
                style_v=style_v,
                latent_seq_len=latent_seq_len,
                text_seq_len=text_seq_len,
            )

        x = self.vector_field.last_convnext(x, lat_mask_ntc)
        v_ntc = self.vector_field.proj_out(x)                # (B, T_lat, 144)
        return v_ntc.transpose(0, 2, 1)                      # (B, 144, T_lat)

    # ── T.5.3 — pre-projected K/V path ────────────────────────────
    def precompute_cross_kv(
        self,
        text_emb: mx.array,          # (B, 256, T_text) channels-first
        style_k: mx.array,           # (B, 50, 256)
        style_v: mx.array,           # (B, 50, 256)
        text_mask: mx.array,         # (B, 1, T_text)
    ) -> tuple[list[tuple[mx.array, mx.array]], list[tuple[mx.array, mx.array]]]:
        """Project K/V for every text_attn and style_attn block exactly once.

        Returns ``(text_kv_list, style_kv_list)`` — both ordered to align with
        the corresponding blocks encountered when iterating ``main_blocks``.
        These tensors are invariant across the 5 Euler steps of one TTS
        call; pre-projecting them once and feeding the result into
        :meth:`velocity_cached` cuts ~ 4 × 2 × 5 = 40 redundant matmuls.
        """
        style_k = self._style_k_for_precompute(style_k, style_v)
        text_seq_len = mx.sum(text_mask, axis=(1, 2))
        text_ntc = text_emb.transpose(0, 2, 1)               # (B, T_text, 256)

        text_kv: list[tuple[mx.array, mx.array]] = []
        style_kv: list[tuple[mx.array, mx.array]] = []
        for blk in self.vector_field.main_blocks:
            if isinstance(blk, TextCrossAttnBlock):
                text_kv.append(blk.attn.project_kv(text_ntc, text_seq_len=text_seq_len))
            elif isinstance(blk, StyleCrossAttnBlock):
                style_kv.append(blk.attention.project_kv(style_k, style_v))
        return text_kv, style_kv

    def velocity_cached(
        self,
        noisy_latent: mx.array,
        latent_mask: mx.array,
        text_mask: mx.array,
        t_norm: mx.array,
        text_kv: list[tuple[mx.array, mx.array]],
        style_kv: list[tuple[mx.array, mx.array]],
    ) -> mx.array:
        """Same as :meth:`velocity` but reads K/V from pre-projected caches.

        ``text_kv`` and ``style_kv`` must come from :meth:`precompute_cross_kv`
        applied to the same (batched) conditioning tensors that will be
        active for this call.
        """
        x = noisy_latent.transpose(0, 2, 1)
        lat_mask_ntc = latent_mask.transpose(0, 2, 1)

        x = self.vector_field.proj_in(x)
        t_emb = self.vector_field.time_encoder(t_norm)
        latent_seq_len = mx.sum(latent_mask, axis=(1, 2))

        ti = 0
        si = 0
        for blk in self.vector_field.main_blocks:
            if isinstance(blk, TextCrossAttnBlock):
                x = blk(
                    x, lat_mask_ntc,
                    text_mask=text_mask,
                    latent_seq_len=latent_seq_len,
                    kv_cache=text_kv[ti],
                )
                ti += 1
            elif isinstance(blk, StyleCrossAttnBlock):
                x = blk(x, lat_mask_ntc, kv_cache=style_kv[si])
                si += 1
            else:
                x = blk(x, lat_mask_ntc, t_emb=t_emb)

        x = self.vector_field.last_convnext(x, lat_mask_ntc)
        v_ntc = self.vector_field.proj_out(x)
        return v_ntc.transpose(0, 2, 1)

    def __call__(
        self,
        noisy_latent: mx.array,      # (B, 144, T_lat) channels-first per ONNX I/O
        text_emb: mx.array,          # (B, 256, T_text) channels-first
        style_ttl: mx.array,         # (B, 50, 256) — V side for cond style attention
        latent_mask: mx.array,       # (B, 1, T_lat)
        text_mask: mx.array,         # (B, 1, T_text)
        current_step: mx.array,      # (B,)
        total_step: mx.array,        # (B,)
        cfg: bool = True,
    ) -> mx.array:
        """Run one Euler step with CFG (matches ONNX semantics).

        With ``cfg=True`` (default) the model runs both conditional and
        unconditional paths in a single batched forward and combines via
        ``final = noisy + (4*cond_v - 3*uncond_v) / total_step``.

        With ``cfg=False`` only the conditional path runs — half the work, but
        produces a different (lower-quality) output. Useful for speed bench /
        sanity tests.
        """
        B = noisy_latent.shape[0]
        t_norm = current_step.astype(mx.float32) / total_step.astype(mx.float32)

        if not cfg:
            style_key = self._conditional_style_key(B, style_ttl.dtype)
            v = self.velocity(
                noisy_latent, text_emb, style_key, style_ttl,
                latent_mask, text_mask, t_norm,
            )
            return noisy_latent + v / total_step.reshape(-1, 1, 1).astype(noisy_latent.dtype)

        # CFG branch — build (2B, ...) inputs by concatenating cond + uncond.
        # uncond text_emb = text_special_token broadcast to (B, 256, T_text).
        # cond style_k = fixed style_key broadcast; uncond style_k/style_v are
        # the learned special tokens broadcast to the batch.
        text_uncond = mx.broadcast_to(
            self.uncond_masker.text_special_token, (B, TEXT_DIM, text_emb.shape[2])
        )
        style_k_uncond = mx.broadcast_to(
            self.uncond_masker.style_key_special_token, (B, STYLE_LEN, STYLE_DIM)
        )
        style_v_uncond = mx.broadcast_to(
            self.uncond_masker.style_value_special_token, (B, STYLE_LEN, STYLE_DIM)
        )
        style_key_cond = self._conditional_style_key(B, style_ttl.dtype)

        noisy_2 = mx.concatenate([noisy_latent, noisy_latent], axis=0)
        text_2 = mx.concatenate([text_emb, text_uncond], axis=0)
        style_k_2 = mx.concatenate([style_key_cond, style_k_uncond], axis=0)
        style_v_2 = mx.concatenate([style_ttl, style_v_uncond], axis=0)
        lm_2 = mx.concatenate([latent_mask, latent_mask], axis=0)
        tm_2 = mx.concatenate([text_mask, text_mask], axis=0)
        t_norm_2 = mx.concatenate([t_norm, t_norm], axis=0)

        v_2 = self.velocity(
            noisy_2, text_2, style_k_2, style_v_2, lm_2, tm_2, t_norm_2,
        )                                                   # (2B, 144, T_lat)
        cond_v = v_2[:B]
        uncond_v = v_2[B:2 * B]
        combined_v = self.CFG_COND_SCALE * cond_v - self.CFG_UNCOND_SCALE * uncond_v
        return noisy_latent + combined_v / total_step.reshape(-1, 1, 1).astype(noisy_latent.dtype)


__all__ = [
    "ConvNeXtBlock", "ConvNeXtStack",
    "Stack4Block", "TimeFiLMBlock", "ConvNeXt1Block",
    "TextCrossAttnBlock", "StyleCrossAttnBlock",
    "TimeEncoder", "VectorEstimator",
]
