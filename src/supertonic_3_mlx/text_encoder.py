"""Supertonic 3 text encoder MLX port.

Pipeline (operating in channels-last NTC after the initial conv):

    text_ids   [B, T_text]            int64 character IDs
      → char_embedder (Embedding 8322→256)        [B, T_text, 256]
      → 6× ConvNeXt(dim=256, hidden=1024, k=5, dilations [1,1,2,2,4,4])
      → 4× attn_encoder block:
            RelPosSelfAttn (conv_q/k/v/o, 4 heads × 64) + norm_layers_1
            FFN (conv_1: 256→1024, conv_2: 1024→256) + norm_layers_2
      → speech_prompted_text_encoder:
            cross-attn1: text (Q) × style_ttl (K, V) → text features
            cross-attn2: text (Q) × style_ttl (K, V) → text features
            norm
      → output text_emb [B, 256, T_text]  (channels-first to match vector_estimator)

Inputs:
    text_ids:  (B, T_text) int — character indices
    style_ttl: (B, 50, 256) float — style token bank
    text_mask: (B, 1, T_text) float — 1.0 where valid, 0.0 where padded

Submodule naming matches the ONNX initializer keys exactly so that
``model.load_weights(...)`` succeeds with no remapping.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from supertonic_3_mlx._config import EPS_LN
from supertonic_3_mlx._nn_wrappers import WrappedNorm, WrappedLinear
from supertonic_3_mlx.vector_estimator import (
    ConvNeXtBlock, _pad_sym_edge, _gelu_exact,
)


# Vocab + dims (frozen by checkpoint)
VOCAB_SIZE = 8322
TE_DIM = 256
TE_CONVNEXT_HIDDEN = 1024
TE_CONVNEXT_K = 5
TE_CONVNEXT_NUM_LAYERS = 6
TE_CONVNEXT_DILATIONS = (1, 1, 2, 2, 4, 4)

TE_ATTN_NUM_LAYERS = 4
TE_ATTN_HEADS = 4
TE_ATTN_HEAD_DIM = TE_DIM // TE_ATTN_HEADS   # 64
TE_FFN_HIDDEN = 1024


class TextConvNeXtBlock(nn.Module):
    """ConvNeXt for the text encoder (dim=256, hidden=1024).

    Shares the same architecture as ``vector_estimator.ConvNeXtBlock`` but is
    redefined here with text-encoder-specific defaults to keep the modules
    self-contained.
    """

    def __init__(self, dilation: int = 1) -> None:
        super().__init__()
        self.dim = TE_DIM
        self.dilation = dilation
        self.pad = dilation * (TE_CONVNEXT_K - 1) // 2
        self.dwconv = nn.Conv1d(
            TE_DIM, TE_DIM, kernel_size=TE_CONVNEXT_K, padding=0,
            dilation=dilation, groups=TE_DIM, bias=True,
        )
        self.norm = WrappedNorm(TE_DIM, eps=EPS_LN)
        self.pwconv1 = nn.Linear(TE_DIM, TE_CONVNEXT_HIDDEN, bias=True)
        self.pwconv2 = nn.Linear(TE_CONVNEXT_HIDDEN, TE_DIM, bias=True)
        self.gamma = mx.zeros((TE_DIM,))

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # x: (B, T_text, 256)
        residual = x
        y = _pad_sym_edge(x, self.pad)
        y = self.dwconv(y)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = _gelu_exact(y)
        y = self.pwconv2(y)
        y = y * self.gamma
        out = residual + y
        if mask is not None:
            out = out * mask
        return out


class TextConvNeXtStack(nn.Module):
    """6 stacked ConvNeXt blocks. Loaded as ``convnext.convnext.[0..5].X``."""

    def __init__(self) -> None:
        super().__init__()
        self.convnext = [TextConvNeXtBlock(d) for d in TE_CONVNEXT_DILATIONS]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for b in self.convnext:
            x = b(x, mask)
        return x


class _ConvLayer(nn.Module):
    """Conv1d k=1 expressed via the ONNX-style ``X.weight (out, in, 1) + X.bias``.

    The attn_encoder uses Conv1d k=1 instead of nn.Linear for its Q/K/V/O.
    This wrapper keeps the weight shape (out, in, 1) intact and runs as a
    Conv1d (the equivalent of a Linear when k=1).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = mx.zeros((out_dim, 1, in_dim))   # (C_out, K=1, C_in)
        self.bias = mx.zeros((out_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, in_dim) — channels-last
        # equivalent to nn.Conv1d(in_dim, out_dim, k=1) in NTC layout
        return mx.conv1d(x, self.weight, stride=1, padding=0) + self.bias


REL_POS_WINDOW = 4   # rel_pos table size = 2*4 + 1 = 9


def _rel_to_abs(x: mx.array) -> mx.array:
    """[B, h, L, 2L-1] → [B, h, L, L] via the VITS shifted-skew reshape."""
    B, h, L, _ = x.shape
    x = mx.concatenate([x, mx.zeros((B, h, L, 1), dtype=x.dtype)], axis=-1)
    x_flat = x.reshape(B, h, L * 2 * L)
    x_flat = mx.concatenate([x_flat, mx.zeros((B, h, L - 1), dtype=x.dtype)], axis=-1)
    x_final = x_flat.reshape(B, h, L + 1, 2 * L - 1)
    return x_final[:, :, :L, L - 1:]


def _abs_to_rel(x: mx.array) -> mx.array:
    """[B, h, L, L] → [B, h, L, 2L-1] (inverse of _rel_to_abs)."""
    B, h, L, _ = x.shape
    x = mx.concatenate([x, mx.zeros((B, h, L, L - 1), dtype=x.dtype)], axis=-1)
    x_flat = x.reshape(B, h, L * (2 * L - 1))
    x_flat = mx.concatenate([mx.zeros((B, h, L), dtype=x.dtype), x_flat], axis=-1)
    x_final = x_flat.reshape(B, h, L, 2 * L)
    return x_final[:, :, :, 1:]


def _slice_rel_emb(rel: mx.array, length: int, window: int) -> mx.array:
    """``rel`` (1, 2W+1, d) → (1, 2L-1, d) by zero-padding/slicing."""
    pad_l = max(length - (window + 1), 0)
    if pad_l > 0:
        zero = mx.zeros((1, pad_l, rel.shape[-1]), dtype=rel.dtype)
        padded = mx.concatenate([zero, rel, zero], axis=1)
    else:
        padded = rel
    start = max(window + 1 - length, 0)
    return padded[:, start: start + 2 * length - 1]


class RelPosSelfAttention(nn.Module):
    """VITS-style relative-position self-attention with window=4.

    Adds two contributions to vanilla MHA:
    - ``rel_logits = q @ rel_k.T`` then ``_rel_to_abs`` and added to attention logits
    - ``rel_attn = _abs_to_rel(softmax(logits))`` then ``@ rel_v`` and added to output

    Loaded keys (per layer):
        ``conv_q/k/v/o.weight`` (256, 256, 1) and ``.bias`` (256)
        ``emb_rel_k`` (1, 9, 64), ``emb_rel_v`` (1, 9, 64)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_q = _ConvLayer(TE_DIM, TE_DIM)
        self.conv_k = _ConvLayer(TE_DIM, TE_DIM)
        self.conv_v = _ConvLayer(TE_DIM, TE_DIM)
        self.conv_o = _ConvLayer(TE_DIM, TE_DIM)
        self.window = REL_POS_WINDOW
        self.emb_rel_k = mx.zeros((1, 2 * REL_POS_WINDOW + 1, TE_ATTN_HEAD_DIM))
        self.emb_rel_v = mx.zeros((1, 2 * REL_POS_WINDOW + 1, TE_ATTN_HEAD_DIM))

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, _ = x.shape
        H, D = TE_ATTN_HEADS, TE_ATTN_HEAD_DIM
        q = self.conv_q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.conv_k(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.conv_v(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        scale = D ** -0.5

        # Standard attention logits
        logits = (q @ k.transpose(0, 1, 3, 2)) * scale          # (B, H, T, T)

        # VITS relative-position contribution to logits
        rel_k = _slice_rel_emb(self.emb_rel_k, T, self.window)  # (1, 2T-1, D)
        rel_logits = q @ rel_k.transpose(0, 2, 1)[:, None, :, :]  # (B, H, T, 2T-1)
        rel_logits = _rel_to_abs(rel_logits * scale)            # (B, H, T, T)
        logits = logits + rel_logits

        if mask is not None:
            key_mask = mask[:, :, 0][:, None, None, :]
            neg_inf = mx.array(-1e4, dtype=logits.dtype)
            logits = mx.where(key_mask.astype(mx.bool_), logits, neg_inf)

        attn = mx.softmax(logits, axis=-1)                       # (B, H, T, T)
        out = attn @ v                                            # (B, H, T, D)

        # VITS rel-pos value contribution
        rel_v = _slice_rel_emb(self.emb_rel_v, T, self.window)   # (1, 2T-1, D)
        rel_weights = _abs_to_rel(attn)                          # (B, H, T, 2T-1)
        out = out + rel_weights @ rel_v[:, None, :, :]           # (B, H, T, D)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.conv_o(out)


class FFN(nn.Module):
    """FFN with Conv1d k=1 wrappers: conv_1 (256→1024) + ReLU + conv_2 (1024→256).

    Activation is ReLU (confirmed by ONNX graph node ``Relu`` in ``ffn_layers.N``),
    not GELU. The mask is applied before each Conv to match the ONNX semantics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = _ConvLayer(TE_DIM, TE_FFN_HIDDEN)
        self.conv_2 = _ConvLayer(TE_FFN_HIDDEN, TE_DIM)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        if mask is not None:
            x = x * mask
        y = self.conv_1(x)
        y = mx.maximum(y, mx.array(0.0, dtype=y.dtype))
        if mask is not None:
            y = y * mask
        y = self.conv_2(y)
        if mask is not None:
            y = y * mask
        return y


class AttnEncoder(nn.Module):
    """Stack of (RelPosSelfAttn + norm1) + (FFN + norm2) × 4."""

    def __init__(self) -> None:
        super().__init__()
        self.attn_layers = [RelPosSelfAttention() for _ in range(TE_ATTN_NUM_LAYERS)]
        self.norm_layers_1 = [WrappedNorm(TE_DIM, eps=EPS_LN) for _ in range(TE_ATTN_NUM_LAYERS)]
        self.ffn_layers = [FFN() for _ in range(TE_ATTN_NUM_LAYERS)]
        self.norm_layers_2 = [WrappedNorm(TE_DIM, eps=EPS_LN) for _ in range(TE_ATTN_NUM_LAYERS)]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for i in range(TE_ATTN_NUM_LAYERS):
            y = self.attn_layers[i](x, mask=mask)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, mask)
            x = self.norm_layers_2[i](x + y)
        return x


class _TextEmbedder(nn.Module):
    """char_embedder: VOCAB → TE_DIM. Loaded as ``char_embedder.weight (8322, 256)``."""

    def __init__(self) -> None:
        super().__init__()
        self.char_embedder = nn.Embedding(VOCAB_SIZE, TE_DIM)

    def __call__(self, text_ids: mx.array) -> mx.array:
        return self.char_embedder(text_ids)


class _InnerTextEncoder(nn.Module):
    """Pure text encoder before speech prompting. Loaded as ``text_encoder.X.Y``."""

    def __init__(self) -> None:
        super().__init__()
        self.text_embedder = _TextEmbedder()
        self.convnext = TextConvNeXtStack()
        self.attn_encoder = AttnEncoder()

    def __call__(self, text_ids: mx.array, mask: mx.array) -> mx.array:
        x = self.text_embedder(text_ids)        # (B, T, 256)
        if mask is not None:
            x = x * mask
        x = self.convnext(x, mask)
        x = self.attn_encoder(x, mask)
        return x


class _StyleEncoder(nn.Module):
    """Holds ``style_token_layer.style_key`` (1, 50, 256)."""

    def __init__(self) -> None:
        super().__init__()
        # Use a child module so the parameter path matches ``style_token_layer.style_key``
        class _StyleTokenLayer(nn.Module):
            def __init__(_):
                super().__init__()
                _.style_key = mx.zeros((1, 50, 256))
        self.style_token_layer = _StyleTokenLayer()


class _SpeechPromptedAttn(nn.Module):
    """Cross-attention from text (Q) to style_ttl (K, V). Single head, 256-d."""

    def __init__(self) -> None:
        super().__init__()
        self.W_query = WrappedLinear(TE_DIM, TE_DIM, bias=True)
        self.W_key = WrappedLinear(TE_DIM, TE_DIM, bias=True)
        self.W_value = WrappedLinear(TE_DIM, TE_DIM, bias=True)
        self.out_fc = WrappedLinear(TE_DIM, TE_DIM, bias=True)

    def __call__(self, x: mx.array, style: mx.array) -> mx.array:
        # x: (B, T_text, 256); style: (B, 50, 256)
        # Single-head cross attention.
        B, T, D = x.shape
        q = self.W_query(x)
        k = self.W_key(style)
        v = self.W_value(style)
        scale = D ** -0.5
        logits = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(logits, axis=-1)
        out = attn @ v
        return self.out_fc(out)


class _SpeechPromptedTextEncoder(nn.Module):
    """Two cross-attention layers modulating text features with style_ttl."""

    def __init__(self) -> None:
        super().__init__()
        self.attention1 = _SpeechPromptedAttn()
        self.attention2 = _SpeechPromptedAttn()
        self.norm = WrappedNorm(TE_DIM, eps=EPS_LN)

    def __call__(self, x: mx.array, style: mx.array) -> mx.array:
        x = x + self.attention1(x, style)
        x = x + self.attention2(x, style)
        return self.norm(x)


class _RootTextEncoder(nn.Module):
    """Top-level container matching ONNX ``tts.ttl.*`` namespace."""

    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = _InnerTextEncoder()
        self.style_encoder = _StyleEncoder()
        self.speech_prompted_text_encoder = _SpeechPromptedTextEncoder()


class _TtsContainer(nn.Module):
    """Outer container so weight keys ``tts.ttl.X.Y`` resolve."""

    def __init__(self) -> None:
        super().__init__()
        self.ttl = _RootTextEncoder()


class TextEncoder(nn.Module):
    """Top-level text encoder: ``text_ids + style_ttl + text_mask → text_emb (B, 256, T)``.

    Submodule naming matches the ONNX initializer keys after a single
    ``tts.ttl.`` prefix wrap (so weight keys look like
    ``tts.ttl.text_encoder.convnext.convnext.0.dwconv.weight``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.tts = _TtsContainer()

    def __call__(
        self,
        text_ids: mx.array,    # (B, T_text) int
        style_ttl: mx.array,   # (B, 50, 256)
        text_mask: mx.array,   # (B, 1, T_text)
    ) -> mx.array:
        mask_ntc = text_mask.transpose(0, 2, 1)         # (B, T_text, 1)
        x = self.tts.ttl.text_encoder(text_ids, mask_ntc)
        x = self.tts.ttl.speech_prompted_text_encoder(x, style_ttl)
        if mask_ntc is not None:
            x = x * mask_ntc
        # Return channels-first (B, 256, T_text) to match the vector_estimator input.
        return x.transpose(0, 2, 1)


__all__ = ["TextEncoder", "VOCAB_SIZE", "TE_DIM"]
