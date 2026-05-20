"""Supertonic 3 duration predictor — predicts total audio duration in seconds.

Pipeline (channels-last NTC throughout):

    text_ids   [B, T]           int64 character IDs
      → char_embed (Embedding 8322→64)        [B, T, 64]
      → prepend sentence_token (1, 64, 1)      [B, T+1, 64]
      → 6× ConvNeXt (dim=64, hidden=256, k=5, all dilations=1)
      → 2× RelPosSelfAttn (heads=2, head_dim=32, window=4) + norm + FFN + norm
      → proj_out (Conv1d k=1: 64→64) applied to slot 0 (sentence token)
      → concat with style_dp flattened (B, 8×16=128)              [B, 192]
      → Linear(192 → 128) → PReLU → Linear(128 → 1) → exp → duration [B]

Inputs:
    text_ids: (B, T) int — character indices
    style_dp: (B, 8, 16) — style summary tokens
    text_mask: (B, 1, T) — 1.0 valid, 0.0 padded
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from supertonic_3_mlx._config import EPS_LN
from supertonic_3_mlx._nn_wrappers import WrappedNorm
from supertonic_3_mlx.vector_estimator import _pad_sym_edge, _gelu_exact


DP_VOCAB = 8322
DP_DIM = 64
DP_CONVNEXT_HIDDEN = 256
DP_CONVNEXT_K = 5
DP_CONVNEXT_NUM_LAYERS = 6
DP_ATTN_NUM_LAYERS = 2
DP_ATTN_HEADS = 2
DP_ATTN_HEAD_DIM = DP_DIM // DP_ATTN_HEADS   # 32
DP_FFN_HIDDEN = 256
DP_REL_POS_WINDOW = 4
DP_N_STYLE = 8
DP_STYLE_DIM = 16
DP_MLP_IN = DP_DIM + DP_N_STYLE * DP_STYLE_DIM   # 64 + 128 = 192
DP_MLP_HIDDEN = 128


class _DPConvNeXtBlock(nn.Module):
    """ConvNeXt block (dim=64, hidden=256, dilation=1)."""

    def __init__(self) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
            DP_DIM, DP_DIM, kernel_size=DP_CONVNEXT_K, padding=0,
            dilation=1, groups=DP_DIM, bias=True,
        )
        self.norm = WrappedNorm(DP_DIM, eps=EPS_LN)
        self.pwconv1 = nn.Linear(DP_DIM, DP_CONVNEXT_HIDDEN, bias=True)
        self.pwconv2 = nn.Linear(DP_CONVNEXT_HIDDEN, DP_DIM, bias=True)
        self.gamma = mx.zeros((DP_DIM,))
        self.pad = (DP_CONVNEXT_K - 1) // 2

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
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


class _DPConvNeXtStack(nn.Module):
    """``convnext.[0..5]`` — 6 ConvNeXt blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.convnext = [_DPConvNeXtBlock() for _ in range(DP_CONVNEXT_NUM_LAYERS)]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for b in self.convnext:
            x = b(x, mask)
        return x


class _DPConvLayer(nn.Module):
    """Conv1d k=1 with weight (out, 1, in) — matches ONNX storage."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = mx.zeros((out_dim, 1, in_dim))
        self.bias = mx.zeros((out_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.conv1d(x, self.weight, stride=1, padding=0) + self.bias


def _dp_rel_to_abs(x: mx.array) -> mx.array:
    """(B, h, L, 2L-1) → (B, h, L, L) via VITS shifted-skew reshape."""
    B, h, L, _ = x.shape
    x = mx.concatenate([x, mx.zeros((B, h, L, 1), dtype=x.dtype)], axis=-1)
    x_flat = x.reshape(B, h, L * 2 * L)
    x_flat = mx.concatenate([x_flat, mx.zeros((B, h, L - 1), dtype=x.dtype)], axis=-1)
    x_final = x_flat.reshape(B, h, L + 1, 2 * L - 1)
    return x_final[:, :, :L, L - 1:]


def _dp_abs_to_rel(x: mx.array) -> mx.array:
    """(B, h, L, L) → (B, h, L, 2L-1)."""
    B, h, L, _ = x.shape
    x = mx.concatenate([x, mx.zeros((B, h, L, L - 1), dtype=x.dtype)], axis=-1)
    x_flat = x.reshape(B, h, L * (2 * L - 1))
    x_flat = mx.concatenate([mx.zeros((B, h, L), dtype=x.dtype), x_flat], axis=-1)
    x_final = x_flat.reshape(B, h, L, 2 * L)
    return x_final[:, :, :, 1:]


def _dp_slice_rel(rel: mx.array, length: int, window: int) -> mx.array:
    """(1, 2W+1, d) → (1, 2L-1, d) by zero-padding/slicing."""
    pad_l = max(length - (window + 1), 0)
    if pad_l > 0:
        zero = mx.zeros((1, pad_l, rel.shape[-1]), dtype=rel.dtype)
        padded = mx.concatenate([zero, rel, zero], axis=1)
    else:
        padded = rel
    start = max(window + 1 - length, 0)
    return padded[:, start: start + 2 * length - 1]


class _DPRelPosSelfAttn(nn.Module):
    """VITS-style rel-pos self-attention (2 heads × 32 head_dim, window=4).

    Includes both rel-pos contributions (q × rel_k → logits, abs_to_rel(attn) × rel_v → out).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_q = _DPConvLayer(DP_DIM, DP_DIM)
        self.conv_k = _DPConvLayer(DP_DIM, DP_DIM)
        self.conv_v = _DPConvLayer(DP_DIM, DP_DIM)
        self.conv_o = _DPConvLayer(DP_DIM, DP_DIM)
        self.emb_rel_k = mx.zeros((1, 2 * DP_REL_POS_WINDOW + 1, DP_ATTN_HEAD_DIM))
        self.emb_rel_v = mx.zeros((1, 2 * DP_REL_POS_WINDOW + 1, DP_ATTN_HEAD_DIM))

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, _ = x.shape
        H, D = DP_ATTN_HEADS, DP_ATTN_HEAD_DIM
        q = self.conv_q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.conv_k(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.conv_v(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        scale = D ** -0.5

        logits = (q @ k.transpose(0, 1, 3, 2)) * scale

        rel_k = _dp_slice_rel(self.emb_rel_k, T, DP_REL_POS_WINDOW)
        rel_logits = q @ rel_k.transpose(0, 2, 1)[:, None, :, :]
        rel_logits = _dp_rel_to_abs(rel_logits * scale)
        logits = logits + rel_logits

        if mask is not None:
            key_mask = mask[:, :, 0][:, None, None, :]
            neg_inf = mx.array(-1e4, dtype=logits.dtype)
            logits = mx.where(key_mask.astype(mx.bool_), logits, neg_inf)

        attn = mx.softmax(logits, axis=-1)
        out = attn @ v

        rel_v = _dp_slice_rel(self.emb_rel_v, T, DP_REL_POS_WINDOW)
        rel_weights = _dp_abs_to_rel(attn)
        out = out + rel_weights @ rel_v[:, None, :, :]

        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.conv_o(out)


class _DPFFN(nn.Module):
    """FFN with two Conv1d k=1 — 64 → 256 → 64, ReLU + mask."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = _DPConvLayer(DP_DIM, DP_FFN_HIDDEN)
        self.conv_2 = _DPConvLayer(DP_FFN_HIDDEN, DP_DIM)

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


class _DPAttnEncoder(nn.Module):
    """2× (attn + norm) + (ffn + norm)."""

    def __init__(self) -> None:
        super().__init__()
        self.attn_layers = [_DPRelPosSelfAttn() for _ in range(DP_ATTN_NUM_LAYERS)]
        self.norm_layers_1 = [WrappedNorm(DP_DIM, eps=EPS_LN) for _ in range(DP_ATTN_NUM_LAYERS)]
        self.ffn_layers = [_DPFFN() for _ in range(DP_ATTN_NUM_LAYERS)]
        self.norm_layers_2 = [WrappedNorm(DP_DIM, eps=EPS_LN) for _ in range(DP_ATTN_NUM_LAYERS)]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for i in range(DP_ATTN_NUM_LAYERS):
            x = self.norm_layers_1[i](x + self.attn_layers[i](x, mask=mask))
            x = self.norm_layers_2[i](x + self.ffn_layers[i](x, mask))
        return x


class _DPSentenceEncoder(nn.Module):
    """Text → 64-d sentence vector via prepended ``sentence_token`` slot."""

    def __init__(self) -> None:
        super().__init__()
        class _TextEmb(nn.Module):
            def __init__(_):
                super().__init__()
                _.char_embedder = nn.Embedding(DP_VOCAB, DP_DIM)
            def __call__(_, ids):
                return _.char_embedder(ids)
        self.text_embedder = _TextEmb()
        self.convnext = _DPConvNeXtStack()
        self.attn_encoder = _DPAttnEncoder()
        # proj_out keeps the .net.weight (out, 1, in) Conv1d-k1 layout
        self.proj_out = _DPProjOut()
        # sentence_token (1, DIM, 1) — prepended as the first time slot
        self.sentence_token = mx.zeros((1, DP_DIM, 1))

    def __call__(self, text_ids: mx.array, text_mask: mx.array) -> mx.array:
        x = self.text_embedder(text_ids)            # (B, T, 64)
        # Prepend sentence_token: shape (1, 64, 1) → (B, 1, 64)
        B = x.shape[0]
        sentence = self.sentence_token.transpose(0, 2, 1)
        sentence = mx.broadcast_to(sentence, (B, 1, DP_DIM))
        x = mx.concatenate([sentence, x], axis=1)   # (B, T+1, 64)

        # Extend mask with a leading 1 (sentence token always valid)
        if text_mask is not None:
            extra = mx.ones((B, 1, 1), dtype=text_mask.dtype)
            mask_ntc = mx.concatenate([extra, text_mask.transpose(0, 2, 1)], axis=1)
        else:
            mask_ntc = None

        x = self.convnext(x, mask_ntc)
        x = self.attn_encoder(x, mask_ntc)

        # Take slot 0 (sentence token output) → (B, 1, 64)
        sentence_out = x[:, :1, :]                  # (B, 1, 64)
        # proj_out (Conv1d k=1) — applied along time, output (B, 1, 64)
        sentence_out = self.proj_out(sentence_out)
        return sentence_out.reshape(B, DP_DIM)      # (B, 64)


class _DPProjOut(nn.Module):
    """Conv1d k=1 64→64. No bias in ONNX (confirmed via graph inspection)."""

    def __init__(self) -> None:
        super().__init__()
        class _Net(nn.Module):
            def __init__(_):
                super().__init__()
                _.weight = mx.zeros((DP_DIM, 1, DP_DIM))
            def __call__(_, x):
                return mx.conv1d(x, _.weight, stride=1, padding=0)
        self.net = _Net()

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class _DPPredictor(nn.Module):
    """Linear(192 → 128) + PReLU + Linear(128 → 1).

    PReLU is stored under ``activation.weight (1,)`` — a single learnable
    negative-slope coefficient.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(DP_MLP_IN, DP_MLP_HIDDEN, bias=True),
            nn.Linear(DP_MLP_HIDDEN, 1, bias=True),
        ]
        # PReLU: activation.weight shape (1,) — single scalar slope
        class _Activation(nn.Module):
            def __init__(_):
                super().__init__()
                _.weight = mx.zeros((1,))
            def __call__(_, x):
                # PReLU(x) = max(0, x) + slope * min(0, x)
                neg = mx.minimum(x, mx.array(0.0, dtype=x.dtype))
                pos = mx.maximum(x, mx.array(0.0, dtype=x.dtype))
                return pos + _.weight * neg
        self.activation = _Activation()

    def __call__(self, x: mx.array) -> mx.array:
        h = self.layers[0](x)         # (B, 128)
        h = self.activation(h)
        h = self.layers[1](h)         # (B, 1)
        return h


class _DPRoot(nn.Module):
    """``tts.dp.X`` namespace container."""

    def __init__(self) -> None:
        super().__init__()
        self.sentence_encoder = _DPSentenceEncoder()
        self.predictor = _DPPredictor()


class _DPContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dp = _DPRoot()


class DurationPredictor(nn.Module):
    """Predicts total audio duration (seconds) for an utterance.

    Submodule namespace matches ONNX keys ``tts.dp.X.Y`` exactly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tts = _DPContainer()

    def __call__(
        self,
        text_ids: mx.array,    # (B, T) int
        style_dp: mx.array,    # (B, 8, 16)
        text_mask: mx.array,   # (B, 1, T)
    ) -> mx.array:
        sentence = self.tts.dp.sentence_encoder(text_ids, text_mask)   # (B, 64)
        style_flat = style_dp.reshape(style_dp.shape[0], -1)            # (B, 128)
        joined = mx.concatenate([sentence, style_flat], axis=-1)        # (B, 192)
        log_dur = self.tts.dp.predictor(joined).reshape(-1)             # (B,)
        return mx.exp(log_dur)                                          # duration in seconds


__all__ = ["DurationPredictor"]
