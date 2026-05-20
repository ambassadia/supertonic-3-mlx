"""Supertonic 3 vocoder — latent → 44.1 kHz waveform, MLX port.

Pipeline (operating in channels-last NTC layout, then converted to channels-first
for output reshape):

    latent  [B, 144, T_lat]    (output of vector_estimator)
      → /= normalizer.scale (scalar)
      → reshape [B, 24, T_lat*6]                        # de-compress
      → (* latent_std + latent_mean)                    # de-normalise
      → transpose to NTC                                 [B, T_lat*6, 24]
      → embed Conv1d(24→512, k=7, sym-edge pad)         [B, T_lat*6, 512]
      → 10× ConvNeXt(dim=512, hidden=2048, k=7,
                     dilations [1,2,4,1,2,4,1,1,1,1])
      → final_norm: BatchNorm1d (eval-time: running stats only)
      → head.layer1: Conv1d(512→2048, k=3, sym-edge pad)
      → PReLU (with per-channel learnable slope)
      → head.layer2: Conv1d(2048→512, k=1, no bias)
      → transpose to (B, 512, T_lat*6) → flatten → wav (B, T_lat*6*512)

The 512 samples/step × 6 chunk × 44.1 kHz → T_lat steps of about 0.0697 s each.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from supertonic_3_mlx._config import EPS_LN
from supertonic_3_mlx._nn_wrappers import WrappedNorm
from supertonic_3_mlx.vector_estimator import _gelu_exact


def _pad_left_edge(x: mx.array, pad: int) -> mx.array:
    """Causal replicate-edge pad on the time axis (axis=1 for [B, T, C]).

    Pads ``pad`` time-steps on the LEFT only by replicating the first frame.
    Matches the ONNX vocoder pads spec ``[0, 0, pad, 0, 0, 0]``.
    """
    if pad == 0:
        return x
    left = mx.broadcast_to(x[:, :1, :], (x.shape[0], pad, x.shape[2]))
    return mx.concatenate([left, x], axis=1)


VOC_DIM = 512
VOC_HIDDEN = 2048
VOC_K = 7
VOC_HEAD_K = 3
VOC_LDIM = 24                       # de-compressed channels (24 × 6 = 144 input)
VOC_CHUNK_COMPRESS = 6
VOC_NUM_CONVNEXT_LAYERS = 10
VOC_DILATIONS = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1)
EPS_BN = 1e-5


class _Conv1dNet(nn.Module):
    """Conv1d wrapped under ``.net`` to match ONNX storage ``.net.weight/bias``."""

    def __init__(self, in_dim: int, out_dim: int, kernel: int, dilation: int = 1,
                 groups: int = 1, bias: bool = True) -> None:
        super().__init__()
        class _Net(nn.Module):
            def __init__(_):
                super().__init__()
                # MLX Conv1d weight: (out, K, in/groups)
                _.weight = mx.zeros((out_dim, kernel, in_dim // groups))
                if bias:
                    _.bias = mx.zeros((out_dim,))
                else:
                    _.bias = None
            def __call__(_, x, dilation=1):
                y = mx.conv1d(x, _.weight, stride=1, padding=0, dilation=dilation,
                              groups=groups)
                if _.bias is not None:
                    y = y + _.bias
                return y
        self.net = _Net()
        self.dilation = dilation
        self.groups = groups
        self.kernel = kernel

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x, dilation=self.dilation)


class _VocConvNeXtBlock(nn.Module):
    """ConvNeXt block matching keys ``convnext.N.{dwconv.net,norm.norm,pwconv1,pwconv2,gamma}``."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.pad = dilation * (VOC_K - 1)
        self.dwconv = _Conv1dNet(VOC_DIM, VOC_DIM, kernel=VOC_K, dilation=dilation,
                                  groups=VOC_DIM, bias=True)
        self.norm = WrappedNorm(VOC_DIM, eps=EPS_LN)
        # pwconv1 / pwconv2 stored as Conv1d k=1 → loaded after squeeze to Linear.
        self.pwconv1 = nn.Linear(VOC_DIM, VOC_HIDDEN, bias=True)
        self.pwconv2 = nn.Linear(VOC_HIDDEN, VOC_DIM, bias=True)
        self.gamma = mx.zeros((VOC_DIM,))

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        y = _pad_left_edge(x, self.pad)
        y = self.dwconv(y)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = _gelu_exact(y)
        y = self.pwconv2(y)
        y = y * self.gamma
        return residual + y


class _BatchNorm1dEval(nn.Module):
    """Eval-mode BatchNorm1d: applies stored running_mean/running_var only.

    Loaded keys: ``norm.{weight,bias,running_mean,running_var}``.
    """

    def __init__(self) -> None:
        super().__init__()
        class _Norm(nn.Module):
            def __init__(_):
                super().__init__()
                _.weight = mx.ones((VOC_DIM,))
                _.bias = mx.zeros((VOC_DIM,))
                _.running_mean = mx.zeros((VOC_DIM,))
                _.running_var = mx.ones((VOC_DIM,))
            def __call__(_, x):
                # x: (B, T, C). BN1d normalises across batch+time per channel.
                # Eval mode: use stored running stats.
                norm = (x - _.running_mean) * mx.rsqrt(_.running_var + EPS_BN)
                return norm * _.weight + _.bias
        self.norm = _Norm()

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(x)


class _VocHeadActivation(nn.Module):
    """PReLU with per-channel learnable slope (weight shape (C,))."""

    def __init__(self) -> None:
        super().__init__()
        # ONNX anonymous PReLU stores slope of shape (1,) sometimes or (C,).
        # We default to (1,) and reshape on load if needed.
        self.weight = mx.zeros((1,))

    def __call__(self, x: mx.array) -> mx.array:
        # PReLU: max(0, x) + slope × min(0, x).
        # slope broadcasts over (B, T, C) or (B, C, T) depending on layout.
        zero = mx.array(0.0, dtype=x.dtype)
        return mx.maximum(x, zero) + self.weight * mx.minimum(x, zero)


class _VocHead(nn.Module):
    """``head.layer1`` (Conv1d 512→2048 k=3) + ``head.act`` (PReLU) + ``head.layer2`` (Conv1d k=1, no bias)."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = _Conv1dNet(VOC_DIM, VOC_HIDDEN, kernel=VOC_HEAD_K, bias=True)
        self.act = _VocHeadActivation()
        # layer2 has no .net wrapper in ONNX (different from layer1)
        # ONNX: head.layer2.weight (512, 2048, 1) — Conv1d k=1, no bias.
        # We represent it directly without .net wrap.
        self.layer2 = _VocLayer2()

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, 512)
        pad = VOC_HEAD_K - 1
        y = _pad_left_edge(x, pad)
        y = self.layer1(y)                          # (B, T, 2048)
        y = self.act(y)
        y = self.layer2(y)                          # (B, T, 512)
        return y


class _VocLayer2(nn.Module):
    """Conv1d k=1 (2048 → 512), no bias. Keys: ``layer2.weight (512, 2048, 1)``."""

    def __init__(self) -> None:
        super().__init__()
        # MLX Conv1d weight shape: (out, K, in/groups) = (512, 1, 2048)
        # ONNX storage: (out, in, 1) = (512, 2048, 1). Same size; reshape on load.
        self.weight = mx.zeros((VOC_DIM, 1, VOC_HIDDEN))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.conv1d(x, self.weight, stride=1, padding=0)


class _VocEmbed(nn.Module):
    """Initial Conv1d(24→512, k=7) with sym-edge pad.

    The weight + bias are anonymous in the ONNX graph (``onnx::Conv_1441`` and
    ``onnx::Conv_1442``); the conversion recovers them via the Conv node path
    ``/decoder/embed/net/Conv`` → structured name ``tts.ae.decoder.embed.net.{weight,bias}``.
    """

    def __init__(self) -> None:
        super().__init__()
        class _Net(nn.Module):
            def __init__(_):
                super().__init__()
                _.weight = mx.zeros((VOC_DIM, VOC_K, VOC_LDIM))
                _.bias = mx.zeros((VOC_DIM,))
            def __call__(_, x):
                return mx.conv1d(x, _.weight, stride=1, padding=0) + _.bias
        self.net = _Net()

    def __call__(self, x: mx.array) -> mx.array:
        pad = VOC_K - 1
        y = _pad_left_edge(x, pad)
        return self.net(y)


class _VocDecoder(nn.Module):
    """``tts.ae.decoder.X`` namespace."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = _VocEmbed()
        self.convnext = [_VocConvNeXtBlock(d) for d in VOC_DILATIONS]
        self.final_norm = _BatchNorm1dEval()
        self.head = _VocHead()


class _AEContainer(nn.Module):
    """``tts.ae.X`` — holds latent_mean, latent_std, decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_mean = mx.zeros((1, VOC_LDIM, 1))
        self.latent_std = mx.ones((1, VOC_LDIM, 1))
        self.decoder = _VocDecoder()


class _TtlContainer(nn.Module):
    """``tts.ttl.normalizer.scale`` (scalar) — divides the latent before de-norm."""

    def __init__(self) -> None:
        super().__init__()
        class _Normalizer(nn.Module):
            def __init__(_):
                super().__init__()
                _.scale = mx.array(1.0)
        self.normalizer = _Normalizer()


class _TtsContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ttl = _TtlContainer()
        self.ae = _AEContainer()


class Vocoder(nn.Module):
    """Latent → waveform decoder (44.1 kHz mono).

    Submodule namespace matches ONNX keys ``tts.X.Y`` exactly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tts = _TtsContainer()

    def __call__(self, latent: mx.array) -> mx.array:
        # latent: (B, 144, T_lat)
        B = latent.shape[0]
        T_lat = latent.shape[2]

        # /= scale (scalar)
        x = latent / self.tts.ttl.normalizer.scale

        # reshape (B, 144, T_lat) → (B, 24, T_lat*6)
        x = x.reshape(B, VOC_LDIM, VOC_CHUNK_COMPRESS, T_lat)   # (B, 24, 6, T_lat)
        x = x.transpose(0, 1, 3, 2)                              # (B, 24, T_lat, 6)
        x = x.reshape(B, VOC_LDIM, T_lat * VOC_CHUNK_COMPRESS)   # (B, 24, T_lat*6)

        # De-normalise: (* std + mean)
        x = x * self.tts.ae.latent_std + self.tts.ae.latent_mean

        # Transpose to NTC for Conv1d layers
        x = x.transpose(0, 2, 1)                                 # (B, T_lat*6, 24)

        # embed
        x = self.tts.ae.decoder.embed(x)                         # (B, T_lat*6, 512)

        # 10× ConvNeXt
        for blk in self.tts.ae.decoder.convnext:
            x = blk(x)

        # final_norm (BatchNorm1d eval)
        x = self.tts.ae.decoder.final_norm(x)

        # head
        x = self.tts.ae.decoder.head(x)                          # (B, T_lat*6, 512)

        # Flatten time × channels row-major → waveform (matches ONNX:
        # head.layer2 Conv (B, 512, T_lat*6) → Transpose to (B, T_lat*6, 512) →
        # Reshape to (B, T_lat*6*512). Since the head already runs in NTC, we
        # are already in the post-Transpose layout and only the Reshape remains).
        wav = x.reshape(B, -1)                                   # (B, T_lat*6*512)
        return wav


__all__ = ["Vocoder", "VOC_DIM", "VOC_HIDDEN", "VOC_LDIM", "VOC_CHUNK_COMPRESS"]
