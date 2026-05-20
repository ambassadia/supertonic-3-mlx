"""Locked hyperparameters for Supertonic 3 MLX port.

Derived from the official ``Supertone/supertonic-3/onnx/tts.json``.
Changing these = re-running parity tests.
"""
from __future__ import annotations

# Vector estimator (the flow-matching denoiser)
DIM: int = 512                  # backbone width
LATENT_CH: int = 144            # 24 * chunk_compress_factor (6)
CONVNEXT_HIDDEN: int = 2048     # main_blocks ConvNeXt intermediate dim (2× vs s2)
CONVNEXT_K: int = 5
LAST_CONVNEXT_NUM: int = 4      # last_convnext is a 4-layer stack (dilations [1,1,1,1])

# 24 main_blocks = 4 cycles × 6 sub-blocks (cycle: stack4, time, cn1, text_attn, cn1, style_attn)
NUM_CYCLES: int = 4
BLOCKS_PER_CYCLE: int = 6
NUM_MAIN_BLOCKS: int = NUM_CYCLES * BLOCKS_PER_CYCLE
BLOCK_CYCLE = ("stack4", "time", "cn1", "text_attn", "cn1", "style_attn")

# ConvNeXt stack 4 (in stack4 blocks) — dilation schedule
STACK4_DILATIONS = (1, 2, 4, 8)

# Text cross-attention (RoPE) — block type "text_attn"
TEXT_DIM: int = 256
TEXT_HEADS: int = 8                       # 2× vs s2 (4)
TEXT_HEAD_DIM: int = DIM // TEXT_HEADS    # 512/8 = 64
ROTARY_BASE: int = 10_000
ROTARY_SCALE: int = 10

# Style cross-attention — block type "style_attn"
STYLE_DIM: int = 256
STYLE_LEN: int = 50               # 50 style tokens (n_style)
STYLE_HEADS: int = 2
STYLE_HEAD_DIM: int = 128

# Time encoding (sinusoidal + MLP)
TIME_EMB_DIM: int = 64
TIME_MLP_HIDDEN: int = 256

# LayerNorm epsilon
EPS_LN: float = 1e-6

# Chunk compress factor (used by AE)
CHUNK_COMPRESS: int = 6
LATENT_DIM: int = 24              # ldim before chunk compression

# Sample rate
SAMPLE_RATE: int = 44_100

# HF references (will be pinned to SHA after first download)
SUPERTONIC3_HF_REPO: str = "Supertone/supertonic-3"
ONNX_VECTOR_ESTIMATOR: str = "onnx/vector_estimator.onnx"
ONNX_TEXT_ENCODER: str = "onnx/text_encoder.onnx"
ONNX_DURATION_PREDICTOR: str = "onnx/duration_predictor.onnx"
ONNX_VOCODER: str = "onnx/vocoder.onnx"
ONNX_TTS_JSON: str = "onnx/tts.json"
ONNX_UNICODE_INDEXER: str = "onnx/unicode_indexer.json"
