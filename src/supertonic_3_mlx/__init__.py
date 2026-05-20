"""Supertonic 3 — MLX-native TTS for Apple Silicon.

31-language text-to-speech, 5 Euler steps with classifier-free guidance, in
pure MLX. On M4 the full pipeline runs at ~x100 realtime.

Quickstart
----------

    from supertonic_3_mlx import Pipeline
    pipe = Pipeline.from_pretrained("ambassadia/supertonic-3-mlx")
    wav = pipe.generate("Hello world from Apple Silicon.", voice="F1", lang="en")
    # wav is a 1-D ``numpy.float32`` array at 44.1 kHz.

The model weights are released under the BigScience OpenRAIL-M license
(see LICENSE in the Hugging Face repository). This MLX port code is
Apache-2.0. Together they form a dual-license package; Attachment A use
restrictions of OpenRAIL-M govern downstream use of the generated audio.

Public API:
    Pipeline                 — end-to-end TTS, ``from_pretrained`` + ``generate``
    VectorEstimator          — the 24-block CFG flow-matching net (sub-model 1/4)
    TextEncoder              — character → text embedding (sub-model 2/4)
    DurationPredictor        — text → duration in seconds (sub-model 3/4)
    Vocoder                  — latent → 44.1 kHz waveform (sub-model 4/4)
"""
from supertonic_3_mlx._config import (
    DIM, LATENT_CH, CONVNEXT_HIDDEN, CONVNEXT_K,
    NUM_MAIN_BLOCKS, NUM_CYCLES, BLOCKS_PER_CYCLE, BLOCK_CYCLE, STACK4_DILATIONS,
    TEXT_HEADS, TEXT_HEAD_DIM, TEXT_DIM, ROTARY_BASE, ROTARY_SCALE,
    STYLE_HEADS, STYLE_HEAD_DIM, STYLE_LEN, STYLE_DIM,
    TIME_EMB_DIM, TIME_MLP_HIDDEN,
    EPS_LN, CHUNK_COMPRESS, LATENT_DIM, SAMPLE_RATE,
    SUPERTONIC3_HF_REPO,
)
from supertonic_3_mlx.duration_predictor import DurationPredictor
from supertonic_3_mlx.text_encoder import TextEncoder
from supertonic_3_mlx.vector_estimator import VectorEstimator
from supertonic_3_mlx.vocoder import Vocoder
from supertonic_3_mlx.pipeline import SupertonicMLXPipeline as Pipeline

__all__ = [
    "Pipeline",
    "DurationPredictor", "TextEncoder", "VectorEstimator", "Vocoder",
    "DIM", "LATENT_CH", "CONVNEXT_HIDDEN", "CONVNEXT_K",
    "NUM_MAIN_BLOCKS", "NUM_CYCLES", "BLOCKS_PER_CYCLE", "BLOCK_CYCLE", "STACK4_DILATIONS",
    "TEXT_HEADS", "TEXT_HEAD_DIM", "TEXT_DIM", "ROTARY_BASE", "ROTARY_SCALE",
    "STYLE_HEADS", "STYLE_HEAD_DIM", "STYLE_LEN", "STYLE_DIM",
    "TIME_EMB_DIM", "TIME_MLP_HIDDEN",
    "EPS_LN", "CHUNK_COMPRESS", "LATENT_DIM", "SAMPLE_RATE",
    "SUPERTONIC3_HF_REPO",
]
