"""Supertonic 3 end-to-end MLX pipeline.

Stitches the four MLX sub-models (DurationPredictor → TextEncoder →
VectorEstimator → Vocoder) into a single ``generate(text, voice, lang)`` call
that returns a 44.1 kHz mono numpy waveform.

Flow:

    text  ──tokenize(unicode_indexer)──▶  text_ids (B, T_text)
                                                │
    voice_style (.json)  ──▶  style_ttl (B, 50, 256), style_dp (B, 8, 16)
                                                │
    duration_predictor(text_ids, style_dp, text_mask) ──▶  duration_s (B,)
                                                │
    text_encoder(text_ids, style_ttl, text_mask) ──▶  text_emb (B, 256, T_text)
                                                │
    noise ~ N(0, I) of shape (B, 144, T_lat)
       where T_lat = ceil(duration_s × 44100 / (512 × 6))
                                                │
    vector_estimator 5-step Euler with CFG (4×cond − 3×uncond):
        for step in [0..4]:
            x ← VE(x, text_emb, style_ttl, masks, current_step=step+1, total_step=5)
                                                │
    vocoder(audio_latent) ──▶ wav (B, T_lat × 6 × 512)

Public API:

    pipe = SupertonicMLXPipeline.from_pretrained("/tmp/supertonic3/model")
    wav = pipe.generate("Hello world", voice="F1", lang="en")
    import soundfile as sf
    sf.write("out.wav", wav, pipe.sample_rate)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from supertonic_3_mlx._config import SAMPLE_RATE
from supertonic_3_mlx.duration_predictor import DurationPredictor
from supertonic_3_mlx.text_encoder import TextEncoder
from supertonic_3_mlx.vector_estimator import VectorEstimator
from supertonic_3_mlx.vocoder import Vocoder


# Latent rate: at 44.1 kHz with hop=512 and chunk_compress=6, one latent step
# covers 512 × 6 = 3072 samples = 69.7 ms.
SAMPLES_PER_LATENT_STEP = 512 * 6      # 3072


# ── Shared ONNX → MLX weight extraction ─────────────────────────────


def _convert_onnx(onnx_path: str | Path) -> dict:
    """Return a dict of ``{clean_key: mx.array}`` for a Supertonic ONNX file.

    Combines the three extraction stages discovered during the per-component
    ports (T.3.1, T.3.2, T.3.3):

    1. Named ``tts.*`` initialisers with shape transforms (dwconv, gamma,
       pwconv, head.layer2).
    2. Anonymous MatMul weights recovered via the MatMul output path.
    3. Anonymous Conv weights and PReLU slopes recovered the same way.
    """
    import onnx
    import onnx.numpy_helper as nh

    m = onnx.load(str(onnx_path))

    def _matmul_clean(out_name: str) -> str:
        p = out_name.lstrip("/")
        if p.endswith("/MatMul_output_0"):
            p = p[: -len("/MatMul_output_0")]
        # Drop the leading model-name path (e.g. /text_encoder/, /duration_predictor/, /vector_estimator/)
        for prefix in ("text_encoder/", "duration_predictor/", "vector_estimator/", "vocoder/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        return p.replace("/", ".") + ".weight"

    def _conv_clean(out_name: str) -> str:
        p = out_name.lstrip("/")
        if p.endswith("/Conv_output_0"):
            p = p[: -len("/Conv_output_0")]
        for prefix in ("vocoder/", "vector_estimator/", "text_encoder/", "duration_predictor/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        return "tts.ae." + p.replace("/", ".")

    def _prelu_clean(out_name: str) -> str:
        p = out_name.lstrip("/")
        if p.endswith("/PRelu_output_0"):
            p = p[: -len("/PRelu_output_0")]
        for prefix in ("vocoder/", "vector_estimator/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        return "tts.ae." + p.replace("/", ".") + ".weight"

    # Detect which model this file is — affects how we wrap named init keys
    name_prefixes = {init.name.split(".")[0] for init in m.graph.initializer if "." in init.name}
    is_text_encoder = "tts" in name_prefixes and any(
        i.name.startswith("tts.ttl.text_encoder") for i in m.graph.initializer
    )

    weights: dict[str, mx.array] = {}

    # Stage 1: named initialisers
    for init in m.graph.initializer:
        n = init.name
        # Determine if this is a structured (named) weight or an anonymous graph const
        if not (n.startswith("tts.") or "vector_estimator.tts.ttl." in n or "uncond_masker." in n):
            continue

        # Strip the vector_estimator-specific prefix so all 4 models share a name space.
        if n.startswith("vector_estimator.tts.ttl."):
            clean = n[len("vector_estimator.tts.ttl."):]
        else:
            clean = n

        arr = nh.to_array(init)

        # Shape transforms
        if (clean.endswith(".dwconv.weight") and arr.ndim == 3
                and arr.shape[1] == 1 and arr.shape[2] != 1):
            arr = np.transpose(arr, (0, 2, 1))
        if (clean.endswith(".dwconv.net.weight") and arr.ndim == 3
                and arr.shape[1] == 1):
            arr = np.transpose(arr, (0, 2, 1))
        if (clean.endswith(".gamma") and arr.ndim == 3
                and arr.shape[0] == 1 and arr.shape[2] == 1):
            arr = arr.reshape(arr.shape[1])
        if ((clean.endswith(".pwconv1.weight") or clean.endswith(".pwconv2.weight"))
                and arr.ndim == 3 and arr.shape[-1] == 1):
            arr = arr.squeeze(-1)
        if clean.endswith(".net.weight") and arr.ndim == 3 and arr.shape[-1] == 1:
            # Conv1d k=1 wrapped via .net (e.g. proj_in/proj_out)
            arr = arr.squeeze(-1)
        # vocoder head.layer2 (out, in, 1) → MLX Conv1d (out, K=1, in)
        if clean == "tts.ae.decoder.head.layer2.weight" and arr.ndim == 3:
            arr = np.transpose(arr, (0, 2, 1))
        # vocoder head.layer1.net.weight (out, in, K) → MLX Conv1d (out, K, in)
        if clean == "tts.ae.decoder.head.layer1.net.weight" and arr.ndim == 3:
            arr = np.transpose(arr, (0, 2, 1))

        weights[clean] = mx.array(arr)

    # Stage 2: MatMul weight recovery
    inits_map = {init.name: init for init in m.graph.initializer}
    for node in m.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        winp = node.input[1]
        if winp not in inits_map or winp.startswith("tts.") or "vector_estimator.tts" in winp:
            continue
        arr = nh.to_array(inits_map[winp])
        if arr.ndim == 2:
            arr = arr.T            # ONNX (in, out) → MLX Linear (out, in)
        clean = _matmul_clean(node.output[0])
        # Build the leading namespace from the file context (already in tts.*)
        if not clean.startswith(("tts.", "vector_field.", "uncond_masker.")):
            clean = "tts.ttl." + clean if is_text_encoder else clean
        weights[clean] = mx.array(arr)

    # Stage 3: anonymous Conv + PReLU (vocoder embed / head)
    for node in m.graph.node:
        if node.op_type == "Conv":
            for i, inp in enumerate(node.input[1:], 1):
                if inp not in inits_map or inp.startswith("tts."):
                    continue
                arr = nh.to_array(inits_map[inp])
                base = _conv_clean(node.output[0])
                if "dwconv" in base:
                    continue
                if i == 1 and arr.ndim == 3:
                    arr = np.transpose(arr, (0, 2, 1))   # ONNX (out, in, K) → MLX (out, K, in)
                key = base + (".weight" if i == 1 else ".bias")
                weights[key] = mx.array(arr)
        elif node.op_type == "PRelu":
            for inp in node.input[1:]:
                if inp in inits_map and not inp.startswith("tts."):
                    weights[_prelu_clean(node.output[0])] = mx.array(nh.to_array(inits_map[inp]))

    return weights


def _load_into(model, weights: dict) -> int:
    """Match converted weights to model params (shape-tolerant via reshape).

    Returns the number of successfully matched tensors.
    """
    from mlx.utils import tree_flatten
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    matched = {}
    for k, exp_shape in expected.items():
        if k not in weights:
            continue
        v = weights[k]
        if tuple(v.shape) != exp_shape:
            if v.size == np.prod(exp_shape):
                v = v.reshape(exp_shape)
            else:
                continue
        matched[k] = v
    model.load_weights(list(matched.items()), strict=False)
    return len(matched)


# ── Tokenization ────────────────────────────────────────────────────


_ENDING_PUNCT = ".!?,;:'\")]}»›"


def _preprocess_text(text: str, lang: str = "en") -> str:
    """Mirror the SDK's UnicodeProcessor._preprocess_text contract.

    Supertonic 3 is multilingual; the model is trained with utterances
    wrapped in ``<lang>...</lang>`` language tokens (Supertone's
    ``UnicodeProcessor._add_language_token``). Bypassing this wrapping was
    the secondary bug that compounded with the off-by-one Euler schedule to
    produce structureless audio (verified by ONNX-only ablation in
    ``debug/supertonic3_schedule_ablation.py``).

    Minimum viable port of the SDK's pipeline:
      1. NFKD unicode normalisation
      2. Whitespace collapse + strip
      3. Trailing period if the string doesn't end with punctuation
      4. Language token wrap ``<lang>text</lang>``

    The SDK additionally performs emoji removal, symbol normalisation,
    abbreviation expansion, and quote deduplication — those are quality
    polish and can be ported later; they are not load-bearing for the
    primary fix.
    """
    import unicodedata, re
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in _ENDING_PUNCT:
        text += "."
    if lang is not None:
        text = f"<{lang}>{text}</{lang}>"
    return text


def _encode_text(text: str, indexer: list[int], lang: str = "en") -> np.ndarray:
    """Encode a text string into character IDs via the SDK-compatible pipeline.

    ``indexer`` is a flat list of size 65536; ``indexer[ord(c)]`` gives the
    token ID for character ``c`` (-1 = unknown). The text is first
    preprocessed via :func:`_preprocess_text` so the encoding matches what
    Supertonic 3 was trained on (NFKD-normalised + ``<lang>``-wrapped).
    """
    text = _preprocess_text(text, lang=lang)
    ids = []
    for c in text:
        cp = ord(c)
        if 0 <= cp < len(indexer):
            tok = indexer[cp]
            if tok >= 0:
                ids.append(tok)
    if not ids:
        ids = [indexer[ord(" ")]] if indexer[ord(" ")] >= 0 else [0]
    return np.asarray(ids, dtype=np.int32)


# ── Pipeline ────────────────────────────────────────────────────────


class SupertonicMLXPipeline:
    """End-to-end Supertonic 3 TTS pipeline in pure MLX.

    Loads four sub-models (duration_predictor, text_encoder, vector_estimator,
    vocoder), the unicode tokenizer, and exposes ``generate(text, voice, lang)``.
    """

    sample_rate: int = SAMPLE_RATE
    # Locked by the model architecture: Supertonic 3 is a flow-matching + CFG
    # model trained for exactly 5 Euler steps with t ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
    # and the combination 4×cond − 3×uncond. Any other step count or skipping
    # CFG produces an essentially uncorrelated waveform (verified by
    # ``sub-projects/supertonic3-mlx/bench_n_steps.py``: cosine drops to
    # ≤ 0.5 for n∈{3,4,6} and ≈ 0.05 for cfg=False). Reducing inference
    # latency further would require distilling a shorter-schedule model.
    n_euler_steps: int = 5

    def __init__(
        self,
        duration_predictor: DurationPredictor,
        text_encoder: TextEncoder,
        vector_estimator: VectorEstimator,
        vocoder: Vocoder,
        unicode_indexer: list[int],
        voice_dir: Path,
    ) -> None:
        self.duration_predictor = duration_predictor
        self.text_encoder = text_encoder
        self.vector_estimator = vector_estimator
        self.vocoder = vocoder
        self.unicode_indexer = unicode_indexer
        self.voice_dir = voice_dir

        # T.5 — compile the hot loops. ``mx.compile`` caches a kernel graph keyed
        # by input shapes; the 5× CFG Euler loop and the single vocoder pass
        # both gain from fused kernel dispatch (~50–100 layer ops collapse into
        # one dispatch per cached graph).

        # T.5.3 — also pre-project text and style K/V outside the step. They
        # are invariant across the 5 Euler steps, so the 4 text_attn + 4
        # style_attn blocks no longer re-run their W_key / W_value / RoPE_K
        # matmuls on every step (saves 40 matmuls per generate).
        cond_scale = self.vector_estimator.CFG_COND_SCALE
        uncond_scale = self.vector_estimator.CFG_UNCOND_SCALE

        def _cached_step(
            noisy, lat_mask_2, text_mask_2, t_norm_2, total_step, kv_flat,
        ):
            noisy_2 = mx.concatenate([noisy, noisy], axis=0)
            text_kv = [(kv_flat[2 * i], kv_flat[2 * i + 1]) for i in range(4)]
            style_kv = [(kv_flat[8 + 2 * i], kv_flat[8 + 2 * i + 1]) for i in range(4)]
            v_2 = self.vector_estimator.velocity_cached(
                noisy_2, lat_mask_2, text_mask_2, t_norm_2, text_kv, style_kv,
            )
            B = noisy.shape[0]
            cond_v = v_2[:B]
            uncond_v = v_2[B:]
            combined = cond_scale * cond_v - uncond_scale * uncond_v
            return noisy + combined / total_step.reshape(-1, 1, 1).astype(combined.dtype)

        def _voc_step(latent):
            return self.vocoder(latent)

        self._cached_step_compiled = mx.compile(_cached_step)
        self._voc_compiled = mx.compile(_voc_step)

        # Pick the runtime dtype from any leaf weight of the vector estimator —
        # ``from_pretrained(dtype=...)`` may have cast the model to ``bf16``,
        # in which case all inputs to the compiled hot loops must be cast to
        # match (mixed-dtype Conv/MatMul is not legal in MLX).
        from mlx.utils import tree_flatten
        leaves = [v for _, v in tree_flatten(vector_estimator.parameters())
                  if isinstance(v, mx.array)]
        self.dtype = leaves[0].dtype if leaves else mx.float32

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        dtype: mx.Dtype | None = None,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
    ) -> "SupertonicMLXPipeline":
        """Construct the pipeline from a model snapshot.

        Three sources are accepted, auto-detected:

        1. **Hugging Face Hub repo id** (e.g. ``"ambassadia/supertonic-3-mlx"``):
           weights are downloaded via :func:`huggingface_hub.snapshot_download`
           into ``cache_dir`` (defaults to the standard HF cache) and loaded
           directly from the bundled ``weights/*.safetensors`` files.
        2. **Local path with a** ``weights/`` **subdir**: the MLX-native
           layout (4 safetensors + ``unicode_indexer.json`` + ``voice_styles/``).
           Fast path — no ONNX conversion at runtime.
        3. **Local path with an** ``onnx/`` **subdir**: the upstream
           ``Supertone/supertonic-3`` snapshot layout. Weights are converted
           from ONNX on the fly (~ 1 s per sub-model on M4). Useful for
           development or when starting from the original upstream release.

        Optional kwargs:
          dtype     — if non-None and not float32, cast all weights to the
                      given dtype after load (only ``mx.bfloat16`` is
                      currently meaningful; see README "BF16 note").
          cache_dir — passed to ``huggingface_hub.snapshot_download``.
          revision  — branch / tag / commit sha on the Hub.
        """
        # 1. Resolve the local snapshot directory
        if isinstance(model_id_or_path, str) and "/" in model_id_or_path \
                and not Path(model_id_or_path).exists():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise ImportError(
                    "Loading from the Hugging Face Hub requires "
                    "``huggingface_hub`` — install with ``pip install "
                    "supertonic-3-mlx[hub]`` or ``pip install huggingface_hub``."
                ) from e
            local_dir = Path(snapshot_download(
                repo_id=model_id_or_path,
                cache_dir=cache_dir,
                revision=revision,
                allow_patterns=[
                    "weights/*.safetensors",
                    "unicode_indexer.json",
                    "voice_styles/*.json",
                ],
            ))
        else:
            local_dir = Path(model_id_or_path)

        # 2. Detect layout
        weights_dir = local_dir / "weights"
        onnx_dir = local_dir / "onnx"
        if weights_dir.exists():
            return cls._from_safetensors(local_dir, dtype=dtype)
        if onnx_dir.exists():
            return cls._from_onnx(local_dir, dtype=dtype)
        raise FileNotFoundError(
            f"{local_dir} contains neither ``weights/`` (safetensors layout) "
            f"nor ``onnx/`` (upstream layout); cannot load."
        )

    @classmethod
    def _from_safetensors(
        cls, local_dir: Path, dtype: mx.Dtype | None = None,
    ) -> "SupertonicMLXPipeline":
        from mlx.utils import tree_flatten
        weights_dir = local_dir / "weights"
        voice_dir = local_dir / "voice_styles"
        unicode_indexer = json.loads((local_dir / "unicode_indexer.json").read_text())

        def _build(cls_, name):
            model = cls_()
            w = mx.load(str(weights_dir / f"{name}.safetensors"))
            # Reshape any mismatched leaves (defensive; the converter already
            # produced shape-correct tensors but a future re-export may not).
            expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
            for k in list(w.keys()):
                if k in expected and tuple(w[k].shape) != expected[k]:
                    if w[k].size == int(np.prod(expected[k])):
                        w[k] = w[k].reshape(expected[k])
            model.load_weights(list(w.items()), strict=False)
            return model

        ve = _build(VectorEstimator, "vector_estimator")
        te = _build(TextEncoder, "text_encoder")
        dp = _build(DurationPredictor, "duration_predictor")
        voc = _build(Vocoder, "vocoder")

        # Conditional-style-attention K is the shared style_key bank that lives
        # in TextEncoder. Wire it into VectorEstimator's uncond_masker so the
        # CFG (4*cond - 3*uncond) combine has a valid K on the cond branch.
        # Without this, low-norm voice styles (M3) collapse to near-DC noise.
        ve.uncond_masker.style_key = te.tts.ttl.style_encoder.style_token_layer.style_key

        if dtype is not None and dtype != mx.float32:
            cls._cast_all(dp, te, ve, voc, dtype=dtype)

        return cls(dp, te, ve, voc, unicode_indexer, voice_dir)

    @classmethod
    def _from_onnx(
        cls, local_dir: Path, dtype: mx.Dtype | None = None,
    ) -> "SupertonicMLXPipeline":
        onnx_dir = local_dir / "onnx"
        voice_dir = local_dir / "voice_styles"
        unicode_indexer = json.loads((onnx_dir / "unicode_indexer.json").read_text())

        ve = VectorEstimator()
        _load_into(ve, _convert_onnx(onnx_dir / "vector_estimator.onnx"))
        te = TextEncoder()
        _load_into(te, _convert_onnx(onnx_dir / "text_encoder.onnx"))
        dp = DurationPredictor()
        _load_into(dp, _convert_onnx(onnx_dir / "duration_predictor.onnx"))
        voc = Vocoder()
        _load_into(voc, _convert_onnx(onnx_dir / "vocoder.onnx"))

        ve.uncond_masker.style_key = te.tts.ttl.style_encoder.style_token_layer.style_key

        if dtype is not None and dtype != mx.float32:
            cls._cast_all(dp, te, ve, voc, dtype=dtype)

        return cls(dp, te, ve, voc, unicode_indexer, voice_dir)

    @staticmethod
    def _cast_all(*models, dtype: mx.Dtype) -> None:
        """Cast all fp32 leaves of each model to ``dtype`` (in-place)."""
        from mlx.utils import tree_map

        def _cast(p):
            if not isinstance(p, mx.array) or p.dtype != mx.float32:
                return p
            return p.astype(dtype)

        for m_ in models:
            m_.update(tree_map(_cast, m_.parameters()))

    def _load_voice(self, voice: str) -> tuple[mx.array, mx.array]:
        """Load ``voice_styles/<voice>.json`` and return (style_ttl, style_dp).

        ``voice`` can be either a preset name (``"F1"``..``"F5"``,
        ``"M1"``..``"M5"``) or a custom voice constructed via
        :meth:`create_voice` (then ``voice`` is the dict directly — but
        the helper inside :meth:`generate` handles that case).
        """
        path = self.voice_dir / f"{voice}.json"
        data = json.loads(path.read_text())
        style_ttl = np.asarray(data["style_ttl"]["data"], dtype=np.float32)   # (1, 50, 256)
        style_dp = np.asarray(data["style_dp"]["data"], dtype=np.float32)     # (1, 8, 16)
        return mx.array(style_ttl), mx.array(style_dp)

    # ── Voice mixing API ──────────────────────────────────────────────
    def create_voice(self, blend: dict[str, float],
                     interp: str = "slerp") -> dict[str, mx.array]:
        """Create a custom voice as a weighted mix of preset voices.

        The voice style is a 50×256 ``style_ttl`` tensor that lives on a
        12 800-D hypersphere of radius ≈ 7.1 (verified empirically across
        the 10 presets). Linear or spherical interpolation between the
        preset points stays in the trained distribution and produces
        intelligible new voices.

        Args:
            blend: mapping ``preset_name → weight``. Weights are
                renormalised to sum to 1. Use 2-4 voices for best
                results; mixing more than 4 tends toward the centroid.
            interp: ``"slerp"`` (default, spherical interpolation,
                preserves norm — recommended) or ``"lerp"`` (linear
                weighted average, then renormalise).

        Returns:
            A custom voice descriptor (a dict) that can be passed
            anywhere the API takes a ``voice=...`` argument.

        Examples:
            # 70 % F2 + 30 % M1 → semi-androgynous
            voice = pipe.create_voice({"F2": 0.7, "M1": 0.3})
            wav = pipe.generate("Bonjour", voice=voice, lang="fr")

            # Equal mix of all 5 male voices → 'average male' timbre
            avg_male = pipe.create_voice({f"M{i}": 0.2 for i in range(1, 6)})
        """
        if not blend:
            raise ValueError("blend dict cannot be empty")
        if interp not in ("slerp", "lerp"):
            raise ValueError(f"interp must be 'slerp' or 'lerp', got {interp!r}")

        # Load each preset, normalise weights
        total = sum(blend.values())
        if total <= 0:
            raise ValueError(f"blend weights must sum to > 0, got {total}")
        weights = {k: v / total for k, v in blend.items()}

        ttls: list[tuple[float, np.ndarray]] = []
        dps: list[tuple[float, np.ndarray]] = []
        norms: list[float] = []
        for preset, w in weights.items():
            stl, sdp = self._load_voice(preset)
            stl_np = np.array(stl)
            ttls.append((w, stl_np))
            dps.append((w, np.array(sdp)))
            norms.append(float(np.linalg.norm(stl_np.flatten())))
        target_norm = float(np.mean(norms))

        if interp == "lerp":
            mixed_ttl = sum(w * x for w, x in ttls)
            mixed_dp = sum(w * x for w, x in dps)
        else:
            # SLERP across multiple voices: chain pairwise — order matters.
            # We use a stable iterative slerp from the highest-weighted voice
            # outward (so the final point reflects the dominant voice).
            ordered = sorted(zip(weights.values(), ttls, dps),
                             key=lambda t: -t[0])
            cum_w = ordered[0][0]
            mixed_ttl = ordered[0][1][1].copy()
            mixed_dp = ordered[0][2][1].copy()
            for w, (w_, stl), (_, sdp) in ordered[1:]:
                # The slerp t for this addition is w / (cum_w + w)
                t = w / (cum_w + w)
                a = mixed_ttl.flatten()
                b = stl.flatten()
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                dot = (a @ b) / (na * nb + 1e-8)
                theta = float(np.arccos(np.clip(dot, -1, 1)))
                if theta < 1e-6:
                    mixed_ttl = (1 - t) * mixed_ttl + t * stl
                else:
                    sin_t = np.sin(theta)
                    coef_a = np.sin((1 - t) * theta) / sin_t
                    coef_b = np.sin(t * theta) / sin_t
                    mixed_ttl = (coef_a * a + coef_b * b).reshape(mixed_ttl.shape)
                # dp is small + low-norm, lerp is fine
                mixed_dp = (1 - t) * mixed_dp + t * sdp
                cum_w += w

        # Renormalise ttl to the average source norm
        cur_norm = float(np.linalg.norm(mixed_ttl.flatten()))
        if cur_norm > 1e-6:
            mixed_ttl = mixed_ttl * (target_norm / cur_norm)

        return {
            "style_ttl": mx.array(mixed_ttl.astype(np.float32)),
            "style_dp":  mx.array(mixed_dp.astype(np.float32)),
            "_meta": {"blend": dict(weights), "interp": interp},
        }

    def generate(
        self,
        text: str,
        voice: str = "F1",
        lang: str = "en",
        seed: int = 99,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Synthesise a single utterance. Returns a 1D float32 numpy waveform.

        Note on ``seed``: the initial Gaussian noise draw conditions the
        Euler trajectory the model uses to denoise into audio. Some seed
        values land in a "luckier" region of the noise space — empirically
        ``seed=99`` minimises the worst-case voice (M3 on long FR
        utterances) and maximises Whisper-large-v3 word overlap across
        the (voice × text) matrix: average 98 %, min 87.5 %, σ 3.4 % over
        6 voices × 4 utterances. ``seed=42`` (the previous default)
        scored 75 % on the worst case. If a particular utterance sounds
        garbled, simply retry with another seed: the model is calibrated
        to the SDK schedule but is FP32-noise sensitive on long
        sequences. See ``debug/seed_sweep.py`` for the methodology.
        """
        n_steps = n_steps if n_steps is not None else self.n_euler_steps

        # Tokenize
        text_ids_np = _encode_text(text, self.unicode_indexer, lang)
        text_ids = mx.array(text_ids_np[None, :])                     # (1, T_text)
        T_text = text_ids.shape[1]
        text_mask = mx.ones((1, 1, T_text), dtype=self.dtype)

        # Style — accept either a preset name (str) or a custom voice descriptor
        # (dict returned by ``create_voice``).
        if isinstance(voice, dict):
            style_ttl = voice["style_ttl"]
            style_dp = voice["style_dp"]
        else:
            style_ttl, style_dp = self._load_voice(voice)
        if self.dtype != mx.float32:
            style_ttl = style_ttl.astype(self.dtype)
            style_dp = style_dp.astype(self.dtype)

        # Duration → latent length
        duration_s = self.duration_predictor(text_ids, style_dp, text_mask)
        mx.eval(duration_s)
        duration_val = max(float(duration_s[0].item()), 0.5)          # clamp to ≥ 0.5 s
        T_lat = max(int(math.ceil(duration_val * self.sample_rate / SAMPLES_PER_LATENT_STEP)), 1)

        # Text embedding
        text_emb = self.text_encoder(text_ids, style_ttl, text_mask)  # (1, 256, T_text)

        # Initial noise — fixed seed for reproducibility
        key = mx.random.key(seed)
        noise = mx.random.normal((1, 144, T_lat), key=key).astype(self.dtype)
        latent_mask = mx.ones((1, 1, T_lat), dtype=self.dtype)

        # T.5.3 — build the (2B) CFG conditioning tensors once and pre-project
        # K/V for every text_attn / style_attn block. ``kv_flat`` is the 16
        # ``(K, V)`` arrays flattened into a list for the compiled step.
        B = noise.shape[0]
        ve = self.vector_estimator
        text_uncond = mx.broadcast_to(
            ve.uncond_masker.text_special_token, (B, text_emb.shape[1], text_emb.shape[2])
        ).astype(self.dtype)
        style_k_uncond = mx.broadcast_to(
            ve.uncond_masker.style_key_special_token, (B, style_ttl.shape[1], style_ttl.shape[2])
        ).astype(self.dtype)
        style_v_uncond = mx.broadcast_to(
            ve.uncond_masker.style_value_special_token, (B, style_ttl.shape[1], style_ttl.shape[2])
        ).astype(self.dtype)
        text_emb_2 = mx.concatenate([text_emb, text_uncond], axis=0)
        style_k_2 = mx.concatenate([style_ttl, style_k_uncond], axis=0)
        style_v_2 = mx.concatenate([style_ttl, style_v_uncond], axis=0)
        text_mask_2 = mx.concatenate([text_mask, text_mask], axis=0)
        latent_mask_2 = mx.concatenate([latent_mask, latent_mask], axis=0)

        text_kv, style_kv = ve.precompute_cross_kv(
            text_emb_2, style_k_2, style_v_2, text_mask_2,
        )
        kv_flat = []
        for k, v in text_kv:
            kv_flat.extend([k, v])
        for k, v in style_kv:
            kv_flat.extend([k, v])

        # Euler with CFG — 5 steps by default.
        # NOTE: ONNX SDK passes ``current_step = 0..N-1`` and computes
        # ``t_norm = current_step / total_step`` → schedule = [0.0, 0.2,
        # 0.4, 0.6, 0.8]. Previously we were passing ``step + 1`` which
        # shifted the schedule to [0.2, 0.4, 0.6, 0.8, 1.0]; the flow-matching
        # model is trained on the SDK schedule and the off-by-one collapses
        # the audio to structureless noise (verified by ONNX-only ablation
        # in debug/supertonic3_schedule_ablation.py — wav cosine 0.0037).
        x = noise
        total_step = mx.array([float(n_steps)], dtype=self.dtype)
        for step in range(n_steps):
            current_step = mx.array([float(step)], dtype=self.dtype)
            t_norm = current_step / total_step
            t_norm_2 = mx.concatenate([t_norm, t_norm], axis=0)
            x = self._cached_step_compiled(
                x, latent_mask_2, text_mask_2, t_norm_2, total_step, kv_flat,
            )
            mx.eval(x)

        # Decode latent → waveform
        wav = self._voc_compiled(x)
        mx.eval(wav)
        if wav.dtype != mx.float32:
            wav = wav.astype(mx.float32)
        return np.array(wav)[0]      # (T_lat × 6 × 512,)

    # ── Streaming ────────────────────────────────────────────────────
    @staticmethod
    def _split_for_streaming(text: str, max_chars: int = 220) -> list[str]:
        """Split text into chunks at sentence-ending punctuation.

        Each chunk keeps its terminator. Long sentences exceeding ``max_chars``
        are further split on ``,`` ``;`` ``:`` to keep TTFB low and respect
        the model's training distribution (it sees medium-length utterances).
        """
        import re
        # Split on sentence-ending punctuation, retaining it
        sentences = re.findall(r"[^.!?…]+[.!?…]?", text, flags=re.UNICODE)
        chunks: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) <= max_chars:
                chunks.append(s)
                continue
            # Long sentence — split on secondary punctuation
            parts = re.findall(r"[^,;:]+[,;:]?", s, flags=re.UNICODE)
            buf = ""
            for p in parts:
                if len(buf) + len(p) <= max_chars:
                    buf += p
                else:
                    if buf:
                        chunks.append(buf.strip())
                    buf = p
            if buf:
                chunks.append(buf.strip())
        return chunks

    def generate_stream(
        self,
        text: str,
        voice: str = "F1",
        lang: str = "en",
        seed: int = 99,
        n_steps: Optional[int] = None,
        max_chunk_chars: int = 220,
    ):
        """Generator that yields ``(chunk_idx, wav_chunk)`` tuples as chunks are synthesised.

        The text is split at sentence-ending punctuation (``. ! ?``); long
        sentences are further split at secondary punctuation (``, ; :``) so the
        first chunk reaches the caller in ~ one VE forward (≈ 30-50 ms on M4).
        The caller can start playing chunk 0 while subsequent chunks
        synthesise — TTS speed is x100+ so audio playback never starves.

        Usage:

            for i, wav in pipe.generate_stream("Phrase 1. Phrase 2.", voice="F1", lang="fr"):
                play_audio(wav)              # start playback as soon as chunk 0 arrives

        For non-streaming consumers, use :meth:`SupertonicMLXPipeline.concat_chunks`
        on the collected list.
        """
        chunks = self._split_for_streaming(text, max_chars=max_chunk_chars)
        if not chunks:
            return
        for idx, chunk in enumerate(chunks):
            wav = self.generate(chunk, voice=voice, lang=lang, seed=seed + idx, n_steps=n_steps)
            yield idx, wav

    @staticmethod
    def concat_chunks(chunks: list[np.ndarray], gap_ms: int = 80,
                      sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Concatenate streaming chunks with a short silence between to mask
        the prosody discontinuity that comes from independent generation.

        ``gap_ms`` defaults to 80 ms which roughly matches the natural inter-
        sentence pause in human speech.
        """
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
        out = [chunks[0]]
        for c in chunks[1:]:
            out.extend([gap, c])
        return np.concatenate(out, axis=0)


__all__ = ["SupertonicMLXPipeline"]
