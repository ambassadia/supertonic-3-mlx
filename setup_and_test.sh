#!/usr/bin/env bash
# Quick install + sanity-test for the supertonic-3-mlx standalone package.
#
# Creates a local ``.venv`` next to this script, installs the package and its
# runtime deps, version-checks MLX, downloads the model weights from the
# Hugging Face Hub on first run, and synthesises one short utterance to
# ``hello.wav``. Idempotent: re-running reuses the existing venv and cached
# weights.
#
# Usage:
#   ./setup_and_test.sh                 # default: en F1, "Hello world…"
#   ./setup_and_test.sh fr F2 "Bonjour."
#
set -euo pipefail

# ── 0. Inputs ────────────────────────────────────────────────────────
LANG_CODE="${1:-en}"
VOICE="${2:-F1}"
TEXT="${3:-Hello world from Apple Silicon. Supertonic 3 runs at one hundred times realtime.}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

# ── 1. Python version gate ──────────────────────────────────────────
if ! command -v python3 >/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first." >&2
    exit 1
fi
PYVER="$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
PYMAJ="${PYVER%.*}"; PYMIN="${PYVER#*.}"
if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required, found $PYVER." >&2
    exit 1
fi
echo "→ python3: $PYVER"

# ── 2. venv ─────────────────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ]; then
    echo "→ creating venv at $VENV …"
    python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

# ── 3. dependencies ─────────────────────────────────────────────────
echo "→ installing dependencies …"
"$PIP" install --quiet --upgrade pip
# Install the package + the optional runtime deps. The package itself pulls in
# mlx + numpy via its pyproject.toml; we add huggingface_hub for the Hub
# download path, hf_transfer for large-blob throughput, and soundfile so the
# test script can write a WAV.
"$PIP" install --quiet "$HERE" huggingface_hub soundfile

# ── 4. MLX version gate + optional patch hook ───────────────────────
"$PY" - <<'PYEOF'
import sys

try:
    import mlx.core as mx
except ImportError:
    print("ERROR: mlx not importable. Are you on Apple Silicon? "
          "MLX is macOS-on-Apple-Silicon only.", file=sys.stderr)
    sys.exit(1)

ver_str = getattr(mx, "__version__", "0.0.0")
ver = tuple(int(p) for p in ver_str.split(".")[:3] if p.isdigit())
print(f"→ mlx version: {ver_str}")

# Minimum tested combination — bumped as the upstream API changes.
MIN_OK = (0, 21, 0)
if ver < MIN_OK:
    print(f"  WARNING: mlx < {'.'.join(map(str, MIN_OK))}. Upgrading …",
          file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "--upgrade", "mlx"])
    print("  → upgraded. Re-run the script to pick up the new version.",
          file=sys.stderr)
    sys.exit(2)

# Patches we currently know about — none. This is the slot where future
# MLX-specific shims would land (e.g. a workaround for an upstream Conv1d
# regression). Keep the dispatch table here so the script stays a single
# source of truth.
PATCHES: dict[tuple[int, int, int], str] = {
    # (broken_version): "patch description"
}
applied = [desc for v, desc in PATCHES.items() if v == ver]
if applied:
    for desc in applied:
        print(f"  applied patch: {desc}")
else:
    print(f"  no patches needed for mlx {ver_str}")
PYEOF

# ── 5. quickstart generate ──────────────────────────────────────────
echo "→ generating audio …"
LANG_CODE="$LANG_CODE" VOICE="$VOICE" TEXT="$TEXT" \
HF_HUB_DISABLE_XET=1 \
HF_HUB_ENABLE_HF_TRANSFER=1 \
"$PY" - <<'PYEOF'
import os, time
from supertonic_3_mlx import Pipeline
import soundfile as sf

lang = os.environ["LANG_CODE"]
voice = os.environ["VOICE"]
text = os.environ["TEXT"]

# First call downloads ~ 400 MB of weights into the HF cache. Subsequent
# runs reuse the cache and load in ~ 20 ms.
t0 = time.perf_counter()
pipe = Pipeline.from_pretrained("ambassadia/supertonic-3-mlx")
load_t = time.perf_counter() - t0
print(f"  load        : {load_t*1000:.0f} ms")

# Warmup (compiles the kernel graph for this shape).
pipe.generate("Warm.", voice=voice, lang=lang)

t0 = time.perf_counter()
wav = pipe.generate(text, voice=voice, lang=lang, seed=42)
gen_t = time.perf_counter() - t0
dur = len(wav) / pipe.sample_rate
print(f"  generate    : {gen_t*1000:.0f} ms")
print(f"  audio       : {dur:.2f} s ({len(wav)} samples @ {pipe.sample_rate} Hz)")
print(f"  RTF         : x{dur/gen_t:.0f}")
print(f"  max amp     : {abs(wav).max():.4f}")

sf.write("hello.wav", wav, pipe.sample_rate)
print("\n✓ wrote hello.wav — open it to verify the synthesis sounds correct.")
PYEOF
