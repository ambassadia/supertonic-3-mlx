"""Minimal Supertonic 3 MLX usage — 5 lines, no fluff.

Run from anywhere AFTER ``pip install supertonic-3-mlx`` (or from inside
this directory after ``pip install ./``):

    python examples/quickstart.py
"""
from supertonic_3_mlx import Pipeline
import soundfile as sf

# When the package has been pip-installed, this auto-downloads from the Hub
# (~ 400 MB) into the standard Hugging Face cache. After the first run, the
# weights are reused from cache and cold start is ~ 11 ms on M4.
pipe = Pipeline.from_pretrained("ambassadia/supertonic-3-mlx")

wav = pipe.generate(
    "Hello world from Apple Silicon. Supertonic 3 runs at one hundred times realtime.",
    voice="F1",  # one of F1..F5, M1..M5
    lang="en",   # ISO 639-1
)

sf.write("hello.wav", wav, pipe.sample_rate)
print(f"wrote hello.wav — {len(wav) / pipe.sample_rate:.2f}s of audio")
