"""Streaming TTS demo — start audio playback before synthesis finishes.

For an interactive agent the time-to-first-byte (TTFB) of the TTS pipeline
determines how snappy the conversation feels. With Supertonic 3 MLX the
first audio chunk is ready in ~ 50 ms on M4 — well under the 100 ms
threshold for "instantaneous".

This example streams chunks into a queue and plays them through
``sounddevice`` in real time. Replace the queue with whatever pipe / WS
connection your app uses.

    pip install sounddevice
    python examples/streaming_demo.py

If you don't have a speaker, drop ``sounddevice`` and just measure the
chunk timings (the loop body shows how to do that).
"""
import time
from supertonic_3_mlx import Pipeline

PARAGRAPH = (
    "Bonjour, je m'appelle Olivier. "
    "Je travaille sur un projet d'intelligence artificielle. "
    "Le modèle Supertonic est porté vers MLX pour fonctionner nativement sur Apple Silicon. "
    "Le streaming permet à l'application de jouer l'audio avant la fin de la synthèse."
)

pipe = Pipeline.from_pretrained("ambassadia/supertonic-3-mlx")

# Optional playback via sounddevice — comment out if not installed
try:
    import sounddevice as sd
    have_audio = True
except ImportError:
    have_audio = False
    print("(install sounddevice for live playback — measuring chunk timings only)")

t_start = time.perf_counter()
for idx, wav in pipe.generate_stream(PARAGRAPH, voice="F2", lang="fr"):
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    label = "← TTFB" if idx == 0 else ""
    print(f"chunk {idx}: ready in {elapsed_ms:>6.0f} ms  ({len(wav) / pipe.sample_rate:>4.2f}s of audio) {label}")
    if have_audio:
        sd.play(wav, pipe.sample_rate, blocking=False)
        sd.wait()

print("\ndone.")
