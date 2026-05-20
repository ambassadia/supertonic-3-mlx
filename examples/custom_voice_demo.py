"""Create custom voices by mixing presets.

The 10 preset voices (F1..F5, M1..M5) live on a hypersphere of radius ≈ 7.1
in a 12 800-D style-token space. Spherical-linear interpolation (slerp)
between any two presets lands in the trained distribution and produces a
new, intelligible voice.

    pip install soundfile
    python examples/custom_voice_demo.py
"""
from supertonic_3_mlx import Pipeline
import soundfile as sf

pipe = Pipeline.from_pretrained("ambassadia/supertonic-3-mlx")

TEXT = "Bonjour, je suis une voix personnalisée créée par interpolation des voix préréglées."

# 1. A 70 / 30 mix of two presets — primary F2, slight masculine tint from M1.
voice = pipe.create_voice({"F2": 0.7, "M1": 0.3})
wav = pipe.generate(TEXT, voice=voice, lang="fr")
sf.write("voice_F2_M1.wav", wav, pipe.sample_rate)
print("wrote voice_F2_M1.wav (70 % F2, 30 % M1, slerp)")

# 2. Average of all five female voices — 'mean feminine' timbre.
voice = pipe.create_voice({f"F{i}": 0.2 for i in range(1, 6)})
wav = pipe.generate(TEXT, voice=voice, lang="fr")
sf.write("voice_avg_female.wav", wav, pipe.sample_rate)
print("wrote voice_avg_female.wav")

# 3. Linear interpolation (lerp) instead of slerp — gives a slightly
#    different timbre because lerp doesn't preserve the hypersphere norm.
voice = pipe.create_voice({"F4": 0.6, "F5": 0.4}, interp="lerp")
wav = pipe.generate(TEXT, voice=voice, lang="fr")
sf.write("voice_warm_lerp.wav", wav, pipe.sample_rate)
print("wrote voice_warm_lerp.wav (lerp)")

# 4. A custom voice descriptor is just a dict — you can hand-build it,
#    save it to JSON, share it. The `style_ttl` shape is (1, 50, 256) and
#    `style_dp` shape is (1, 8, 16); both float32. Norms ≈ 7.1 and ≈ 0.3
#    respectively across the 10 presets.
print(f"\nVoice descriptor keys: {sorted(voice.keys())}")
print(f"  style_ttl shape: {voice['style_ttl'].shape}")
print(f"  style_dp  shape: {voice['style_dp'].shape}")
print(f"  blend metadata:  {voice['_meta']}")
