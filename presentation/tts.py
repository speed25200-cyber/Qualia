"""Synthèse de la voix off avec Piper (voix fr_FR-siwis-medium).

Produit build/voice.wav et build/timeline.json (début/fin de chaque réplique).
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

from script import LINES

HERE = Path(__file__).parent
BUILD = HERE / "build"
LEAD_IN = 1.2      # silence avant la première réplique
LENGTH_SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.1


def main(model_path):
    BUILD.mkdir(exist_ok=True)
    voice = PiperVoice.load(model_path)
    sr = voice.config.sample_rate
    cfg = SynthesisConfig(length_scale=LENGTH_SCALE, noise_scale=0.6, noise_w_scale=0.7)
    pieces = [np.zeros(int(LEAD_IN * sr), np.float32)]
    t = LEAD_IN
    timeline = []
    for key, caption, spoken, pause in LINES:
        audio = np.concatenate([c.audio_float_array for c in voice.synthesize(spoken or caption, cfg)])
        # retire les silences de bord générés par le modèle
        idx = np.where(np.abs(audio) > 0.01)[0]
        audio = audio[max(idx[0] - 200, 0): idx[-1] + 400]
        dur = len(audio) / sr
        timeline.append({"id": key, "caption": caption, "start": round(t, 3), "end": round(t + dur, 3)})
        pieces += [audio, np.zeros(int(pause * sr), np.float32)]
        t += dur + pause
    audio = np.concatenate(pieces)
    audio = audio / max(np.abs(audio).max(), 1e-6) * 0.89
    with wave.open(str(BUILD / "voice.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    (BUILD / "timeline.json").write_text(json.dumps({"duration": t, "lines": timeline}, ensure_ascii=False, indent=1))
    for x in timeline:
        print(f'{x["start"]:6.2f}-{x["end"]:6.2f}  {x["id"]}')
    print("total", round(t, 2))


if __name__ == "__main__":
    main(sys.argv[1])
