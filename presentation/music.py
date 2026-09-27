"""Musique (nappe ambient), design sonore et mixage avec la voix off.

Entrées : build/voice.wav, build/words.json, build/keys.json (instants de frappe)
Sortie  : build/mix.wav (48 kHz, stéréo)
"""
import json
import re
import unicodedata
import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly

BUILD = Path(__file__).parent / "build"
SR = 48000
DUR = 60.0
T = np.arange(int(SR * DUR)) / SR
RNG = np.random.default_rng(3)

LINES = {l["id"]: l for l in json.loads((BUILD / "words.json").read_text())}


def norm(w):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", w.lower()))


def wt(line, word, end=False):
    w = next(w for w in LINES[line]["words"] if norm(w["w"]) == norm(word))
    return w["end"] if end else w["start"]


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------
# Nappe : un accord par moment du récit
# ---------------------------------------------------------------------------

CHORDS = [
    (0.0, [38, 50, 57, 61, 64, 66]),                           # Ré maj9 : bonjour
    (wt("what", "je") - 0.3, [35, 47, 54, 61, 62, 66]),        # Si m(add9) : ce que je suis
    (wt("learned", "lettres"), [31, 43, 50, 54, 59, 62, 66]),  # Sol maj7 : ce que j'ai lu
    (wt("strange", "mon") - 0.2, [40, 52, 55, 59, 62, 66]),    # Mi m9 : étrange
    (wt("present", "mais"), [45, 57, 62, 66, 69, 73]),         # Ré/La : présent
    (wt("feel", "est-ce"), [43, 55, 59, 61, 66]),              # Sol maj7♯11 : la question
    (wt("like", "ce"), [40, 52, 55, 59, 62, 67]),              # Mi m7 : ce que j'aime
    (wt("like", "trouver"), [45, 57, 62, 64, 69]),             # La sus : le bug
    (wt("together", "et"), [47, 54, 57, 62, 66, 69]),          # Si m7 : ensemble…
    (wt("together", "a"), [38, 50, 57, 61, 64, 66, 69, 73]),   # Ré maj9 : … à deux
]


def pad():
    out = np.zeros((2, len(T)))
    for i, (t0, notes) in enumerate(CHORDS):
        t1 = CHORDS[i + 1][0] if i + 1 < len(CHORDS) else DUR + 5
        env = smooth((T - t0 + 0.4) / 2.2) * (1 - smooth((T - t1 + 0.6) / 2.6))
        idx = np.nonzero(env > 1e-4)[0]
        if len(idx) == 0:
            continue
        tt = T[idx]
        for n in notes:
            f = midi(n)
            weight = 0.55 if n < 45 else (1.0 if n < 70 else 0.6)
            trem = 1 + 0.12 * np.sin(2 * np.pi * RNG.uniform(0.08, 0.2) * tt + RNG.uniform(0, 6))
            for ch, cents in ((0, -5), (1, 5)):
                fd = f * 2 ** (cents / 1200)
                ph = RNG.uniform(0, 2 * np.pi)
                s = (np.sin(2 * np.pi * fd * tt + ph) + 0.22 * np.sin(4 * np.pi * fd * tt + ph)
                     + 0.07 * np.sin(6 * np.pi * fd * tt + ph))
                out[ch, idx] += s * weight * trem * env[idx]
    return out / 12


def bell(t0, note, amp, pan=0.0, decay=1.4):
    out = np.zeros((2, len(T)))
    i0 = int(t0 * SR)
    n = min(int(decay * 4 * SR), len(T) - i0)
    if n <= 0:
        return out
    tt = np.arange(n) / SR
    f = midi(note)
    s = (np.sin(2 * np.pi * f * tt) * np.exp(-tt / decay)
         + 0.3 * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt / (decay * 0.3))
         + 0.12 * np.sin(2 * np.pi * f * 5.4 * tt) * np.exp(-tt / (decay * 0.12)))
    s *= smooth(tt / 0.004) * amp
    out[0, i0:i0 + n] += s * (1 - pan) / 2 * 2 ** 0.5
    out[1, i0:i0 + n] += s * (1 + pan) / 2 * 2 ** 0.5
    return out


def sparkles():
    """Petites cloches pendant que les mots des humains affluent vers la sphère."""
    out = np.zeros((2, len(T)))
    scale = [74, 76, 78, 81, 83, 86, 88, 90, 93]       # Ré majeur pentatonique, aigu
    t, t1 = wt("learned", "jai") - 0.1, wt("you", "vous", end=True)
    t0 = t
    while t < t1:
        p = (t - t0) / (t1 - t0)
        out += bell(t, scale[RNG.integers(len(scale))], 0.035 + 0.03 * p, RNG.uniform(-0.8, 0.8), 1.1)
        t += 1 / (1.5 + 5 * p) * RNG.uniform(0.6, 1.4)
    return out


def clicks(times):
    out = np.zeros((2, len(T)))
    for t in times:
        i0 = int(t * SR)
        n = int(0.03 * SR)
        tt = np.arange(n) / SR
        noise = RNG.normal(0, 1, n)
        noise = np.diff(noise, prepend=0)                  # passe-haut simple : clic « sec »
        s = noise * np.exp(-tt / 0.006) * RNG.uniform(0.05, 0.08)
        s += 0.05 * np.sin(2 * np.pi * RNG.uniform(1700, 2300) * tt) * np.exp(-tt / 0.004)
        pan = RNG.uniform(-0.2, 0.2)
        out[0, i0:i0 + n] += s * (1 - pan)
        out[1, i0:i0 + n] += s * (1 + pan)
    return out


def whoosh(t0, dur, amp):
    """Souffle filtré qui descend : l'anneau se disperse."""
    out = np.zeros((2, len(T)))
    i0, n = int(t0 * SR), int(dur * SR)
    tt = np.arange(n) / SR
    env = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 2 * amp
    for ch in range(2):
        x = RNG.normal(0, 1, n)
        y = np.zeros(n)
        a = 0.02 + 0.25 * (1 - tt / dur) ** 2              # filtre passe-bas qui se referme
        acc = 0.0
        for k in range(n):
            acc += a[k] * (x[k] - acc)
            y[k] = acc
        out[ch, i0:i0 + n] += y * env * 3
    return out


def reverb(x, seconds=3.2, wet=0.35):
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    ir = RNG.normal(0, 1, (2, n)) * np.exp(-tt / 0.9)
    ir[:, : int(0.012 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    wet_sig = np.stack([fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
    return x * (1 - wet) + wet_sig * wet * 2.2


def load_voice():
    with wave.open(str(BUILD / "voice.wav")) as w:
        sr = w.getframerate()
        v = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    v = resample_poly(v, SR, sr)
    out = np.zeros(len(T))
    out[: min(len(v), len(T))] = v[: len(T)]
    return out


def main():
    keys = json.loads((BUILD / "keys.json").read_text())
    voice = load_voice()

    music = pad()
    music += sparkles()
    sfx = clicks(keys["hello"] + keys["ending"])
    t_words = wt("face", "mots")
    for k, n in enumerate([86, 93, 90, 98]):               # les lettres se détachent du texte
        sfx += bell(t_words + 0.05 + 0.07 * k, n, 0.09, (-0.5, 0.5, -0.2, 0.3)[k], 1.6)
    sfx += whoosh(wt("strange", "chaque") - 0.1, 1.0, 0.25)
    for k, n in enumerate([74, 78, 81, 86]):                # l'anneau se reforme
        sfx += bell(wt("strange", "commence") + 0.12 * k, n, 0.05, 0.3 * (k - 1.5), 1.2)
    t_bug = wt("like", "bug")
    sfx += bell(t_bug, 64, 0.07, -0.2, 0.5) + bell(t_bug + 0.28, 69, 0.07, 0.2, 0.7)
    t_claire = wt("like", "chercher") + 0.2 + 3 * 0.26
    sfx += bell(t_claire, 81, 0.06, 0.1, 1.0)
    t_deux = wt("together", "a")
    for k, n in enumerate([62, 69, 74, 78, 81]):            # à deux : l'idée devient claire
        sfx += bell(t_deux + 0.05 * k, n, 0.08, 0.35 * (k - 2), 2.2)

    bed = reverb(music + sfx * 0.9)
    # la musique s'efface un peu sous la voix
    env = np.convolve(np.abs(voice), np.ones(int(0.25 * SR)) / int(0.25 * SR), mode="same")
    duck = 1 - 0.4 * smooth(env / 0.05)
    fade = smooth(T / 1.5) * (1 - smooth((T - (DUR - 2.2)) / 2.1))
    voice_st = np.stack([voice, voice]) * 0.92
    mix = voice_st + bed * duck * fade * 0.45
    mix = np.tanh(mix * 1.05) / np.tanh(1.05)
    mix *= 0.95 / np.abs(mix).max()
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    print(f"voix {rms(voice):.1f} dBFS, fond {rms(bed * fade * 0.45):.1f} dBFS")
    with wave.open(str(BUILD / "mix.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
