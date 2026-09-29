"""« ENTRÉE » : morceau électro de 30 s, 128 BPM, fa mineur, entièrement synthétisé.

Aucun échantillon : chaque son (kick, clap, charleston, basse, accords,
arpège, mélodie, nappe, effets) est calculé ici avec NumPy/SciPy.
Sortie : build/music.wav (48 kHz, stéréo, 16 bits).
"""
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt, sosfilt_zi

import grid as g

SR = 48000
N = int(SR * g.DURATION)
TT = np.arange(N) / SR
RNG = np.random.default_rng(128)
BUILD = Path(__file__).parent / "build"


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos"), x)


def saw(f, n, phase=0.0):
    ph = (phase + np.cumsum(np.broadcast_to(f, (n,))) / SR) % 1.0
    return 2 * ph - 1


def adsr(n, a=0.005, d=0.1, s=0.6, r=0.03):
    t = np.arange(n) / SR
    dur = n / SR
    env = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = np.clip((dur - t) / r, 0, 1)
    return env * rel


class Bus:
    def __init__(self):
        self.x = np.zeros((2, N))

    def add(self, start, sig, pan=0.0, gain=1.0):
        i0 = int(round(start * SR))
        if i0 >= N:
            return
        sig = sig[: N - i0] * gain
        l, r = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
        self.x[0, i0:i0 + len(sig)] += sig * l
        self.x[1, i0:i0 + len(sig)] += sig * r


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

def kick(soft=False):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 44 + 120 * np.exp(-t / 0.032)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.2 if soft else 0.3))
    s += RNG.normal(0, 1, n) * np.exp(-t / 0.0018) * (0.05 if soft else 0.35)
    s = np.tanh(2.0 * s)
    return lp(s, 180) * 1.4 if soft else s


def clap():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    noise = RNG.normal(0, 1, n)
    env = sum(np.where(t >= o, np.exp(-(t - o) / 0.005), 0) for o in (0, 0.011, 0.023))
    env = env + 0.8 * np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.11), 0)
    return bp(noise * env, 900, 5000) * 1.6


def snare(v=1.0):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    noise = hp(RNG.normal(0, 1, n), 1500) * np.exp(-t / 0.08)
    return (0.6 * tone + noise) * v


def hat(open_=False):
    n = int((0.3 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    s = hp(RNG.normal(0, 1, n), 7500, 4) * np.exp(-t / (0.14 if open_ else 0.022))
    return s


def crash():
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    return hp(RNG.normal(0, 1, n), 4500, 2) * np.exp(-t / 0.7) * 0.5


def boom():
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    f = 32 + 60 * np.exp(-t / 0.08)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    s += lp(RNG.normal(0, 1, n), 900) * np.exp(-t / 0.25) * 0.6
    return np.tanh(1.5 * s)


def supersaw(notes, dur, voices=5, detune=14, cutoff=3000, a=0.004, d=0.18, s=0.0, r=0.05):
    n = int(dur * SR)
    out = np.zeros(n)
    for m in notes:
        for k in range(voices):
            c = (k - (voices - 1) / 2) / ((voices - 1) / 2) * detune if voices > 1 else 0
            out += saw(hz(m) * 2 ** (c / 1200), n, RNG.random())
    out = lp(out / (voices * len(notes)) * 2.2, cutoff)
    return out * adsr(n, a, d, s, r)


def bass_note(m, dur, cutoff=700):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    s = 0.6 * saw(f, n) + 0.4 * saw(f * 1.006, n, 0.3)
    fenv = cutoff * (0.5 + 1.6 * np.exp(-t / 0.06))
    # filtre enveloppé : traitement par petits blocs
    y, zi, blk = np.zeros(n), None, 256
    for i in range(0, n, blk):
        sos = butter(2, fenv[i], "low", fs=SR, output="sos")
        if zi is None:
            zi = sosfilt_zi(sos) * s[0]
        y[i:i + blk], zi = sosfilt(sos, s[i:i + blk], zi=zi)
    sub = np.sin(2 * np.pi * f / 2 * t)
    return (np.tanh(1.6 * y) * 0.7 + sub * 0.42) * adsr(n, 0.003, 0.12, 0.7, 0.02)


def pluck(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    s = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + 0.5 * saw(f * 1.003, n)
    return s * np.exp(-t / 0.09) * adsr(n, 0.002, 1, 1, 0.01)


def lead_note(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + (2 ** (10 / 1200) - 1) * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
    f = hz(m) * vib
    s = 0.55 * saw(f, n) + 0.45 * np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))
    return lp(s, 3200) * adsr(n, 0.008, 0.3, 0.75, 0.08)


def blip(m, dur=0.25):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t / 0.07)


def riser(dur, lo=300, hi=9000):
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    noise = RNG.normal(0, 1, n)
    y, zi, blk = np.zeros(n), None, 512
    for i in range(0, n, blk):
        fc = lo * (hi / lo) ** u[i]
        sos = butter(2, [fc * 0.7, min(fc * 1.4, SR * 0.45)], "band", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y[i:i + blk], zi = sosfilt(sos, noise[i:i + blk], zi=zi)
    tone = np.sin(2 * np.pi * np.cumsum(110 * 2 ** (3 * u)) / SR) * 0.25
    return (y * 2.5 + tone) * u ** 2


def reverse_cymbal(dur):
    return crash()[: int(dur * SR)][::-1] * 1.2


# ---------------------------------------------------------------------------
# Arrangement
# ---------------------------------------------------------------------------

def arp_cutoff(time):
    b = time / g.BAR
    if b < 4:
        return 600 * (7000 / 600) ** (b / 4) ** 1.4
    if b < 12:
        return 5200
    if b < 13:
        return 900
    return 6000


def main():
    drums, bass, syn, fx, send = Bus(), Bus(), Bus(), Bus(), Bus()

    # --- batterie
    for k in g.kicks():
        drums.add(k, kick(), gain=0.95)
    for k in g.claps():
        s = clap()
        drums.add(k, s, gain=0.5)
        send.add(k, s, gain=0.25)
    for k in g.heartbeats():
        drums.add(k, kick(soft=True), gain=0.9)
        drums.add(k + g.BEAT * 0.3, kick(soft=True), gain=0.55)
    for bar in range(g.BARS):
        for six in range(16):
            tm = g.t(bar, six / 4)
            if bar == 2 and six % 2 == 0:
                drums.add(tm, hat(), pan=0.3, gain=0.18 + 0.08 * (six % 4 == 2))
            if bar == 3 and six < 12:
                drums.add(tm, hat(), pan=0.3, gain=0.2)
            if (g.DROP[0] <= bar < g.DROP[1] or bar == 14) and six % 4 == 2:
                drums.add(tm, hat(open_=True), pan=-0.25, gain=0.2)
            if 8 <= bar < 12 and six % 4 != 2:
                drums.add(tm, hat(), pan=0.35, gain=0.1 + 0.06 * (six % 2))
    # roulements : fin de montée et fin de pause
    for k, beat in enumerate([0, 0.5, 1, 1.5, 2, 2.25, 2.5, 2.75]):
        drums.add(g.t(3, beat), snare(0.3 + 0.7 * k / 7), gain=0.4)
    for six in range(8, 16):
        drums.add(g.t(13, six / 4), snare(0.3 + 0.7 * (six - 8) / 7), gain=0.42)
    for six in (12, 13, 14, 15):
        drums.add(g.t(11, six / 4), snare(0.7), gain=0.35)

    # --- effets
    fx.add(g.t(2), riser(g.GAP[0] - g.t(2)), gain=0.4)
    fx.add(g.GAP[0], reverse_cymbal(g.BEAT), gain=0.6)
    fx.add(g.t(12, 3), riser(g.t(14) - g.t(12, 3), 200, 7000), gain=0.35)
    for tm in g.IMPACTS:
        fx.add(tm, boom(), gain=0.7)
        s = crash()
        fx.add(tm, s, gain=0.55)
        send.add(tm, s, gain=0.3)
    for k, m in enumerate((81, 84, 88)):                    # 3, 2, 1
        s = blip(m)
        fx.add(g.t(13, 1 + k), s, gain=0.3)
        send.add(g.t(13, 1 + k), s, gain=0.4)

    # --- basse
    for bar in range(g.BARS):
        root, _ = g.chord_at(bar)
        if bar in (0, 1):
            bass.add(g.t(bar), np.sin(2 * np.pi * hz(root - 12) * np.arange(int(1.7 * SR)) / SR)
                     * np.exp(-np.arange(int(1.7 * SR)) / SR / 0.7), gain=0.3)
        if g.DROP[0] <= bar < g.DROP[1] or bar == 14:
            for e in range(8):
                if bar == 11 and e >= 6:
                    continue
                m = root + (12 if e == 7 else 0)
                bass.add(g.t(bar, e / 2), bass_note(m, g.BEAT / 2 * 0.9), gain=0.5 + 0.12 * (e % 2))
    bass.add(g.t(15), bass_note(41, 1.6, 400), gain=0.6)

    # --- accords (stabs) et nappe
    for bar in list(range(*g.DROP)) + [14]:
        _, ch = g.chord_at(bar)
        for six in (2, 6, 11, 14):
            s = supersaw(ch, 0.3)
            syn.add(g.t(bar, six / 4), s, gain=0.32)
            send.add(g.t(bar, six / 4), s, gain=0.18)
    for bar, ch in ((12, g.CHORDS[1][1]), (13, g.CHORDS[3][1])):
        s = supersaw([m - 12 for m in ch] + ch, g.BAR + 0.4, voices=7, detune=20, cutoff=1500,
                     a=0.5, d=1.0, s=0.9, r=0.4)
        syn.add(g.t(bar), s, gain=0.4)
        send.add(g.t(bar), s, gain=0.45)
    final = supersaw([53, 60, 65, 68, 72], 1.9, voices=7, detune=18, cutoff=4000, a=0.003, d=0.8, s=0.3, r=0.3)
    syn.add(g.t(15), final, gain=0.45)
    send.add(g.t(15), final, gain=0.6)

    # --- arpège
    arp = Bus()
    for bar in range(g.BARS):
        if bar == 13 or bar == 15:
            continue
        _, ch = g.chord_at(bar)
        seq = [ch[0], ch[1], ch[2], ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[0] + 12, ch[2]]
        for six in range(16):
            tm = g.t(bar, six / 4)
            if g.GAP[0] <= tm < g.GAP[1]:
                continue
            arp.add(tm, pluck(seq[six % 8], g.SIX * 0.95), pan=0.45 * (1 if six % 2 else -1), gain=0.26)
    # filtre qui s'ouvre au fil du morceau (blocs d'une double croche)
    blk = int(g.SIX * SR)
    zi = [None, None]
    for i in range(0, N, blk):
        sos = butter(2, arp_cutoff(i / SR), "low", fs=SR, output="sos")
        for c in range(2):
            if zi[c] is None:
                zi[c] = np.zeros((sos.shape[0], 2))
            arp.x[c, i:i + blk], zi[c] = sosfilt(sos, arp.x[c, i:i + blk], zi=zi[c])
    syn.x += arp.x
    send.x += arp.x * 0.2

    # --- mélodie (mesures 8-11, reprise en 14)
    melody = [
        (8, 0, 72, 1), (8, 1, 75, .5), (8, 1.5, 72, .5), (8, 2, 68, 1), (8, 3, 67, .5), (8, 3.5, 68, .5),
        (9, 0, 73, 1.5), (9, 1.5, 72, .5), (9, 2, 68, 2),
        (10, 0, 72, 1), (10, 1, 75, .5), (10, 1.5, 77, .5), (10, 2, 75, 1), (10, 3, 72, 1),
        (11, 0, 70, 3), (11, 3, 67, .5), (11, 3.5, 70, .5),
    ]
    lead = Bus()
    for bar, beat, m, d in melody + [(14, b, m, d) for (bb, b, m, d) in melody if bb == 8]:
        lead.add(g.t(bar, beat), lead_note(m, d * g.BEAT * 0.95), gain=0.2)
    delay = int(3 * g.SIX * SR)
    echo = np.zeros_like(lead.x)
    for k, gain in enumerate((0.38, 0.22, 0.12), 1):
        c = k % 2                                       # écho en ping-pong
        echo[c, delay * k:] += lead.x.sum(0)[:-delay * k] * gain / 2
    syn.x += lead.x + echo
    send.x += (lead.x + echo) * 0.3

    # --- compression « pompe » sur la basse et les synthés
    pump = np.ones(N)
    for k in g.kicks():
        i0 = int(k * SR)
        n = min(int(0.4 * SR), N - i0)
        tt = np.arange(n) / SR
        pump[i0:i0 + n] = np.minimum(pump[i0:i0 + n], 1 - 0.75 * np.exp(-tt / 0.09))
    bass.x *= pump
    syn.x *= 0.35 + 0.65 * pump

    # --- réverbération
    n_ir = int(2.4 * SR)
    t_ir = np.arange(n_ir) / SR
    ir = RNG.normal(0, 1, (2, n_ir)) * np.exp(-t_ir / 0.55)
    ir = np.stack([lp(hp(ir[c], 300), 7000) for c in range(2)])
    ir /= np.sqrt((ir ** 2).sum(1, keepdims=True))
    wet = np.stack([fftconvolve(send.x[c], ir[c])[:N] for c in range(2)]) * 0.9

    mix = drums.x + bass.x + syn.x + fx.x + wet
    # silence net juste avant le drop (sauf la cymbale inversée)
    gap = np.ones(N)
    a, b = int(g.GAP[0] * SR), int(g.GAP[1] * SR)
    gap[a:b] = 0
    mix = mix * gap + fx.x * (1 - gap)
    # fondu de sortie très court (pas de clic à 30 s)
    mix *= np.clip((g.DURATION - TT) / 0.35, 0, 1)
    mix = hp(mix, 25)
    mix /= np.abs(mix).max()
    mix = np.tanh(1.8 * mix) / np.tanh(1.8)             # colle et saturation douce
    mix *= 0.93 / np.abs(mix).max()

    BUILD.mkdir(exist_ok=True)
    with wave.open(str(BUILD / "music.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
    rms = 20 * np.log10(np.sqrt((mix ** 2).mean()))
    print(f"music.wav : {g.DURATION:.1f} s, RMS {rms:.1f} dBFS")


if __name__ == "__main__":
    main()
