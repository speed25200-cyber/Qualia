"""« ENTRÉE » : rendu image par image du teaser (1920×1080, 30 i/s, 30 s).

Esthétique : trame 1 bit (dithering de Bayer) en noir / papier / bleu
électrique, typographie condensée géante, interface technique, coupes sur
chaque temps de la musique. Chaque image est une fonction pure du temps.

    python3 render.py still 9.4 out.png
    python3 render.py segment 0 225 seg0.mp4
"""
import math
import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import grid as g

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
BUILD = HERE / "build"
W, H, FPS = 1920, 1080, 30
SW, SH = W // 2, H // 2                      # scènes calculées en demi-résolution, puis pixels ×2

INK = np.array([11, 11, 13], np.float32) / 255
PAPER = np.array([236, 234, 227], np.float32) / 255
BLUE = np.array([38, 56, 255], np.float32) / 255
PAL = {                                      # (fond, encre)
    "bw": (INK, PAPER), "wb": (PAPER, INK), "blue": (BLUE, PAPER),
    "kb": (INK, BLUE), "wblue": (PAPER, BLUE),
}
ANTON = "Anton-Regular.ttf"
MONO, MONO_B = "SpaceMono-Regular.ttf", "SpaceMono-Bold.ttf"
SYMBOL = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@lru_cache(None)
def font(name, size):
    return ImageFont.truetype(name if name.startswith("/") else str(FONTS / name), size)


# ---------------------------------------------------------------------------
# Bruit, trame
# ---------------------------------------------------------------------------

LATTICE = np.random.default_rng(7).random((256, 256)).astype(np.float32)
BAYER = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                  [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                  [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                  [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], np.float32)
BAYER = (BAYER + 0.5) / 64


def vnoise(x, y):
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    xi, yi = xi.astype(np.int64) & 255, yi.astype(np.int64) & 255
    x1, y1 = (xi + 1) & 255, (yi + 1) & 255
    u, v = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    a, b = LATTICE[yi, xi], LATTICE[yi, x1]
    c, d = LATTICE[y1, xi], LATTICE[y1, x1]
    return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v


def fbm(x, y, octaves=4):
    s, amp, norm = 0.0, 0.5, 0.0
    for o in range(octaves):
        s = s + amp * vnoise(x, y)
        norm += amp
        x, y, amp = x * 2.03 + 17.1, y * 2.03 + 3.7, amp * 0.5
    return s / norm


def coords(w, h, zoom=1.0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    s = 2.0 / h * zoom
    return (x - w / 2) * s, (y - h / 2) * s


def dither(field, w, h):
    th = np.tile(BAYER, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return (np.clip(field, 0, 1) > th).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectre de la musique (pour le vu-mètre et la scène « barres »)
# ---------------------------------------------------------------------------

def load_spectrum():
    with wave.open(str(BUILD / "music.wav")) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).mean(1) / 32768
    n, bands = 2048, 48
    edges = np.geomspace(40, 12000, bands + 1)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    out = []
    for fi in range(int(g.DURATION * FPS)):
        c = int(fi / FPS * sr)
        seg = x[max(c - n, 0):c]
        seg = np.pad(seg, (n - len(seg), 0))
        mag = np.abs(np.fft.rfft(seg * np.hanning(n)))
        e = [mag[(freqs >= a) & (freqs < b)].mean() if np.any((freqs >= a) & (freqs < b)) else 0
             for a, b in zip(edges[:-1], edges[1:])]
        out.append(np.log10(np.array(e) + 1e-3))
    s = np.array(out)
    s = (s - np.percentile(s, 5)) / (np.percentile(s, 99.5) - np.percentile(s, 5))
    return np.clip(s, 0, 1).astype(np.float32)


SPECTRUM = load_spectrum()


def spec_at(t):
    return SPECTRUM[min(int(t * FPS), len(SPECTRUM) - 1)]


# ---------------------------------------------------------------------------
# Scènes : champ scalaire [0, 1] -> trame
# ---------------------------------------------------------------------------

def sc_specks(X, Y, t, density=0.03):
    n = vnoise(X * 60 + 13, Y * 60 + t * 3) * 0.6 + vnoise(X * 140, Y * 140 - t * 5) * 0.4
    return (n > 1 - density).astype(np.float32) * 1.0


def sc_noise(X, Y, t):
    q = fbm(X * 1.4 + t * 0.25, Y * 1.4, 3)
    f = fbm(X * 2.2 + 2.2 * q + 3, Y * 2.2 + 2.2 * q + t * 0.4, 4)
    return (f - 0.3) * 2.4


def sc_sphere(X, Y, t, cx=0.0, cy=0.0, R=0.72, bands=True):
    dx, dy = X - cx, Y - cy
    d2 = dx * dx + dy * dy
    inside = d2 < R * R
    z = np.sqrt(np.clip(R * R - d2, 0, None))
    nx, ny, nz = dx / R, dy / R, z / R
    lx, ly, lz = math.cos(t * 1.3), -0.55, 0.6 + 0.4 * math.sin(t * 1.3)
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lam = np.clip((nx * lx + ny * ly + nz * lz) / ln, 0, 1)
    shade = 0.08 + 0.92 * lam ** 1.2
    if bands:
        lat = np.arcsin(np.clip(ny, -1, 1))
        shade = shade * (0.75 + 0.25 * (np.sin(lat * 18 + t * 2) > 0))
    bg = 0.04 + 0.06 * (Y + 1)
    return np.where(inside, shade, bg)


def sc_topo(X, Y, t):
    n = fbm(X * 0.9 + t * 0.08, Y * 0.9 - t * 0.05, 4)
    k = n * 14 + t * 0.6
    return (np.abs(k - np.floor(k) - 0.5) < 0.09).astype(np.float32)


def sc_moire(X, Y, t):
    a = 0.35 * math.sin(t * 1.1)
    d1 = np.sqrt((X - a) ** 2 + Y ** 2)
    d2 = np.sqrt((X + a) ** 2 + (Y - 0.15 * math.cos(t)) ** 2)
    return 0.5 + 0.5 * np.sin(d1 * 70) * np.sin(d2 * 70 + t * 2)


def sc_tunnel(X, Y, t, speed=1.6):
    r = np.sqrt(X * X + Y * Y) + 1e-3
    a = np.arctan2(Y, X)
    u = 0.32 / r + t * speed
    v = a * 6 / np.pi + t * 0.35
    chk = ((np.floor(u * 3) + np.floor(v)) % 2).astype(np.float32)
    return chk * np.clip(r * 1.2, 0, 1)


def sc_bars(X, Y, t):
    s = spec_at(t)
    idx = np.clip(((X + 1.7) / 3.4 * len(s)).astype(int), 0, len(s) - 1)
    fracx = (X + 1.7) / 3.4 * len(s) % 1
    h = s[idx] * 1.7
    return ((Y > 0.95 - h) & (fracx < 0.72) & (np.abs(X) < 1.7)).astype(np.float32)


def sc_halftone(X, Y, t, cell=0.07):
    cx = (np.floor(X / cell) + 0.5) * cell
    cy = (np.floor(Y / cell) + 0.5) * cell
    v = fbm(cx * 1.3 + t * 0.35, cy * 1.3 - t * 0.2, 3)
    v = np.clip((v - 0.3) * 2.2, 0, 1)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return (r < v * cell * 0.62).astype(np.float32)


def sc_ridges(X, Y, t, lines=58):
    out = np.zeros_like(X)
    xs = X[0]
    env = np.exp(-(xs / 0.75) ** 4)
    for i in range(lines):
        base = -0.85 + 1.7 * i / (lines - 1)
        amp = 0.35 * env * fbm(xs * 3 + i * 0.7, np.full_like(xs, i * 1.3 + t * 1.2), 3) ** 2 * 2.2
        yl = base - amp
        below = Y > yl[None, :]
        out[below] = 0
        out[np.abs(Y - yl[None, :]) < 0.011] = 1
    return out


SCENES = {
    "specks": sc_specks, "noise": sc_noise, "sphere": sc_sphere, "topo": sc_topo,
    "moire": sc_moire, "tunnel": sc_tunnel, "bars": sc_bars, "halftone": sc_halftone,
    "ridges": sc_ridges,
}


def scene_rgb(name, pal, t, w=SW, h=SH, zoom=1.0, **kw):
    """Scène tramée, colorisée, agrandie ×2 (pixels francs)."""
    X, Y = coords(w, h, zoom)
    bit = dither(SCENES[name](X, Y, t, **kw), w, h)
    bg, fg = PAL[pal]
    img = bg[None, None] * (1 - bit[..., None]) + fg[None, None] * bit[..., None]
    return img.repeat(2, 0).repeat(2, 1)


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------

@lru_cache(None)
def word_mask(text, fname=ANTON, max_w=1760, max_h=620, cy=H // 2, cx=W // 2):
    f = font(fname, 400)
    l, tp, r, b = f.getbbox(text)
    scale = min(max_w / (r - l), max_h / (b - tp))
    size = int(400 * scale)
    f = font(fname, size)
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).text((cx, cy), text, font=f, fill=255, anchor="mm")
    return np.asarray(img, np.float32) / 255


@lru_cache(None)
def line_mask(text, fname, size, x, y, anchor="la"):
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).text((x, y), text, font=font(fname, size), fill=255, anchor=anchor)
    return np.asarray(img, np.float32) / 255


def paint(canvas, mask, color, alpha=1.0):
    m = mask[..., None] * alpha
    canvas *= 1 - m
    canvas += m * color


def rect(canvas, x0, y0, x1, y1, color, alpha=1.0):
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - alpha) + color * alpha


# ---------------------------------------------------------------------------
# Découpage : un plan par temps
# ---------------------------------------------------------------------------

DROP_WORDS = [
    # (mot, scène, palette)
    ("BRUIT", "noise", "bw"), ("FORME", "sphere", "wb"), ("SENS", "topo", "blue"),
    ("DOUTE", "moire", "wb"), ("CODE", "tokens", "kb"), ("POÈME", "ridges", "wb"),
    ("RATURE", "halftone", "blue"), ("RÉÉCRIRE", "tokens", "bw"),
    ("RIRE", "bars", "blue"), ("VERTIGE", "tunnel", "bw"), ("IDÉE", "sphere", "kb"),
    ("AUTRE IDÉE", "noise", "wblue"), ("MIEUX", "topo", "bw"), ("ENCORE", "tunnel", "blue"),
    ("PRESQUE", "ridges", "wb"), ("VOILÀ.", "halftone", "blue"),
]
INVERT = {"bw": "wb", "wb": "bw", "blue": "wblue", "wblue": "blue", "kb": "blue"}
# lettres évidées : la scène vue à travers les lettres prend une palette qui tranche avec le fond
KO_INNER = {"bw": "wb", "wb": "bw", "blue": "bw", "wblue": "blue", "kb": "blue"}

SECTIONS = [(0, "00 — INTRO"), (2, "01 — MONTÉE"), (4, "02 — DROP"), (12, "03 — PAUSE"), (14, "04 — FINAL")]

TOKENS = [
    ("« surprends-moi »", None),
    ("→ lire        0.91", None), ("→ comprendre  0.74", None), ("→ hésiter     0.33", "x"),
    ("→ une image ? 0.58", None), ("→ un son ?    0.61", None), ("→ les deux    0.87", "ok"),
    ("→ 128 bpm     0.66", None), ("→ fa mineur   0.71", None), ("→ trop sage   0.12", "x"),
    ("→ plus fort   0.83", "ok"), ("→ 30 secondes 1.00", "ok"), ("→ un mot/temps 0.79", None),
    ("→ cliché ?    0.21", "x"), ("→ recommencer 0.64", None), ("→ garder      0.92", "ok"),
]


def last_kick(t):
    ks = [k for k in g.kicks() if k <= t]
    return t - ks[-1] if ks else 99.0


def last_clap(t):
    cs = [c for c in g.claps() if c <= t]
    return t - cs[-1] if cs else 99.0


def draw_tokens(canvas, pal, t, beat_idx):
    """Fausse trace de « réflexion » : candidats et probabilités qui défilent."""
    bg, fg = PAL[pal]
    canvas[:] = bg
    off = (t * 2.2) % 1.0
    size, lh = 34, 62
    first = int(t * 2.2)
    for k in range(-1, 16):
        i = (first + k) % len(TOKENS)
        text, mark = TOKENS[i]
        y = 150 + (k - off) * lh
        if y < 90 or y > H - 120:
            continue
        x = 260
        m = line_mask(text, MONO_B, size, x, int(y))
        if mark == "ok":
            rect(canvas, x - 14, y - 6, x + 640, y + lh - 16, BLUE if pal != "blue" else INK)
            paint(canvas, m, PAPER)
        else:
            paint(canvas, m, fg, 0.35 if mark == "x" else 0.9)
            if mark == "x":
                rect(canvas, x, y + 24, x + 560, y + 28, fg, 0.8)


def shot(t):
    """Décrit le plan en cours : dict(scene, pal, word, style, ...)."""
    b = t / g.BEAT
    bi = int(b)
    bar = bi // 4
    if bar < 2:
        text = {2: "AU DÉBUT,", 3: "AU DÉBUT,", 4: "IL N'Y A", 5: "IL N'Y A", 6: "RIEN.", 7: "RIEN."}.get(bi)
        return dict(scene="specks", pal="bw", word=text, style="plain", density=0.012 + 0.01 * b / 8,
                    square=bi < 2)
    if bar < 4:
        if bi == 15:
            return dict(kind="enter")
        if bi >= 13:
            return dict(kind="prompt")
        text = {8: "PUIS", 9: "PUIS", 10: "QUELQU'UN", 11: "QUELQU'UN", 12: "ÉCRIT :"}[bi]
        return dict(scene="specks", pal="wb", word=text, style="plain", density=0.03 + 0.05 * (b - 8) / 5)
    if bar < 12:
        k = (bi - 16) // 2
        word, scene, pal = DROP_WORDS[k]
        second = (bi - 16) % 2 == 1
        if bar >= 8:                      # deuxième moitié : écrans partagés
            style = "split" if not second else "knockout"
        else:
            style = "knockout" if not second else "label"
        if scene == "tokens":
            style = "label"
        if second:
            pal = INVERT[pal]
        return dict(scene=scene, pal=pal, word=word, style=style, index=k)
    if bar < 14:
        if bi >= 53:
            n = {53: "3", 54: "2", 55: "1"}[bi]
            return dict(scene="tunnel", pal=("blue", "bw", "blue")[bi - 53], word=n, style="count")
        text = {48: "TOUT ÇA", 49: "TOUT ÇA", 50: "POUR", 51: "POUR", 52: "UNE QUESTION."}[bi]
        return dict(scene="sphere", pal="wb", word=text, style="plain_small")
    if bar == 14:
        seq = [("tunnel", "blue"), ("noise", "bw"), ("topo", "wblue"), ("moire", "bw")]
        scene, pal = seq[bi - 56]
        return dict(scene=scene, pal=pal, word="ENTRÉE", style="knockout")
    return dict(kind="title")


def hud(canvas, t, ink):
    b = t / g.BEAT
    bar = int(b) // 4
    m = 36
    rect(canvas, m, m, W - m, m + 1, ink, 0.55)
    rect(canvas, m, H - m - 1, W - m, H - m, ink, 0.55)
    rect(canvas, m, m, m + 1, H - m, ink, 0.55)
    rect(canvas, W - m - 1, m, W - m, H - m, ink, 0.55)
    fr = int(round(t * FPS))
    sec = next(s for b0, s in reversed(SECTIONS) if bar >= b0)
    items = [
        ("ENTRÉE — TEASER", 58, 56, "la"), ("CLAUDE / 2026", 58, 82, "la"),
        (f"TC 00:00:{int(t):02d}:{fr % FPS:02d}", W - 58, 56, "ra"),
        (f"MESURE {min(bar, 15) + 1:02d} · TEMPS {int(b) % 4 + 1}", W - 58, 82, "ra"),
        (sec, 58, H - 104, "la"), ("128 BPM · FA MINEUR", 58, H - 78, "la"),
    ]
    for text, x, y, anchor in items:
        paint(canvas, line_mask(text, MONO, 18, x, y, anchor), ink, 0.9)
    # vu-mètre (vraie analyse de la musique)
    s = spec_at(t)[::3]
    for i, v in enumerate(s):
        x = W - 58 - (len(s) - i) * 9
        rect(canvas, x, H - 60 - 44 * v, x + 6, H - 60, ink, 0.9)
    # progression
    rect(canvas, m, H - m - 4, m + (W - 2 * m) * t / g.DURATION, H - m, BLUE if ink is not BLUE else PAPER)


def glitch(img, t):
    """Sur chaque clap : tranches décalées et canaux séparés, pendant ~3 images."""
    dt = last_clap(t)
    if dt > 0.1:
        return img
    k = 1 - dt / 0.1
    r = np.random.default_rng(int(t * 1000))
    out = img.copy()
    for _ in range(5):
        y0 = r.integers(0, H - 40)
        h = r.integers(12, 90)
        out[y0:y0 + h] = np.roll(out[y0:y0 + h], int(r.integers(-60, 60) * k), axis=1)
    sh = int(10 * k)
    out[..., 0] = np.roll(out[..., 0], sh, axis=1)
    out[..., 2] = np.roll(out[..., 2], -sh, axis=1)
    return out


def frame(t):
    b = t / g.BEAT
    bi = int(b)
    s = shot(t)
    kind = s.get("kind", "scene")
    zoom = 1 - 0.07 * math.exp(-last_kick(t) / 0.09)          # « coup » de zoom sur le kick
    local = (b - bi) * g.BEAT
    ink = PAPER

    if kind == "scene":
        pal = s["pal"]
        bg, fg = PAL[pal]
        ink = fg if pal not in ("kb",) else PAPER
        kw = {"density": s["density"]} if "density" in s else {}
        if s["scene"] == "tokens":
            canvas = np.empty((H, W, 3), np.float32)
            draw_tokens(canvas, pal, t, bi)
        else:
            canvas = scene_rgb(s["scene"], pal, t, zoom=zoom, **kw)
        style, word = s["style"], s.get("word")
        if style == "plain" and word:
            paint(canvas, word_mask(word, max_w=1500, max_h=300), fg)
        elif style == "plain_small" and word:
            paint(canvas, word_mask(word, max_w=1200, max_h=180, cy=H - 250), fg)
        elif style == "knockout":
            mask = word_mask(word)
            inner = scene_rgb(s["scene"], KO_INNER[pal], t, zoom=zoom, **kw)
            solid = np.empty_like(canvas)
            solid[:] = bg
            canvas = inner * mask[..., None] + solid * (1 - mask[..., None])
            ink = fg
        elif style == "label":
            big = s["scene"] == "tokens"
            m = word_mask(word, max_w=1100 if big else 900, max_h=300 if big else 150,
                          cy=H // 2 if big else H - 230, cx=W - 560 if big else W // 2)
            ys, xs = np.nonzero(m > 0.5)
            rect(canvas, xs.min() - 30, ys.min() - 26, xs.max() + 30, ys.max() + 26, fg)
            paint(canvas, m, bg)
        elif style == "split":
            # trois panneaux : la scène, sa version inversée, et une autre scène
            other = DROP_WORDS[(s["index"] + 3) % len(DROP_WORDS)]
            a = scene_rgb(s["scene"], pal, t, zoom=zoom)
            c = scene_rgb(other[1] if other[1] != "tokens" else "noise", other[2], t + 3, zoom=zoom)
            inv = scene_rgb(s["scene"], INVERT[pal], t + 1.5, zoom=zoom * 1.6)
            canvas = a.copy()
            canvas[:, 640:1280] = inv[:, 640:1280]
            canvas[:, 1280:] = c[:, 1280:]
            rect(canvas, 638, 0, 642, H, PAPER)
            rect(canvas, 1278, 0, 1282, H, PAPER)
            m = word_mask(word, max_w=1500, max_h=260)
            ys, xs = np.nonzero(m > 0.5)
            rect(canvas, xs.min() - 40, ys.min() - 34, xs.max() + 40, ys.max() + 34, INK)
            paint(canvas, m, BLUE if pal != "blue" else PAPER)
            ink = PAPER
        elif style == "count":
            # le tunnel reste visible autour, le chiffre est plein
            mask = word_mask(word, max_w=900, max_h=760)
            ys, xs = np.nonzero(mask > 0.5)
            rect(canvas, xs.min() - 60, ys.min() - 50, xs.max() + 60, ys.max() + 50, bg)
            paint(canvas, mask, fg)
            ink = fg
        if s.get("square"):
            p = math.exp(-local / 0.25)
            sz = 18 + 30 * p
            rect(canvas, W / 2 - sz / 2, H / 2 - sz / 2, W / 2 + sz / 2, H / 2 + sz / 2, BLUE)

    elif kind == "prompt":
        canvas = scene_rgb("specks", "wb", t, density=0.09)
        text = "> surprends-moi"
        k = int(np.clip((b - 13) / 1.75 * len(text), 0, len(text)))
        box = (220, H // 2 - 90, W - 220, H // 2 + 90)
        rect(canvas, *box, PAPER)
        rect(canvas, box[0], box[1], box[2], box[1] + 4, INK)
        rect(canvas, box[0], box[3] - 4, box[2], box[3], INK)
        if k:
            paint(canvas, line_mask(text[:k], MONO_B, 76, 270, H // 2, "lm"), INK)
        cx = 270 + font(MONO_B, 76).getlength(text[:k]) + 8
        rect(canvas, cx, H // 2 - 42, cx + 40, H // 2 + 42, BLUE)
        ink = INK

    elif kind == "enter":
        canvas = np.empty((H, W, 3), np.float32)
        canvas[:] = INK
        p = math.exp(-local / 0.12)
        size = int(520 + 120 * p)
        paint(canvas, line_mask("⏎", SYMBOL, size, W // 2, H // 2, "mm"), BLUE)

    else:  # titre final
        canvas = np.empty((H, W, 3), np.float32)
        canvas[:] = PAPER
        wm = word_mask("ENTRÉE", max_w=1250, max_h=460, cy=H // 2 - 70, cx=W // 2 - 110)
        paint(canvas, wm, BLUE)
        xs = np.nonzero(wm.max(0) > 0.5)[0]
        paint(canvas, line_mask("⏎", SYMBOL, 300, xs.max() + 190, H // 2 - 60, "mm"), INK)
        paint(canvas, line_mask("Tout commence par une question.", MONO_B, 40, W // 2, H // 2 + 250, "mm"), INK)
        paint(canvas, line_mask("image, musique et montage : Claude", MONO, 22, W // 2, H // 2 + 305, "mm"),
              INK, 0.7)
        ink = INK

    # flash blanc sur les impacts
    for imp in g.IMPACTS:
        if 0 <= t - imp < 2 / FPS:
            canvas[:] = PAPER
            ink = INK
    hud(canvas, t, ink)
    canvas = glitch(canvas, t)
    canvas *= np.clip((g.DURATION - t) / 0.3, 0, 1)
    return (np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8)


def encode(frames, out):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
           "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in frames:
        p.stdin.write(frame(fi / FPS).tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    if sys.argv[1] == "still":
        Image.fromarray(frame(float(sys.argv[2]))).save(sys.argv[3])
    elif sys.argv[1] == "segment":
        encode(range(int(sys.argv[2]), int(sys.argv[3])), sys.argv[4])
