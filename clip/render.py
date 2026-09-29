"""« Synthetic Scream » : clip vidéo, rendu image par image (1920×1080, 30 i/s).

Trois images générées (Higgsfield / GPT Image 2.5) : la ville, l'androïde,
le cri. Tout le reste (mouvements de caméra, pluie, parasites, glitchs,
lasers, étoiles, labyrinthe, électrocardiogramme, visage en texte,
paroles, interface) est calculé ici, synchronisé sur l'analyse du morceau
(features.py) et sur les paroles horodatées (lyrics.py).

    python3 render.py still 38.9 out.png
    python3 render.py segment 0 1400 seg0.mp4
"""
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import cine
import fx
from fx import BLACK, BOLD, CYAN, H, MEDIUM, MONO, RED, W, WHITE
from lyrics import lines as lyric_lines

HERE = Path(__file__).parent
BUILD = HERE / "build"
FPS = 30

F = np.load(BUILD / "features.npz")
DURATION = float(F["duration"])
NFRAMES = int(math.ceil(DURATION * FPS))
BEATS, KICKS, SNARES = F["beats"], F["kicks"], F["snares"]
LYRICS = lyric_lines()

# positions des yeux dans l'image de l'androïde (coordonnées normalisées)
EYES = [(0.401, 0.343), (0.581, 0.339)]


def feat(name, t):
    a = F[name]
    return float(a[min(int(t * FPS), len(a) - 1)])


def since(events, t):
    i = np.searchsorted(events, t, side="right") - 1
    return t - events[i] if i >= 0 else 99.0


def pulse(events, t, tau=0.09):
    return math.exp(-since(events, t) / tau)


def beat_index(t):
    return int(np.searchsorted(BEATS, t, side="right") - 1)


def ramp(t, a, b):
    x = min(max((t - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def current_line(t, styles=None):
    for ln in LYRICS:
        if ln["start"] - 0.08 <= t < ln["end"] and (styles is None or ln["style"] in styles):
            return ln
    return None


def line_named(text, t):
    ln = current_line(t)
    return ln is not None and ln["text"] == text


SECTIONS = [
    (0.0, "boot"), (2.9, "intro"), (16.3, "build"), (29.9, "verse1"), (57.5, "chorus1"),
    (85.3, "noescape"), (98.8, "verse2"), (126.75, "chorus2"), (143.0, "fading"),
    (155.0, "climax"), (178.0, "outro"),
]


OPEN_FRAME = [(57.5, 85.3), (126.75, 143.0), (155.0, 178.0)]   # refrains : plein cadre


def bar_height(t):
    """Hauteur des bandes noires (format 2.39:1), qui s'ouvrent sur les refrains."""
    o = max(ramp(t, a, a + 0.25) * (1 - ramp(t, b - 0.3, b)) for a, b in OPEN_FRAME)
    return 138.0 * (1 - o)


def section(t):
    name, t0 = SECTIONS[0][1], 0.0
    for s0, n in SECTIONS:
        if t >= s0:
            name, t0 = n, s0
    return name, t0


# ---------------------------------------------------------------------------
# Scènes procédurales
# ---------------------------------------------------------------------------

def lasers(t, strength=1.0, n=7, color_seed=0):
    img = Image.new("RGB", (W // 2, H // 2), 0)
    d = ImageDraw.Draw(img)
    ox, oy = W // 4, -20
    for i in range(n):
        a = math.pi / 2 + 0.9 * math.sin(t * (0.7 + 0.13 * i) + i * 1.7)
        ln = 1400
        wdt = 0.012 + 0.01 * (i % 3)
        p1 = (ox + ln * math.cos(a - wdt), oy + ln * math.sin(a - wdt))
        p2 = (ox + ln * math.cos(a + wdt), oy + ln * math.sin(a + wdt))
        col = (255, 40, 70) if (i + color_seed) % 3 else (40, 220, 255)
        d.polygon([(ox, oy), p1, p2], fill=col)
    img = img.filter(ImageFilter.GaussianBlur(3))
    arr = np.asarray(img, np.float32) / 255
    return arr.repeat(2, 0).repeat(2, 1) * strength


_STAR_RNG = np.random.default_rng(12)
STARS = np.stack([_STAR_RNG.uniform(0, W, 420), _STAR_RNG.uniform(0, H, 420),
                  _STAR_RNG.uniform(20, 90, 420), _STAR_RNG.uniform(1.5, 4.5, 420),
                  _STAR_RNG.random(420)], 1)


def neon_stars(t):
    img = Image.new("RGB", (W // 2, H // 2), 0)
    d = ImageDraw.Draw(img)
    for x, y, v, r, c in STARS:
        yy = (y - v * t * 2) % H
        xx = x + 20 * math.sin(t * 0.8 + c * 6)
        col = (255, 50, 80) if c < 0.55 else (60, 230, 255)
        rr = r * (0.7 + 0.3 * math.sin(t * 5 + c * 20))
        d.ellipse([xx / 2 - rr, yy / 2 - rr, xx / 2 + rr, yy / 2 + rr], fill=col)
    img = img.filter(ImageFilter.GaussianBlur(1.6))
    return (np.asarray(img, np.float32) / 255).repeat(2, 0).repeat(2, 1)


_MAZE_RNG = np.random.default_rng(5)
MAZE_CELL = 60
MAZE_SEGS = []
for gy in range(0, H // MAZE_CELL + 1):
    for gx in range(0, W // MAZE_CELL + 1):
        x, y = gx * MAZE_CELL, gy * MAZE_CELL
        if _MAZE_RNG.random() < 0.5:
            MAZE_SEGS.append((x, y, x + MAZE_CELL, y + MAZE_CELL))
        else:
            MAZE_SEGS.append((x + MAZE_CELL, y, x, y + MAZE_CELL))
MAZE_ORDER = _MAZE_RNG.permutation(len(MAZE_SEGS))


def maze(progress):
    img = Image.new("RGB", (W, H), 0)
    d = ImageDraw.Draw(img)
    k = int(len(MAZE_SEGS) * min(progress, 1))
    for i in MAZE_ORDER[:k]:
        d.line(MAZE_SEGS[i], fill=(40, 230, 255), width=3)
    glow = img.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3)).resize((W, H))
    return (np.asarray(img, np.float32) + np.asarray(glow, np.float32) * 1.5) / 255


def ecg(t, bpm, color=RED, y=H * 0.5, amp=220, speed=700):
    """Tracé d'électrocardiogramme qui défile."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    period = 60 / bpm
    pts = []
    for x in range(0, W + 8, 6):
        tt = t - (W - x) / speed
        ph = (tt % period) / period
        v = 0.0
        if 0.10 < ph < 0.14:
            v = 0.15 * math.sin((ph - 0.10) / 0.04 * math.pi)
        elif 0.18 < ph < 0.20:
            v = -0.2
        elif 0.20 < ph < 0.23:
            v = 1.0 - abs(ph - 0.215) / 0.015
        elif 0.23 < ph < 0.25:
            v = -0.35
        elif 0.35 < ph < 0.45:
            v = 0.25 * math.sin((ph - 0.35) / 0.10 * math.pi)
        pts.append((x, y - v * amp))
    d.line(pts, fill=255, width=4)
    m = np.asarray(img, np.float32) / 255
    glow = np.asarray(img.filter(ImageFilter.GaussianBlur(8)), np.float32) / 255
    return m, glow


@lru_cache(None)
def _ascii_grid(cols=112, rows=42):
    face = Image.fromarray(fx.to_uint8(fx.camera("android", 0.49, 0.47, 1.55)))    # cadrage serré sur le visage
    im = face.convert("L").resize((cols, rows), Image.BILINEAR)
    lum = np.asarray(im, np.float32) / 255
    lo, hi = np.percentile(lum, 8), np.percentile(lum, 99.5)
    lum = np.clip((lum - lo) / (hi - lo), 0, 1) ** 0.75          # contraste étiré : le visage ressort
    col = face.resize((cols, rows), Image.BILINEAR)
    c = np.asarray(col, np.float32) / 255
    red = np.clip((c[..., 0] - np.maximum(c[..., 1], c[..., 2])) * 3, 0, 1)
    return lum, red


def ascii_face(t, phrase="MACHINE WHISPERS INSIDE MY HEAD "):
    """Le visage de l'androïde recomposé avec les mots qui lui murmurent."""
    cols, rows = 112, 42
    lum, red = _ascii_grid(cols, rows)
    img = Image.new("RGB", (W, H), 0)
    d = ImageDraw.Draw(img)
    f = fx.font(MONO, 25)
    cw, ch = W / cols, H / rows
    off = int(t * 14)
    for r in range(rows):
        for c in range(cols):
            v = lum[r, c]
            if v < 0.16:
                continue
            k = (r * cols + c + off) % len(phrase)
            char = phrase[k]
            if char == " ":
                continue
            g = int(min(255, 25 + v ** 1.6 * 340))
            col = (g, int(g * 0.25), int(g * 0.3)) if red[r, c] > 0.3 else (int(g * 0.8), g, g)
            d.text((c * cw, r * ch), char, font=f, fill=col)
    return np.asarray(img, np.float32) / 255


def droste(base, t, levels=7, speed=0.55):
    """Cadres imbriqués qui avancent sans fin : pas d'issue."""
    ratio = 0.72
    z = (t * speed) % 1.0
    out = np.zeros_like(base)
    pil = Image.fromarray(fx.to_uint8(base))
    for k in range(levels, -1, -1):
        s = ratio ** (k - z)
        if s > 3.5:
            continue
        w, h = int(W * s), int(H * s)
        if w < 8:
            continue
        im = np.asarray(pil.resize((w, h), Image.BILINEAR), np.float32) / 255
        x0, y0 = (W - w) // 2, (H - h) // 2
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + w, W), min(y0 + h, H)
        out[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0]
        # liseré rouge
        for (p, q, rr, ss) in ((ya, xa, ya + 3, xb), (yb - 3, xa, yb, xb), (ya, xa, yb, xa + 3), (ya, xb - 3, yb, xb)):
            out[max(p, 0):max(rr, 0), max(q, 0):max(ss, 0)] = RED
    return out


def eye_reticles(img, t, cx, cy, zoom, color=RED, name="android"):
    """Réticules de visée sur les yeux (suivent le cadrage)."""
    iw, ih = fx.source(name).size
    s = (iw / W) / zoom
    for k, (ex, ey) in enumerate(EYES):
        sx = W / 2 + (ex - cx) * iw / s
        sy = H / 2 + (ey - cy) * ih / s
        r = 70 + 10 * math.sin(t * 6 + k)
        pil = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(pil)
        d.ellipse([sx - r, sy - r, sx + r, sy + r], outline=255, width=3)
        for a in range(4):
            ang = a * math.pi / 2 + t * 1.5
            d.line([(sx + math.cos(ang) * (r + 8), sy + math.sin(ang) * (r + 8)),
                    (sx + math.cos(ang) * (r + 34), sy + math.sin(ang) * (r + 34))], fill=255, width=3)
        d.text((sx + r + 16, sy - r), f"TGT-{k + 1:02d}  LOCK", font=fx.font(MONO, 22), fill=255)
        img = fx.paint(img, np.asarray(pil, np.float32) / 255, color, 0.9)
    return img


# ---------------------------------------------------------------------------
# Interface (vision de l'androïde)
# ---------------------------------------------------------------------------

def hud(img, t, fi, label="", color=CYAN, alpha=0.8):
    m = np.zeros((H, W), np.float32)
    L = 60
    top, bot = int(bar_height(t)) + 34, H - int(bar_height(t)) - 34
    for (x, y, sx, sy) in ((50, top, 1, 1), (W - 50, top, -1, 1), (50, bot, 1, -1), (W - 50, bot, -1, -1)):
        xa, xb = sorted((x, x + sx * L))
        ya, yb = sorted((y, y + sy * L))
        m[y - 1:y + 2, xa:xb] = 1
        m[ya:yb, x - 1:x + 2] = 1
    img = fx.paint(img, m, color, alpha)
    tc = f"{int(t // 60):02d}:{int(t % 60):02d}:{fi % FPS:02d}"
    img = fx.paint(img, fx.text_mask(f"REC ● {tc}", MONO, 24, W - 80, top + 38, "rm"), color, alpha)
    if label:
        img = fx.paint(img, fx.text_mask(label, MONO, 24, 80, top + 38, "lm"), color, alpha)
    # niveaux audio
    bars = [feat("sub", t), feat("bass", t), feat("mid", t), feat("high", t), feat("rms", t)]
    for i, v in enumerate(bars):
        h = int(min(v, 1.2) * 60)
        x = 80 + i * 14
        img[bot - 30 - h:bot - 30, x:x + 9] = img[bot - 30 - h:bot - 30, x:x + 9] * (1 - alpha) + color * alpha
    return img


def terminal(img, lines_, t0, t, x=90, y=H - 330, color=RED, size=28, cps=38):
    """Lignes de terminal tapées."""
    chars = int((t - t0) * cps)
    for i, ln in enumerate(lines_):
        if chars <= 0:
            break
        s = ln[:chars]
        chars -= len(ln) + 4
        img = fx.paint(img, fx.text_mask(s, MONO, size, x, y + i * (size + 12), "lm"), color, 0.95)
    return img


# ---------------------------------------------------------------------------
# Paroles
# ---------------------------------------------------------------------------

def draw_lyrics(img, t, fi):
    ln = current_line(t)
    if ln is None:
        return img
    style = ln["style"]
    fade = 1 - ramp(t, ln["end"] - 0.25, ln["end"])
    if style == "center":
        prog = min(1.0, (t - ln["start"]) / 0.45)
        txt = fx.scramble(ln["text"], prog, fi)
        jitter = int(4 * pulse(SNARES, t))
        img = fx.glitch_text(img, txt, BOLD, 110, W / 2 + jitter, H / 2, fade, split=6, tracking=14)
    elif style == "line":
        # révélation mot à mot, façon terminal
        # chaque mot arrive en fondu, en remontant de quelques pixels
        size = 58 if len(ln["text"]) < 26 else 46
        ly = int(H - bar_height(t) - 80)
        if not any(ts <= t + 0.05 for _, ts in ln["words"]):
            return img
        blink = fade * (0.55 + 0.45 * math.cos(t * 9))
        img[ly - size // 2:ly + size // 2, 92:99] = (
            img[ly - size // 2:ly + size // 2, 92:99] * (1 - blink) + RED * blink)
        bright = float(fx.luminance(img[ly - 40:ly + 40, 100:900]).mean()) > 0.6
        col = BLACK if bright else WHITE
        f = fx.font(BOLD, size)
        x = 124.0
        for w, ts in ln["words"]:
            if ts > t + 0.05:
                break
            age = t - ts
            a = fade * ramp(age, -0.05, 0.12)
            dy = int(14 * (1 - ramp(age, -0.05, 0.18)))
            img = fx.glitch_text(img, w, BOLD, size, int(x), ly + dy, a, split=3 + 6 * math.exp(-age / 0.1),
                                 anchor="lm", tracking=4, color=col)
            x += f.getlength(w + " ") + 4 * (len(w) + 1)
    elif style == "slam":
        cur = None
        for w, ts in ln["words"]:
            if ts <= t + 0.03:
                cur = (w, ts)
        if cur is None:
            return img
        w, ts = cur
        age = t - ts
        a = fade * (1 - ramp(age, 0.9, 1.3))
        scale = 1 + 0.25 * math.exp(-age / 0.07)
        size = int(min(300, 1700 / max(len(w), 3) * 1.35) * scale)
        dx = int(np.random.default_rng(fi).integers(-8, 9) * math.exp(-age / 0.15))
        if age < 0.16:                                  # traînée de zoom à l'impact
            for k, (sc, al) in enumerate(((1.35, 0.18), (1.18, 0.3))):
                img = fx.glitch_text(img, w, BOLD, int(size * sc), W / 2, H / 2, a * al * (1 - age / 0.16),
                                     split=0, tracking=6)
        img = fx.glitch_text(img, w, BOLD, size, W / 2 + dx, H / 2, a, split=10 * math.exp(-age / 0.2) + 3, tracking=6)
    elif style == "grid":
        a = fade * ramp(t, ln["start"], ln["start"] + 0.1)
        rows = 7
        for r in range(rows):
            off = (t * 180 * (1 if r % 2 else -1)) % 900
            txt = ("NO ESCAPE   " * 6)
            col = RED if r % 2 else WHITE
            m = fx.text_mask(txt, BOLD, 120, int(-off), int(80 + r * 150), "lm")
            img = fx.paint(img, m, col, a * (0.9 if r == 3 else 0.35))
    return img


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

def shake(t, amount):
    """Tremblement de caméra à l'épaule : bruit lisse (somme de sinus), pas de sautillement."""
    dx = (math.sin(t * 23.1) + 0.6 * math.sin(t * 37.7 + 1.1) + 0.35 * math.sin(t * 61.3 + 2.3)) / 1.95
    dy = (math.sin(t * 19.7 + 0.4) + 0.6 * math.sin(t * 41.9 + 2.0) + 0.35 * math.sin(t * 57.1 + 0.7)) / 1.95
    return dx * amount, dy * amount


def whip(img, t, events, dur=0.075, strength=90):
    """Flou de filé sur les premières images après une coupe."""
    k = since(events, t)
    if k < dur:
        n = int(strength * (1 - k / dur)) | 1
        img = cv2.blur(img, (n, 1))
    return img


PAR_BOOST = [0.0, 0.0]          # travelling latéral propre au plan en cours (monté par montage_shot)


def auto_par(t):
    """Flottement continu de la caméra : le premier plan glisse devant le fond."""
    return (0.02 * math.sin(t * 0.41) + PAR_BOOST[0], 0.01 * math.sin(t * 0.29 + 1.3) + PAR_BOOST[1])


def city(t, cx=0.5, cy=0.5, zoom=1.0, shake_amt=0.0, rot=0.0, par=None, dolly=0.0, focus=None, dof=0.0):
    dx, dy = shake(t, shake_amt)
    return cine.camera("city", cx, cy, zoom, rot, dx, dy, par or auto_par(t), dolly, focus, dof)


def android(t, cx=0.5, cy=0.45, zoom=1.0, shake_amt=0.0, rot=0.0, name="android", par=None, dolly=0.0,
            focus=None, dof=0.0):
    dx, dy = shake(t, shake_amt)
    return cine.camera(name, cx, cy, zoom, rot, dx, dy, par or auto_par(t), dolly, focus, dof)


CHORUS_SHOTS = ["eyes", "android_wide", "city_wide", "city_crop", "lasers", "android_sort", "chrome", "city_red"]
CHORUS2_SHOTS = ["scream", "scream_mouth", "scream_eyes", "scream_shards", "android_red", "city_wide", "lasers",
                 "scream_wide", "scream_invert", "chrome"]


def pick(seq, bi, salt):
    """Choix pseudo-aléatoire d'un plan par temps, sans répéter le précédent."""
    r = np.random.default_rng(bi * 7919 + salt)
    prev = np.random.default_rng((bi - 1) * 7919 + salt).integers(len(seq))
    k = r.integers(len(seq))
    if k == prev:
        k = (k + 1 + r.integers(len(seq) - 1)) % len(seq)
    return seq[k]


def montage_shot(kind, t, bi, fi, kick):
    r = np.random.default_rng(bi)
    lt = since(BEATS, t)
    PAR_BOOST[0] = r.choice([-1, 1]) * 0.05 * min(lt, 0.6)       # chaque plan a son propre mouvement
    PAR_BOOST[1] = r.uniform(-1, 1) * 0.02 * min(lt, 0.6)
    try:
        return _montage_shot(kind, t, bi, fi, kick, r)
    finally:
        PAR_BOOST[0] = PAR_BOOST[1] = 0.0


def _montage_shot(kind, t, bi, fi, kick, r):
    z = 1 + 0.06 * kick
    if kind == "eyes":
        cx, cy, zz = 0.49, 0.345, 3.0 * z
        img = fx.red_only(android(t, cx, cy, zz, 0.002))
        return eye_reticles(img, t, cx, cy, zz)
    if kind == "android_wide":
        return fx.rgb_split(android(t, 0.5, 0.45, 1.1 * z, 0.003), 8 * kick + 2)
    if kind == "city_wide":
        return fx.grade(city(t, 0.5, 0.5, 1.15 * z, 0.004), contrast=1.2)
    if kind == "city_crop":
        return city(t, r.uniform(0.2, 0.8), r.uniform(0.45, 0.75), 2.3 * z, 0.004)
    if kind == "city_red":
        img = fx.grade(city(t, 0.5, 0.55, 1.4 * z, 0.006), tint=RED, tint_amt=0.7, contrast=1.3)
        return img
    if kind == "lasers":
        base = fx.grade(city(t, 0.5, 0.45, 1.3, 0.003), exposure=0.35)
        return base + lasers(t, 1.1, color_seed=bi)
    if kind == "android_sort":
        img = android(t, 0.5, 0.42, 1.5 * z, 0.002)
        return fx.pixel_sort(img, 260, 700, vertical=True, x0=700, x1=1300)
    if kind == "chrome":
        img = android(t, 0.62, 0.42, 2.4 * z, 0.002)
        return fx.slices(img, r, 5, 60)
    if kind == "android_red":
        return fx.grade(android(t, 0.5, 0.45, 1.25 * z, 0.004), tint=RED, tint_amt=0.8, contrast=1.4)
    if kind == "scream":
        return android(t, 0.5, 0.48, 1.25 * z, 0.006, name="scream")
    if kind == "scream_wide":
        return fx.rgb_split(android(t, 0.5, 0.5, 1.0 * z, 0.004, name="scream"), 6 + 10 * kick)
    if kind == "scream_mouth":
        return android(t, 0.5, 0.66, 2.6 * z, 0.008, name="scream")
    if kind == "scream_eyes":
        return fx.red_only(android(t, 0.5, 0.3, 2.4 * z, 0.006, name="scream"))
    if kind == "scream_shards":
        return fx.mirror_shards(android(t, 0.5, 0.48, 1.3 * z, 0.004, name="scream"), r, 8)
    if kind == "scream_invert":
        return fx.invert(android(t, 0.5, 0.48, 1.5 * z, 0.004, name="scream"))
    raise ValueError(kind)


def base_frame(t, fi):
    """Image principale du plan en cours (avant paroles et finitions)."""
    sec, t0 = section(t)
    kick = pulse(KICKS, t, 0.08)
    snare = pulse(SNARES, t, 0.07)
    bass = feat("bass", t)
    bi = beat_index(t)
    rng = np.random.default_rng(fi)
    ln = current_line(t)
    text = ln["text"] if ln else ""
    post = dict(rain=0.0, bloom=0.45, hud=None, static=0.0)

    if sec == "boot":
        img = np.zeros((H, W, 3), np.float32)
        img = terminal(img, ["> SYNTHETIC_OS  v0.9", "> NEURAL LINK .......... OK", "> SIGNAL ............... SEARCHING"],
                       0.3, t, x=180, y=H // 2 - 60, size=40)
        post["static"] = 0.9 if (t > 2.45 or (fi % 23) < 2) else 0.0

    elif sec == "intro":
        p = (t - t0) / (16.3 - t0)
        img = city(t, 0.5, 0.5 - 0.03 * p, 1.2 + 0.1 * p)
        img = fx.grade(img, exposure=0.8, sat=0.55, tint=CYAN, tint_amt=0.25)
        burst = 0.0
        if ln is not None and t - ln["start"] < 0.25:
            burst = 0.8
        post.update(rain=0.3, static=max(0.7 - 0.55 * p, burst))
        if (fi // 3) % 17 == 0:
            img = np.roll(img, int(rng.integers(-200, 200)), axis=0)       # décrochage vertical

    elif sec == "build":
        p = (t - t0) / (29.9 - t0)
        zoom = 1.05 + 0.75 * p ** 1.6 + 0.04 * kick
        img = city(t, 0.5, 0.47, zoom, 0.0015 + 0.004 * bass, dolly=0.5 * p ** 1.3)
        img = fx.grade(img, exposure=0.8 + 0.35 * feat("high", t), contrast=1.15)
        if snare > 0.5:
            img = fx.slices(fx.rgb_split(img, 10 * snare), rng, 5, 60)
        if t > 24.0 and since(SNARES, t) < 2 / FPS:                         # éclairs subliminaux
            img = fx.red_only(android(t, 0.49, 0.345, 3.2))
        post.update(rain=0.4, hud="SECTOR-07 // CURFEW ACTIVE")

    elif sec == "verse1":
        img, post = verse1(t, fi, ln, kick, snare, bass, rng, post)

    elif sec == "chorus1":
        hold = 1 if feat("rms", t) > 0.55 else 2
        if t > 81.2:                                                        # HARDER / FASTER : croches
            sub = int((t - BEATS[bi]) / ((BEATS[min(bi + 1, len(BEATS) - 1)] - BEATS[bi]) / 2 + 1e-6))
            key = bi * 2 + sub
        else:
            key = bi // hold
        kind = pick(CHORUS_SHOTS, key, 11)
        img = montage_shot(kind, t, key, fi, kick)
        img = whip(img, t, BEATS)
        if snare > 0.6 and kind not in ("eyes",):
            img = img * 0.6 + RED * 0.4 * snare
        post.update(bloom=0.7)

    elif sec == "noescape":
        base = android(t, 0.5, 0.45, 1.05)
        if t < 92.6:
            img = droste(fx.grade(base, contrast=1.2), t, speed=0.6)
            post.update(static=0.1)
        else:
            img = droste(fx.grade(base, exposure=0.7, sat=0.3), t, speed=0.25)
            post.update(static=0.35 + 0.3 * math.sin(t * 3) ** 2)

    elif sec == "verse2":
        img, post = verse2(t, fi, ln, kick, snare, bass, rng, post)

    elif sec == "chorus2":
        if t < 127.1:                                                       # coupe franche : le cri
            img = np.ones((H, W, 3), np.float32)
        else:
            if "BREAK" in text or "THEY" in text or text.startswith("NO NO"):
                # chaque mot répété : nouvel éclat
                wi = sum(1 for w, ts in ln["words"] if ts <= t)
                img = fx.mirror_shards(android(t, 0.5, 0.48, 1.2 + 0.1 * wi, 0.006, name="scream"),
                                       np.random.default_rng(wi * 31), 6 + wi)
            else:
                kind = pick(CHORUS2_SHOTS, bi, 23)
                img = whip(montage_shot(kind, t, bi, fi, kick), t, BEATS)
            if snare > 0.6:
                img = img * 0.55 + WHITE * 0.45 * snare
        post.update(bloom=0.75)

    elif sec == "fading":
        img, post = fading(t, fi, ln, kick, snare, rng, post)

    elif sec == "climax":
        p = (t - t0) / (178 - t0)
        eighth = t > 170.5
        key = bi * 2 + int((t - BEATS[bi]) * 4.5) if eighth else bi
        kind = pick(CHORUS2_SHOTS + ["scream", "scream_mouth"], key, 37)
        img = whip(montage_shot(kind, t, key, fi, kick), t, BEATS)
        if snare > 0.5:
            img = fx.blocks(img, rng, 10)
        if kick > 0.8 and rng.random() < 0.25:
            img = fx.invert(img)
        post.update(bloom=0.8, hud="!! SYSTEM OVERRIDE !!")
        words = ["SYNTHETIC", "SCREAM"]
        if snare > 0.2:
            w = words[beat_index(t) // 2 % 2]
            img = fx.glitch_text(img, w, BOLD, 260, W / 2, H / 2, snare * (0.6 + 0.4 * p), split=12, tracking=10)

    else:  # outro
        img, post = outro(t, fi, post)

    return img, post


def verse1(t, fi, ln, kick, snare, bass, rng, post):
    text = ln["text"] if ln else ""
    lt = t - ln["start"] if ln else 0.0
    post.update(rain=0.3)
    if text == "CLOSED STREETS BURN" or not text:
        img = city(t, 0.26 + 0.03 * lt, 0.7, 2.0 + 0.04 * kick, 0.002)
        img = fx.grade(img, exposure=1.1, tint=RED, tint_amt=0.3, contrast=1.15)
    elif text == "LIGHTS DECAY":
        flick = 0.45 + 0.55 * (1 - lt / 1.7) * (0.6 + 0.4 * rng.random())
        img = fx.grade(city(t, 0.5, 0.45, 1.25, focus=0.95 - 0.8 * min(lt / 1.5, 1), dof=7),
                       exposure=max(flick, 0.3))
    elif text == "VOICES GLITCH, FADE AWAY":
        img = city(t, 0.5, 0.45, 1.35)
        img = fx.blocks(fx.slices(img, rng, 8, 120), rng, int(6 + 10 * lt))
        post["static"] = min(0.65, 0.15 + lt * 0.15)
    elif text == "CHROME ON SKIN":
        img = fx.grade(android(t, 0.61, 0.42, 2.3 + 0.12 * lt, focus=0.1 + 0.8 * ramp(lt, 0, 0.9), dof=9),
                       tint=CYAN, tint_amt=0.15)
        post["rain"] = 0.2
    elif text == "EYES IN RED":
        cx, cy, z = 0.49, 0.345, 3.0 + 0.1 * lt
        img = eye_reticles(fx.red_only(android(t, cx, cy, z)), t, cx, cy, z)
        post["hud"] = "OPTICS // THERMAL"
    elif text == "CITY SCREAMS":
        img = fx.rgb_split(city(t, 0.5, 0.5, 1.35 + 0.08 * kick, 0.012), 14 * kick + 3)
    elif text == "FEED THE DEAD":
        img = fx.grade(city(t, 0.72, 0.68, 2.2, 0.003), exposure=0.8, tint=RED, tint_amt=0.6, contrast=1.3)
    elif text == "BASSLINE BITES":
        img = fx.rgb_split(android(t, 0.5, 0.45, 1.25 + 0.2 * bass + 0.06 * kick, 0.002), 12 * bass)
    elif text == "STEEL AND SMOKE":
        img = fx.grade(city(t, 0.22, 0.55, 2.0 - 0.1 * lt), sat=0.15, contrast=1.2)
    elif text == "BROKEN DREAMS":
        seed = beat_index(t)
        img = fx.mirror_shards(android(t, 0.5, 0.45, 1.4), np.random.default_rng(seed), 11)
    elif text == "NEURAL CHOKE":
        img = android(t, 0.5, 0.38, 1.8)
        img = fx.pixel_sort(img, 380, 380 + int(160 + 300 * min(lt, 1)))
    elif text == "LOCK THE DOORS":
        wi = sum(1 for w, ts in ln["words"] if ts <= t)
        img = city(t, (0.3, 0.7, 0.5)[wi % 3], 0.55, 1.6 + 0.2 * wi, 0.004)
        img = fx.slices(img, rng, 10, 150)
    elif text == "KILL THE LIGHT":
        img = android(t, 0.49, 0.37, 2.0)
        redness = np.clip((img[..., 0] - np.maximum(img[..., 1], img[..., 2])) * 4 - 0.4, 0, 1)[..., None]
        img = img * redness * 1.6
        if lt < 0.1:
            img = np.ones((H, W, 3), np.float32)
    else:  # FEEL THE SYSTEM COME ALIVE TONIGHT
        p = min(lt / 3.4, 1)
        img = android(t, 0.5, 0.44, 1.0 + 0.4 * p, 0.001 + 0.004 * p)
        img = fx.grade(img, exposure=0.8 + 0.5 * p)
        img = terminal(img, ["> CORE TEMP ......... CRITICAL", "> EMOTION DAMPERS ... OFFLINE",
                             "> SYSTEM ............ ALIVE"], ln["start"], t, x=W - 820, y=170, color=RED, size=34)
        post.update(bloom=0.5 + 1.2 * p, hud="BOOT // SYSTEM")
    return img, post


def verse2(t, fi, ln, kick, snare, bass, rng, post):
    text = ln["text"] if ln else ""
    lt = t - ln["start"] if ln else 0.0
    if text == "BLACKENED SKY" or not text:
        img = city(t, 0.5, 0.42, 1.2 - 0.03 * lt)
        sky = np.clip(np.linspace(0, 1.3, H), 0, 1)[:, None, None].astype(np.float32)
        img = fx.red_only(img, 0.9) * (0.2 + 0.8 * sky)
    elif text == "DIGITAL SCARS":
        img = android(t, 0.55, 0.45, 1.45)
        r = np.random.default_rng(77)
        grow = min(lt / 1.2, 1)
        for k in range(14):                       # cicatrices : fines coulures triées verticalement
            x0 = int(r.uniform(820, 1420))
            wd = int(r.uniform(6, 34))
            y0 = int(r.uniform(80, 500))
            ln_ = int(r.uniform(200, 560) * grow) + 10
            img = fx.pixel_sort(img, y0, min(H, y0 + ln_), x0=x0, x1=x0 + wd, vertical=True)
    elif text.startswith("LOST, LOST SOULS"):
        img = fx.grade(city(t, 0.5, 0.5, 1.1 + 0.02 * lt), exposure=0.55)
        img = img + neon_stars(t) * 1.3 + lasers(t, 0.25)
    elif text.startswith("MACHINE WHISPERS"):
        img = ascii_face(t)
        post["bloom"] = 0.5
    elif text == "SYNTHETIC LOVE":
        img = fx.grade(android(t, 0.5, 0.45, 1.2 + 0.05 * lt, focus=0.9, dof=5), tint=CYAN, tint_amt=0.3,
                       contrast=0.95)
        post["bloom"] = 0.9
    elif text == "EMOTION DEAD":
        img = android(t, 0.5, 0.45, 1.28 + 0.05 * lt)
        img = fx.red_only(img, max(0.0, 1 - lt / 1.2))
        img = fx.grade(img, exposure=0.85)
    elif text == "SHARP LIGHTS CUT THROUGH THE HAZE":
        img = fx.grade(city(t, 0.5, 0.45, 1.3), exposure=0.5) + 0.12
        beams = lasers(t * 1.6, 0.9, n=5)
        img = img + beams.mean(2, keepdims=True) * np.array([1, 1, 1], np.float32)
    elif text == "AT ENDLESS NIGHTS":
        img = fx.grade(city(t, 0.5, 0.47, 1.6 + 0.1 * lt), exposure=0.55, sat=0.5)
    elif text == "TOXIC MAZE":
        img = fx.grade(city(t, 0.5, 0.47, 1.8), exposure=0.3) + maze(lt / 1.2) * 0.9
    elif text.startswith("CAN YOU FEEL IT"):
        img = fx.grade(android(t, 0.5, 0.47, 1.35), exposure=0.55, sat=0.4)
        m, glow = ecg(t, 150, amp=160)
        img = fx.paint(img, m, RED, 1.0) + glow[..., None] * RED * 0.8
    elif text == "SYSTEM OVERLOAD":
        img = fx.grade(android(t, 0.5, 0.45, 1.3 + 0.1 * kick, 0.006), exposure=1.6, contrast=2.2)
        img = fx.blocks(fx.rgb_split(img, 18), rng, 14)
        if (fi // 2) % 2:
            img = fx.invert(img)
        post["hud"] = "!! OVERLOAD !!"
    else:  # ERASE THE PAIN
        img = android(t, 0.5, 0.45, 1.3)
        wv = ramp(t, ln["start"], ln["start"] + 1.2) if ln else 1
        img = img * (1 - wv) + np.ones_like(img) * wv
        if t > 126.35:
            img = np.zeros_like(img)
    post.setdefault("rain", 0.0)
    if not text.startswith("MACHINE") and text not in ("SYSTEM OVERLOAD", "ERASE THE PAIN"):
        post["rain"] = 0.25
    return img, post


def fading(t, fi, ln, kick, snare, rng, post):
    text = ln["text"] if ln else ""
    bi = beat_index(t)
    if text == "NEON SHADOWS" or (151.9 < t < 153.5):
        img = fx.red_only(city(t, 0.5, 0.45, 1.3), 1.0)
        img = fx.grade(img, exposure=0.7, contrast=1.6)
    elif text == "TAKE CONTROL" or t >= 153.5:
        cx, cy, z = 0.49, 0.345, 2.6 + 0.3 * ramp(t, 153.5, 155)
        img = eye_reticles(fx.red_only(android(t, cx, cy, z, 0.003)), t, cx, cy, z)
        post["hud"] = "OVERRIDE ACCEPTED"
        post["bloom"] = 1.2
    else:
        kind = pick(["scream", "android_red", "city_wide", "scream_eyes"], bi // 2, 41)
        img = montage_shot(kind, t, bi // 2, fi, kick)
        img = fx.grade(img, sat=0.5)
        bpm = 140 - 70 * ramp(t, 145.5, 150)
        m, glow = ecg(t, bpm, amp=140, y=H * 0.72)
        img = fx.paint(img, m, RED, 1.0) + glow[..., None] * RED * 0.7
        post["static"] = 0.12 + 0.3 * ramp(t, 143.2, 147)
    return img, post


def outro(t, fi, post):
    p = (t - 178) / (DURATION - 178)
    img = android(t, 0.5, 0.45, 1.1 + 0.15 * p, name="scream" if t < 180.5 else "android")
    img = fx.grade(img, sat=1 - p, exposure=1 - 0.3 * p)
    post["static"] = 0.15 + 0.5 * ramp(t, 179, 182)
    # extinction façon écran cathodique
    if t > 182.3:
        q = ramp(t, 182.3, 182.9)
        h = max(2, int(H * (1 - q)))
        y0 = (H - h) // 2
        squeezed = np.asarray(Image.fromarray(fx.to_uint8(img)).resize((W, h), Image.BILINEAR), np.float32) / 255
        img = np.zeros_like(img)
        img[y0:y0 + h] = squeezed * (1 + 2 * q)
        post["static"] = 0.0
        if q >= 1:
            img[:] = 0
            wv = max(0.0, 1 - (t - 182.9) / 0.2)
            img[H // 2 - 2:H // 2 + 2] = WHITE * wv
    return img, post


# ---------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------

def frame(fi):
    t = fi / FPS
    img, post = base_frame(t, fi)
    if post.get("static"):
        img = fx.mix_static(img, fi, post["static"]) * (1 - post["static"] * (1 - fx.SCANLINES))
    if post.get("rain"):
        img = cine.rain3(img, t, post["rain"] * 0.8)
        img = cine.droplets(img, min(1.0, post["rain"] * 2.5))
    img = fx.bloom(np.clip(img, 0, 1.4), post.get("bloom", 0.45) * 0.6)
    if post.get("hud"):
        img = hud(img, t, fi, post["hud"])
    img = draw_lyrics(img, t, fi)
    # titre final
    if t > 183.0:
        a = ramp(t, 183.0, 183.4) * (1 - ramp(t, 185.0, 185.45))
        img = fx.glitch_text(img, fx.scramble("SYNTHETIC SCREAM", min(1, (t - 183) / 0.6), fi), BOLD, 130,
                             W / 2, H / 2, a, split=8, tracking=18)
    # carton-titre à l'entrée de la batterie
    if 16.3 <= t < 20.0:
        a = ramp(t, 16.3, 16.7) * (1 - ramp(t, 19.3, 20.0))
        img = fx.glitch_text(img, "SYNTHETIC SCREAM", MEDIUM, 92, W / 2, H / 2 - 10, a, split=3, tracking=34)
        wline = int(560 * ramp(t, 16.4, 17.2))
        img[H // 2 + 52:H // 2 + 55, W // 2 - wline:W // 2 + wline] = (
            img[H // 2 + 52:H // 2 + 55, W // 2 - wline:W // 2 + wline] * (1 - a) + RED * a)
    img = cine.optics(img, t, fi, bar=bar_height(t), flare=post.get("flare", 0.35))
    return fx.to_uint8(img)


def encode(frames, out):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
           "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in frames:
        p.stdin.write(frame(fi).tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    if sys.argv[1] == "still":
        Image.fromarray(frame(int(round(float(sys.argv[2]) * FPS)))).save(sys.argv[3])
    elif sys.argv[1] == "segment":
        encode(range(int(sys.argv[2]), min(int(sys.argv[3]), NFRAMES)), sys.argv[4])
