"""Boîte à outils visuelle du clip : caméra sur image fixe, étalonnage, glitchs,
pluie, parasites, halo, grain, texte.

Toutes les images manipulées sont des tableaux float32 (H, W, 3) dans [0, 1].
"""
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H = 1920, 1080

RED = np.array([1.0, 0.14, 0.22], np.float32)
CYAN = np.array([0.16, 0.9, 1.0], np.float32)
WHITE = np.array([0.95, 0.95, 0.97], np.float32)
BLACK = np.zeros(3, np.float32)

BOLD = str(HERE / "fonts/ChakraPetch-Bold.ttf")
MEDIUM = str(HERE / "fonts/ChakraPetch-Medium.ttf")
MONO = str(HERE / "fonts/ShareTechMono-Regular.ttf")


# ---------------------------------------------------------------------------
# Images sources et caméra
# ---------------------------------------------------------------------------

@lru_cache(None)
def source(name):
    im = Image.open(HERE / "assets" / f"{name}.png").convert("RGB")
    if im.width < 2600:                                   # la ville est en 1K : on l'agrandit une fois
        im = im.resize((2688, 1520), Image.LANCZOS)
    return im


def camera(name, cx=0.5, cy=0.5, zoom=1.0, rot=0.0, dx=0.0, dy=0.0):
    """Cadre 1920×1080 dans une image : centre (cx, cy) normalisé, zoom 1 = image entière."""
    im = source(name)
    iw, ih = im.size
    s = (iw / W) / zoom
    c, sn = math.cos(rot), math.sin(rot)
    a, b, d, e = c * s, -sn * s, sn * s, c * s
    x0 = cx * iw + dx * iw - (a * W / 2 + b * H / 2)
    y0 = cy * ih + dy * ih - (d * W / 2 + e * H / 2)
    out = im.transform((W, H), Image.AFFINE, (a, b, x0, d, e, y0), resample=Image.BILINEAR)
    return np.asarray(out, np.float32) / 255


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Étalonnage
# ---------------------------------------------------------------------------

def grade(img, exposure=1.0, contrast=1.0, sat=1.0, tint=None, tint_amt=0.0, lift=0.0):
    out = img * exposure
    if contrast != 1.0:
        out = (out - 0.5) * contrast + 0.5
    if sat != 1.0:
        lum = out @ np.array([0.299, 0.587, 0.114], np.float32)
        out = lum[..., None] + (out - lum[..., None]) * sat
    if tint is not None and tint_amt:
        lum = (out @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        out = out * (1 - tint_amt) + lum * tint * tint_amt * 1.6
    if lift:
        out = out + lift
    return np.clip(out, 0, 1.5)


def luminance(img):
    return img @ np.array([0.299, 0.587, 0.114], np.float32)


def red_only(img, keep=1.0):
    """Tout en noir et blanc sauf les rouges (yeux, néons rouges)."""
    lum = luminance(img)[..., None]
    redness = np.clip((img[..., 0] - np.maximum(img[..., 1], img[..., 2])) * 3, 0, 1)[..., None]
    return lum * (1 - redness * keep) + img * redness * keep


# ---------------------------------------------------------------------------
# Glitchs
# ---------------------------------------------------------------------------

def rgb_split(img, px):
    px = int(px)
    if px == 0:
        return img
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], px, axis=1)
    out[..., 2] = np.roll(img[..., 2], -px, axis=1)
    return out


def slices(img, rng, n=6, shift=80):
    out = img.copy()
    for _ in range(n):
        y = int(rng.integers(0, H - 10))
        h = int(rng.integers(4, 70))
        out[y:y + h] = np.roll(out[y:y + h], int(rng.integers(-shift, shift + 1)), axis=1)
    return out


def blocks(img, rng, n=12, size=(40, 220)):
    """Blocs déplacés façon compression cassée (datamosh du pauvre)."""
    out = img.copy()
    for _ in range(n):
        bw, bh = int(rng.integers(*size)), int(rng.integers(size[0] // 2, size[1] // 2))
        x, y = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        sx = int(np.clip(x + rng.integers(-120, 121), 0, W - bw))
        sy = int(np.clip(y + rng.integers(-40, 41), 0, H - bh))
        blk = img[sy:sy + bh, sx:sx + bw]
        if rng.random() < 0.3:
            blk = blk[..., [2, 0, 1]]
        out[y:y + bh, x:x + bw] = blk
    return out


def pixel_sort(img, y0, y1, x0=0, x1=W, vertical=False):
    """Trie les pixels d'une bande par luminance : traînées de « cicatrices numériques »."""
    out = img.copy()
    if vertical:
        reg = out[y0:y1, x0:x1]
        idx = np.argsort(luminance(reg), axis=0)
        out[y0:y1, x0:x1] = np.take_along_axis(reg, idx[..., None], axis=0)
    else:
        reg = out[y0:y1, x0:x1]
        idx = np.argsort(luminance(reg), axis=1)
        out[y0:y1, x0:x1] = np.take_along_axis(reg, idx[..., None], axis=1)
    return out


def mirror_shards(img, rng, n=7):
    """Bandes verticales décalées verticalement : l'image « se brise »."""
    out = img.copy()
    xs = np.sort(rng.integers(0, W, n))
    edges = [0, *xs, W]
    for a, b in zip(edges[:-1], edges[1:]):
        out[:, a:b] = np.roll(img[:, a:b], int(rng.integers(-240, 241)), axis=0)
        if rng.random() < 0.3:
            out[:, a:b] = out[:, a:b] * 1.4
    return out


def invert(img):
    return 1 - np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Matières : pluie, parasites, grain, lignes, halo
# ---------------------------------------------------------------------------

def _rain_texture(seed=3, drops=2600):
    r = np.random.default_rng(seed)
    img = Image.new("L", (W, H * 2), 0)
    d = ImageDraw.Draw(img)
    for _ in range(drops):
        x, y = r.uniform(-200, W), r.uniform(0, H * 2)
        ln = r.uniform(30, 110)
        d.line([(x, y), (x + ln * 0.18, y + ln)], fill=int(r.uniform(60, 200)), width=1)
    arr = np.asarray(img, np.float32) / 255
    return np.concatenate([arr, arr], 0)          # 4H pour un défilement sans couture


RAIN = _rain_texture()


def rain(img, t, amount=0.35, speed=2400, tint=(0.75, 0.9, 1.0)):
    off = int(t * speed) % (H * 2)
    layer = RAIN[off:off + H][..., None] * np.array(tint, np.float32)
    return img + layer * amount


_NOISE = [np.random.default_rng(50 + i).random((H // 2, W // 2), dtype=np.float32) for i in range(8)]


def static(fi, amount=1.0, rng=None):
    """Neige télé (demi-résolution agrandie) avec lignes plus claires."""
    n = _NOISE[fi % len(_NOISE)]
    n = np.roll(n, (fi * 37) % (H // 2), axis=0)
    n = n.repeat(2, 0).repeat(2, 1)
    rows = (np.sin(np.arange(H) * 0.05 + fi * 1.3) > 0.97).astype(np.float32)[:, None] * 0.4
    return np.clip(n * 0.85 + rows, 0, 1)[..., None] * np.ones(3, np.float32)


def mix_static(img, fi, amount):
    if amount <= 0.001:
        return img
    return img * (1 - amount) + static(fi) * amount


_GRAIN = [np.random.default_rng(90 + i).normal(0, 1, (H, W, 1)).astype(np.float32) for i in range(4)]
_Y = np.arange(H)[:, None]
SCANLINES = (1 - 0.13 * ((_Y % 3) == 0)).astype(np.float32)[..., None]
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = np.clip(1.15 - 0.55 * (((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2), 0.25, 1)[..., None]


def finish(img, fi, grain=0.035, scan=True):
    out = img * VIGNETTE
    if scan:
        out = out * SCANLINES
    return out + _GRAIN[fi % 4] * grain


def _box(img, r):
    for ax in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(img, pad, mode="edge"), axis=ax)
        n = img.shape[ax]
        img = (np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, range(0, n), axis=ax)) / (2 * r + 1)
    return img


def bloom(img, strength=0.6, threshold=0.62):
    small = img.reshape(H // 4, 4, W // 4, 4, 3).mean(axis=(1, 3))
    bright = np.clip(small - threshold, 0, None)
    bright = _box(_box(bright, 5), 5)
    up = bright.repeat(4, 0).repeat(4, 1)
    return img + up * strength


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------

@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(48)                  # masques plein écran de 8 Mo : cache court, sinon la mémoire explose
def text_mask(text, path, size, anchor_x, anchor_y, anchor="mm", tracking=0):
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    f = font(path, size)
    if tracking:
        widths = [f.getlength(c) + tracking for c in text]
        total = sum(widths) - tracking
        x = anchor_x - total / 2 if anchor[0] == "m" else (anchor_x - total if anchor[0] == "r" else anchor_x)
        for c, w_ in zip(text, widths):
            d.text((x, anchor_y), c, font=f, fill=255, anchor="l" + anchor[1])
            x += w_
    else:
        d.text((anchor_x, anchor_y), text, font=f, fill=255, anchor=anchor)
    return np.asarray(img, np.float32) / 255


def paint(img, mask, color, alpha=1.0):
    m = mask[..., None] * alpha
    return img * (1 - m) + color * m


def glitch_text(img, text, path, size, x, y, alpha=1.0, split=6, color=WHITE, anchor="mm", tracking=0):
    """Texte blanc avec fantômes rouge et cyan décalés."""
    if alpha <= 0.01 or not text:
        return img
    m = text_mask(text, path, size, int(x), int(y), anchor, tracking)
    if split:
        img = paint(img, np.roll(m, -int(split), axis=1), RED, alpha * 0.85)
        img = paint(img, np.roll(m, int(split), axis=1), CYAN, alpha * 0.85)
    return paint(img, m, color, alpha)


GLYPHS = "01<>/\\#%&$@*+=_[]{}|ABCDEFXYZ"


def scramble(text, progress, seed):
    """Révélation « décodage » : les caractères pas encore fixés défilent au hasard."""
    r = np.random.default_rng(seed)
    n = len(text)
    k = int(progress * n)
    out = []
    for i, ch in enumerate(text):
        if i < k or ch == " ":
            out.append(ch)
        elif i < k + 6:
            out.append(GLYPHS[r.integers(len(GLYPHS))])
        else:
            out.append(" ")
    return "".join(out)
