"""Couche « cinéma » : caméra 2,5D, profondeur de champ, optique et pellicule.

- camera()     : cadrage dans une image fixe avec parallaxe par profondeur
                 (Depth Anything V2), travelling avant « dolly » et mise au point.
- rain3()      : pluie sur trois plans (lointain, moyen, contre la caméra).
- droplets()   : gouttes sur l'objectif, qui réfractent l'image.
- optics()     : courbe film, reflets anamorphiques, halo, aberration
                 chromatique, flottement de pellicule, grain, bandes 2.39:1.
"""
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
W, H = 1920, 1080
_GX, _GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

# gamma par image : étale la profondeur là où c'est utile (la ville est écrasée par la chaussée)
DEPTH_GAMMA = {"city": 0.45, "android": 1.0, "scream": 1.0}


@lru_cache(None)
def src(name):
    im = Image.open(HERE / "assets" / f"{name}.png").convert("RGB")
    if im.width < 2600:
        im = im.resize((2688, 1520), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(im))


@lru_cache(None)
def depth(name):
    d = np.asarray(Image.open(HERE / "assets" / f"{name}_depth.png"), np.float32) / 65535
    ih, iw = src(name).shape[:2]
    d = cv2.resize(d, (iw, ih), interpolation=cv2.INTER_CUBIC)
    d = cv2.GaussianBlur(d, (0, 0), 5)                    # bords doux : pas de déchirures
    return np.clip(d, 0, 1) ** DEPTH_GAMMA[name]


def camera(name, cx=0.5, cy=0.5, zoom=1.0, rot=0.0, dx=0.0, dy=0.0,
           par=(0.0, 0.0), dolly=0.0, focus=None, dof=0.0):
    """Image 1920×1080 (float [0,1]) cadrée dans `name`.

    par   : décalage de parallaxe (fraction de largeur) appliqué au premier plan
            par rapport au fond ; animer `par` = la caméra tourne autour du sujet.
    dolly : le premier plan grossit plus vite que le fond (travelling avant réel).
    focus : profondeur nette (0 = loin, 1 = près) ; dof = flou maximal en pixels.
    """
    im = src(name)
    ih, iw = im.shape[:2]
    s = (iw / W) / zoom
    c, sn = math.cos(rot), math.sin(rot)
    ox = _GX - W / 2
    oy = _GY - H / 2
    mx = (c * ox - sn * oy) * s + (cx + dx) * iw
    my = (sn * ox + c * oy) * s + (cy + dy) * ih
    mx, my = mx.astype(np.float32), my.astype(np.float32)
    d = cv2.remap(depth(name), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rel = d - 0.5
    if par[0] or par[1]:
        mx = mx - rel * par[0] * iw
        my = my - rel * par[1] * ih
    if dolly:
        k = 1 - rel * dolly
        mx = (mx - (cx + dx) * iw) * k + (cx + dx) * iw
        my = (my - (cy + dy) * ih) * k + (cy + dy) * ih
    mx, my = mx.astype(np.float32), my.astype(np.float32)
    out = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
    if dof > 0.3 and focus is not None:
        blur = cv2.GaussianBlur(out, (0, 0), dof)
        w = np.clip(np.abs(d - focus) * 3.0, 0, 1)[..., None]
        out = out * (1 - w) + blur * w
    return out


# ---------------------------------------------------------------------------
# Pluie et gouttes
# ---------------------------------------------------------------------------

def _rain_layer(seed, n, width, length, alpha, blur):
    r = np.random.default_rng(seed)
    img = Image.new("L", (W + 200, H * 2), 0)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        x, y = r.uniform(0, W + 200), r.uniform(0, H * 2)
        ln = r.uniform(*length)
        d.line([(x, y), (x + ln * 0.12, y + ln)], fill=int(255 * r.uniform(*alpha)), width=width)
    a = np.asarray(img, np.float32) / 255
    if blur:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    a = a[:, 100:100 + W]
    return np.concatenate([a, a], 0)


RAIN_LAYERS = [
    # (texture, vitesse px/s, intensité)
    (_rain_layer(1, 4200, 1, (20, 50), (0.25, 0.6), 0), 1500, 0.55),
    (_rain_layer(2, 900, 2, (60, 130), (0.3, 0.7), 0.6), 2600, 0.6),
    (_rain_layer(3, 70, 6, (220, 420), (0.25, 0.5), 3.0), 4200, 0.5),
]


def rain3(img, t, amount=0.35, tint=(0.78, 0.9, 1.0)):
    tint = np.array(tint, np.float32)
    out = img
    for tex, speed, k in RAIN_LAYERS:
        off = int(t * speed) % (H * 2)
        out = out + tex[off:off + H][..., None] * tint * (amount * k)
    return out


def _droplet_maps(seed=9, n=46):
    r = np.random.default_rng(seed)
    mx, my = _GX.copy(), _GY.copy()
    alpha = np.zeros((H, W), np.float32)
    rim = np.zeros((H, W), np.float32)
    for _ in range(n):
        x, y, rad = r.uniform(0, W), r.uniform(0, H), r.uniform(6, 34)
        x0, x1 = int(max(x - rad - 2, 0)), int(min(x + rad + 2, W))
        y0, y1 = int(max(y - rad * 1.2 - 2, 0)), int(min(y + rad * 1.2 + 2, H))
        gx, gy = _GX[y0:y1, x0:x1], _GY[y0:y1, x0:x1]
        dist = np.sqrt(((gx - x) / rad) ** 2 + ((gy - y) / (rad * 1.2)) ** 2)
        inside = dist < 1
        # lentille : l'image est retournée et grossie à l'intérieur de la goutte
        mx[y0:y1, x0:x1] = np.where(inside, x - (gx - x) * 2.6, mx[y0:y1, x0:x1])
        my[y0:y1, x0:x1] = np.where(inside, y - (gy - y) * 2.6, my[y0:y1, x0:x1])
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], np.clip((1 - dist) * 6, 0, 1))
        rim[y0:y1, x0:x1] = np.maximum(rim[y0:y1, x0:x1], np.clip(1 - np.abs(dist - 0.85) * 8, 0, 1))
    return mx, my, alpha, rim


DROPS = _droplet_maps()


def droplets(img, amount=1.0):
    if amount <= 0.01:
        return img
    mx, my, a, rim = DROPS
    ref = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ref = cv2.GaussianBlur(ref, (0, 0), 1.2)
    a = (a * amount)[..., None]
    out = img * (1 - a) + ref * a * 1.1
    return out * (1 - (rim * 0.35 * amount)[..., None])


# ---------------------------------------------------------------------------
# Optique et pellicule
# ---------------------------------------------------------------------------

def _aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


_R = np.sqrt(((_GX - W / 2) / (W / 2)) ** 2 + ((_GY - H / 2) / (W / 2)) ** 2)
_BARREL = (1 + 0.022 * _R ** 2) / (1 + 0.022 * 1.32)      # compensé : pas de bords noirs dans les coins
_CA = {c: (W / 2 + (_GX - W / 2) * _BARREL * k, H / 2 + (_GY - H / 2) * _BARREL * k)
       for c, k in ((0, 0.9985), (1, 0.9965), (2, 0.9945))}
_VIG = np.clip(1.08 - 0.45 * _R ** 2.2, 0.3, 1)[..., None].astype(np.float32)
_GRAIN = [cv2.resize(np.random.default_rng(300 + i).normal(0, 1, (H // 2, W // 2)).astype(np.float32),
                     (W, H), interpolation=cv2.INTER_CUBIC) for i in range(8)]


def streaks(img, strength=0.35, threshold=0.75):
    """Reflets anamorphiques : les néons s'étirent horizontalement, teinte bleutée."""
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    lum = small.max(axis=2)
    b = np.clip(lum - threshold, 0, None)
    b = cv2.blur(b, (61, 1))
    b = cv2.blur(b, (121, 1))
    b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + b[..., None] * np.array([0.45, 0.65, 1.0], np.float32) * strength * 4


def halation(img, strength=0.25, threshold=0.6):
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(np.clip(small - threshold, 0, None), (0, 0), 6)
    b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + b.mean(axis=2, keepdims=True) * np.array([1.0, 0.3, 0.2], np.float32) * strength * 2


def bars_mask(bar):
    m = np.ones((H, 1, 1), np.float32)
    if bar > 0:
        m[:int(bar)] = 0
        m[H - int(bar):] = 0
    return m


def optics(img, t, fi, bar=138, flare=0.35, halo=0.25, grain=0.045, weave=True, tone=0.6):
    out = np.clip(img, 0, 4)
    out = streaks(out, flare)
    out = halation(out, halo)
    # courbe film (ACES approchée), mélangée pour garder du punch
    out = out * (1 - tone) + _aces(out * 0.95) * tone
    # teintes : ombres vers le bleu-vert, hautes lumières légèrement chaudes
    lum = out.mean(axis=2, keepdims=True)
    out = out + (1 - lum) ** 3 * np.array([-0.012, 0.004, 0.02], np.float32) \
        + lum ** 3 * np.array([0.02, 0.004, -0.015], np.float32)
    # aberration chromatique radiale, légère distorsion en barillet, flottement
    wx = 0.7 * math.sin(t * 7.1) + 0.4 * math.sin(t * 13.3) if weave else 0.0
    wy = 0.6 * math.sin(t * 5.3 + 1) + 0.3 * math.sin(t * 17.9) if weave else 0.0
    out = np.clip(out, 0, 1).astype(np.float32)
    chans = [cv2.remap(np.ascontiguousarray(out[..., c]), _CA[c][0] + wx, _CA[c][1] + wy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in range(3)]
    out = np.stack(chans, -1) * _VIG
    # grain de pellicule, plus visible dans les tons moyens
    g = _GRAIN[fi % len(_GRAIN)][..., None]
    lum = out.mean(axis=2, keepdims=True)
    out = out + g * grain * (0.35 + 1.3 * lum * (1 - lum))
    return out * bars_mask(bar)
