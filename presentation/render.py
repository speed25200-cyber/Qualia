"""Rendu image par image de la vidéo « Claude, en une minute ».

Tout est fonction du temps t (aucune simulation incrémentale) : chaque image
peut donc être calculée indépendamment, ce qui permet de répartir le rendu sur
plusieurs processus.

Usage :
    python3 render.py still 12.5 out.png      # une image
    python3 render.py segment 0 450 seg0.mp4  # images [0, 450[ encodées en H.264
"""
import json
import math
import re
import subprocess
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).parent
BUILD = HERE / "build"
FONTS = HERE / "fonts"

W, H, FPS = 1920, 1080, 30
DURATION = 60.0
N = 2600                      # nombre de particules-lettres
CX, CY = 960, 450             # centre de la scène (au-dessus des sous-titres)

IVORY = np.array([242, 237, 228], np.float32) / 255
TERRA = np.array([222, 124, 90], np.float32) / 255
DIM = np.array([150, 141, 128], np.float32) / 255

SERIF, SERIF_IT = "Fraunces-Light.ttf", "Fraunces-Italic-Light.ttf"
SANS, MONO = "Inter-Regular.ttf", "JetBrainsMono-Regular.ttf"
SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

# ---------------------------------------------------------------------------
# Temps : tout est calé sur les mots de la voix off
# ---------------------------------------------------------------------------

LINES = json.loads((BUILD / "words.json").read_text())
BY_ID = {l["id"]: l for l in LINES}


def norm(w):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", w.lower()))


def wt(line, word=None, end=False):
    """Instant (s) où un mot d'une réplique commence (ou finit)."""
    l = BY_ID[line]
    if word is None:
        return l["end"] if end else l["start"]
    w = next(w for w in l["words"] if norm(w["w"]) == norm(word))
    return w["end"] if end else w["start"]


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease(x):
    x = clamp01(x)
    return np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)


def ramp(t, a, b):
    return float(smooth((t - a) / (b - a)))


def window(t, a, b, fade_in=0.3, fade_out=0.3):
    return ramp(t, a, a + fade_in) * (1 - ramp(t, b - fade_out, b))


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------

@lru_cache(None)
def font(name, size):
    path = name if name.startswith("/") else str(FONTS / name)
    return ImageFont.truetype(path, size)


@lru_cache(None)
def text_mask(text, fname, size):
    """Masque alpha (float32) d'un texte + décalage du coin haut-gauche par rapport à l'origine (gauche, ligne de base)."""
    f = font(fname, size)
    asc, desc = f.getmetrics()
    pad = size // 3 + 4
    w = int(math.ceil(f.getlength(text))) + 2 * pad
    h = asc + desc + 2 * pad
    img = Image.new("L", (max(w, 1), h), 0)
    # pas de ligatures (« === » doit rester lisible dans le code)
    feats = ["-liga", "-calt"] if fname == MONO else None
    ImageDraw.Draw(img).text((pad, pad + asc), text, font=f, fill=255, anchor="ls", features=feats)
    return np.asarray(img, np.float32) / 255, -pad, -(pad + asc)


def text_width(text, fname, size):
    return font(fname, size).getlength(text)


def over(canvas, mask, x, y, color, alpha):
    """Compose un masque (couleur, opacité) sur le canevas ; (x, y) = coin haut-gauche."""
    if alpha <= 0.002:
        return
    x, y = int(round(x)), int(round(y))
    h, w = mask.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    region = canvas[y0:y1, x0:x1]
    region *= 1 - m
    region += m * color


def draw_text(canvas, text, fname, size, x, baseline, color, alpha):
    m, ox, oy = text_mask(text, fname, size)
    over(canvas, m, x + ox, baseline + oy, color, alpha)


def draw_rect(canvas, x0, y0, x1, y1, color, alpha):
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if alpha <= 0.002 or x0 >= x1 or y0 >= y1:
        return
    region = canvas[y0:y1, x0:x1]
    region *= 1 - alpha
    region += alpha * color


class Typed:
    """Bloc de texte tapé au clavier, synchronisé sur les mots d'une réplique.

    rows : liste de lignes ; chaque ligne est une liste de segments (texte, police, couleur).
    """

    def __init__(self, rows, size, line_id, cy, leading=1.25):
        self.size = size
        widths = [sum(text_width(t, f, size) for t, f, _ in row) for row in rows]
        left = CX - max(widths) / 2
        n = len(rows)
        self.runs = []                      # (texte, police, couleur, x, ligne de base)
        for i, row in enumerate(rows):
            base = cy + (i - (n - 1) / 2) * size * leading + size * 0.33
            x = left
            for t, f, c in row:
                self.runs.append((t, f, c, x, base))
                x += text_width(t, f, size)
        self.left = left
        self.first_base = self.runs[0][4]
        # instants de frappe : les caractères d'un mot apparaissent pendant qu'il est prononcé
        full = "".join(r[0] for r in self.runs)
        words = BY_ID[line_id]["words"]
        knots_t, knots_k, k = [0.0], [0], 0
        for w in words:
            token = w["w"].replace(" ", " ")
            j = full.find(token.split()[0], k)
            if j < 0:
                continue
            end = j + len(token)
            while end < len(full) and full[end] in " ":
                end += 1
            knots_t += [w["start"], max(w["end"], w["start"] + 0.12)]
            knots_k += [j, end]
            k = end
        self.knots_t, self.knots_k = np.array(knots_t), np.array(knots_k, float)
        self.total = len(full)
        self.key_times = []                 # un « clic » par caractère (pour le son)
        for c in range(self.total):
            self.key_times.append(float(np.interp(c + 0.5, self.knots_k, self.knots_t)))

    def count(self, t):
        return int(np.interp(t, self.knots_t, self.knots_k))

    def mask(self):
        """Masque plein cadre du texte complet (pour y placer des particules)."""
        m = np.zeros((H, W), np.float32)
        tmp = np.zeros((H, W, 3), np.float32)
        for t, f, c, x, base in self.runs:
            draw_text(tmp, t, f, self.size, x, base, np.ones(3, np.float32), 1.0)
        m = tmp[..., 0]
        return m

    def draw(self, canvas, t, alpha=1.0, cursor=True, cursor_alpha=None):
        k = self.count(t)
        cur = (self.left, self.first_base)
        shown = 0
        for text, f, c, x, base in self.runs:
            if shown >= k:
                break
            part = text[: k - shown]
            if part.strip():
                draw_text(canvas, part, f, self.size, x, base, c, alpha)
            cur = (x + text_width(part, f, self.size), base)
            shown += len(text)
        if cursor:
            typing = np.any((t > self.knots_t - 0.05) & (t < self.knots_t + 0.35)) and 0 < k < self.total
            blink = 1.0 if typing else 0.5 + 0.5 * math.cos(2 * math.pi * t * 0.9)
            blink = min(1, max(0, (blink - 0.2) * 1.6))
            ca = alpha * blink * (1 if cursor_alpha is None else cursor_alpha)
            s = self.size
            draw_rect(canvas, cur[0] + 6, cur[1] - 0.74 * s, cur[0] + 12, cur[1] + 0.08 * s, TERRA, ca)
        return cur

    def cursor_home(self):
        return self.left + 9, self.first_base - 0.33 * self.size


def draw_caption(canvas, idx, t):
    """Sous-titres façon karaoké : les mots s'allument quand ils sont prononcés."""
    line = LINES[idx]
    if line["id"] in ("hello", "start"):
        return
    nxt = LINES[idx + 1]["start"] if idx + 1 < len(LINES) else DURATION
    a0, a1 = line["start"] - 0.3, min(nxt - 0.12, line["end"] + 1.4)
    a = window(t, a0, a1, 0.3, 0.25)
    if a <= 0:
        return
    size, maxw, lh = 36, 1320, 52
    space = text_width(" ", SANS, size)
    rows, row, w = [], [], 0.0
    for wd in line["words"]:
        ww = text_width(wd["w"], SANS, size)
        if row and w + space + ww > maxw:
            rows.append((row, w))
            row, w = [], 0.0
        row.append((wd, ww))
        w += (space if len(row) > 1 else 0) + ww
    rows.append((row, w))
    base0 = 1002 - (len(rows) - 1) * lh
    for r, (row, rw) in enumerate(rows):
        x = CX - rw / 2
        for wd, ww in row:
            lit = ramp(t, wd["start"] - 0.06, wd["start"] + 0.14)
            col = DIM * (1 - lit) + IVORY * lit
            draw_text(canvas, wd["w"], SANS, size, x, base0 + r * lh, col, a * (0.42 + 0.53 * lit))
            x += ww + space


# ---------------------------------------------------------------------------
# Lettres-particules
# ---------------------------------------------------------------------------

GLYPHS = (
    "abcdefghijklmnopqrstuvwxyzéèàçêôùABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "{}()<>;=+*/#&@?!%$"
    "αβγδεζηθλμξπσφψωΩΣΔ"
    "абвгджзиклмнпрстуфхцчшщыэюяЖЯ"
    "אבגדהוזחטיכלמנסעפצקרשת"
    "कखगघचजटडतदनपबमयरलवशसह"
    "的一是不了人我在有他这中大来上国字言光夢愛心"
)
SIZES = [12, 16, 23]
SUB = 4                          # positions sous-pixel par axe


@lru_cache(None)
def cmap(path):
    return TTFont(path, fontNumber=0)["cmap"].getBestCmap()


def font_for(ch):
    for name in (SANS, *SYSTEM_FONTS):
        path = name if name.startswith("/") else str(FONTS / name)
        if ord(ch) in cmap(path):
            return name
    return None


def make_sprites():
    """Pour chaque glyphe et taille : SUB×SUB versions décalées d'une fraction de pixel."""
    chars = [c for c in dict.fromkeys(GLYPHS) if font_for(c)]
    sprites = {}
    for gi, ch in enumerate(chars):
        fname = font_for(ch)
        for si, size in enumerate(SIZES):
            f = font(fname, size * SUB)
            box = size * 2                      # sprite de box×box pixels centré sur la particule
            for dy in range(SUB):
                for dx in range(SUB):
                    big = Image.new("L", (box * SUB, box * SUB), 0)
                    ImageDraw.Draw(big).text((box * SUB / 2 + dx, box * SUB / 2 + dy), ch, font=f,
                                             fill=255, anchor="mm")
                    s = (np.asarray(big, np.float32) / 255).reshape(box, SUB, box, SUB).mean(axis=(1, 3))
                    ys, xs = np.nonzero(s > 0.004)
                    if len(ys) == 0:
                        s, oy, ox = np.zeros((1, 1), np.float32), 0, 0
                    else:
                        oy, ox = ys.min(), xs.min()
                        s = s[oy:ys.max() + 1, ox:xs.max() + 1]
                    sprites[gi, si, dx, dy] = (np.ascontiguousarray(s), ox - box // 2, oy - box // 2)
    return chars, sprites


CHARS, SPRITES = make_sprites()
G = len(CHARS)

rng = np.random.default_rng(20260927)
P_CHAR = rng.integers(0, G, N)
P_SIZE = rng.choice(3, N, p=[0.70, 0.25, 0.05])
P_BRIGHT = rng.uniform(0.5, 1.0, N)
P_ACCENT = (rng.random(N) < 0.11).astype(np.float32)
P_R = rng.random(N)
P_ARC = rng.normal(0, 1, (N, 2))
P_W = rng.uniform(0.5, 1.5, (N, 2))
P_PH = rng.uniform(0, 2 * np.pi, (N, 2))
P_FLICK = np.where(rng.random(N) < 0.25, rng.uniform(0.3, 1.2, N), 0.0)
P_THETA = rng.uniform(0, 2 * np.pi, N)
P_RAD = rng.normal(0, 1, N)


def sample_mask(mask, n, seed):
    r = np.random.default_rng(seed)
    ys, xs = np.nonzero(mask > 0.5)
    idx = r.choice(len(ys), n, replace=len(ys) < n)
    return np.stack([xs[idx] + r.random(n), ys[idx] + r.random(n)], 1).astype(np.float32)


def glyph_mask(ch, fname, size, cx, cy):
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).text((cx, cy), ch, font=font(fname, size), fill=255, anchor="mm")
    return np.asarray(img, np.float32) / 255


# --- blocs de texte principaux -------------------------------------------------

HELLO = Typed([[("Bonjour.", SERIF, IVORY)],
               [("Je suis ", SERIF, IVORY), ("Claude.", SERIF_IT, TERRA)]],
              130, "hello", 430)
ENDING = Typed([[("Alors…", SERIF, IVORY)],
                [("par quoi on ", SERIF, IVORY), ("commence ?", SERIF_IT, TERRA)]],
               116, "start", 430)

CODE = [
    "function moyenne(notes) {",
    "  const n = notes.length;",
    "  if (n = 0) return null;",
    "  let somme = 0;",
    "  for (const x of notes) somme += x;",
    "  return somme / n;",
    "}",
]
CODE_SIZE, CODE_LH = 34, 52
CODE_LEFT = CX - max(text_width(l, MONO, CODE_SIZE) for l in CODE) / 2
CODE_BASE0 = CY - (len(CODE) - 1) / 2 * CODE_LH + 12


def code_mask():
    tmp = np.zeros((H, W, 3), np.float32)
    for i, l in enumerate(CODE):
        draw_text(tmp, l, MONO, CODE_SIZE, CODE_LEFT, CODE_BASE0 + i * CODE_LH, np.ones(3, np.float32), 1.0)
    return tmp[..., 0]


# --- formations ---------------------------------------------------------------
# Une formation renvoie (positions (N,2), alpha (N,), chaleur (N,), agitation px).

HELLO_PTS = sample_mask(HELLO.mask(), N, 1)
Q_PTS = sample_mask(glyph_mask("?", SERIF, 640, CX, CY + 10), N, 2)


def heart_pts():
    s = np.linspace(0, 2 * np.pi, 400)
    x = 16 * np.sin(s) ** 3
    y = -(13 * np.cos(s) - 5 * np.cos(2 * s) - 2 * np.cos(3 * s) - np.cos(4 * s))
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon(list(zip(CX + x * 15, CY + 10 + y * 15)), fill=255)
    pts = sample_mask(np.asarray(img, np.float32) / 255, N, 3)
    # appariement optimal « ? » -> cœur, pour une métamorphose lisible
    cost = ((Q_PTS[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(cost)
    return pts[col]


HEART_PTS = heart_pts()

# code : les particules suivent le fil emmêlé dans l'ordre, puis la lecture ligne par ligne
_code = sample_mask(code_mask(), N, 4)
_line = np.round((_code[:, 1] - (CODE_BASE0 - CODE_SIZE * 0.35)) / CODE_LH)
CODE_PTS = _code[np.lexsort((_code[:, 0], _line))]
S_ORDER = np.argsort(P_R)                     # particule -> rang le long du fil
S_PARAM = np.empty(N)
S_PARAM[S_ORDER] = np.linspace(0, 1, N)
CODE_FOR = np.empty((N, 2), np.float32)
CODE_FOR[S_ORDER] = CODE_PTS

FIB = None


def fib_sphere():
    i = np.arange(N) + 0.5
    phi = np.arccos(1 - 2 * i / N)
    th = np.pi * (1 + 5 ** 0.5) * i
    v = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    return v[rng.permutation(N)]


FIB = fib_sphere()


def const(v):
    return np.full(N, v, np.float32)


def f_hello(alpha):
    return lambda t: (HELLO_PTS, const(alpha), const(0), 0.0)


def f_sphere(R=270, c=(CX, CY), warm=0.0, alpha=1.0, spin=0.28, jit=1.0):
    def f(t):
        a, tilt = spin * t, 0.38
        x, y, z = FIB[:, 0], FIB[:, 1], FIB[:, 2]
        x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
        y, z = y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt)
        r = R * (1 + 0.012 * math.sin(1.3 * t))
        persp = 1 / (1 - 0.18 * z)
        pos = np.stack([c[0] + r * x * persp, c[1] + r * y * persp], 1)
        al = alpha * (0.12 + 0.88 * ((z + 1) / 2) ** 1.4)
        return pos, al.astype(np.float32), const(warm), jit
    return f


def f_ring(R=265, alpha=0.95, spread=9, spin=0.12, warm=0.0):
    def f(t):
        th = P_THETA + spin * t
        r = R + P_RAD * spread
        return np.stack([CX + r * np.cos(th), CY + r * np.sin(th)], 1), const(alpha), const(warm), 1.2
    return f


def f_burst(t):
    th = P_THETA + 0.12 * t
    r = 620 + 260 * np.abs(P_RAD)
    return np.stack([CX + r * np.cos(th), CY + r * np.sin(th)], 1), const(0), const(0), 2.0


def f_center(t):
    return np.stack([CX + 6 * P_ARC[:, 0], CY + 6 * P_ARC[:, 1]], 1), const(0), const(0), 0.0


def f_question(jit=2.0, warm=0.0):
    return lambda t: (Q_PTS, const(0.95), const(warm), jit)


def f_heartish(k=0.62):
    return lambda t: (Q_PTS * (1 - k) + HEART_PTS * k, const(0.95), const(0.55), 2.0)


def f_tangle(t):
    s = S_PARAM * 2 * np.pi
    x = (0.55 * np.sin(3 * s + 0.4 + 0.25 * t) + 0.3 * np.sin(7 * s + 1.1) + 0.2 * np.cos(11 * s - 0.3 * t)
         + 0.12 * np.sin(17 * s + 2))
    y = (0.5 * np.cos(2 * s + 0.2) + 0.33 * np.sin(5 * s - 0.2 * t) + 0.22 * np.cos(13 * s + 0.7)
         + 0.1 * np.sin(19 * s + 0.3 * t))
    pos = np.stack([CX + 470 * x + 4 * P_ARC[:, 0], CY + 250 * y + 4 * P_ARC[:, 1]], 1)
    return pos, const(0.9), const(0.12), 1.5


def f_code(alpha):
    return lambda t: (CODE_FOR, const(alpha), const(0.05), 0.4)


LEFT_HALF = P_R < 0.5
CL_Y = 660


def f_two(gap, sigma, alpha, jit):
    def f(t):
        c = np.where(LEFT_HALF, CX - gap, CX + gap)
        d = sigma * (1 + 0.08 * math.sin(2.1 * t))
        pos = np.stack([c + d * P_ARC[:, 0] * 0.9, CL_Y + d * P_ARC[:, 1] * 0.75], 1)
        warm = np.where(LEFT_HALF, 0.0, 1.0).astype(np.float32)
        return pos, const(alpha), warm, jit
    return f


def f_cursor(t):
    x, y = ENDING.cursor_home()
    return np.stack([x + 3 * P_ARC[:, 0], y + 3 * P_ARC[:, 1]], 1), const(0.0), const(1.0), 0.0


# --- chorégraphie -------------------------------------------------------------
# (instant, durée, formation, étalement, arc) ; l'étalement décale chaque particule
# de P_R × étalement, l'arc courbe les trajectoires.

T_WORDS = wt("face", "mots")
KEYS = [
    (0.0, 0.1, f_hello(0.0), 0, 0),
    (T_WORDS - 0.05, 0.35, f_hello(1.0), 0.15, 0),
    (T_WORDS + 0.45, 2.4, f_sphere(), 0.9, 260),
    (wt("you", "tout"), 1.4, f_sphere(warm=0.4, alpha=1.0, R=285), 0.4, 0),
    (wt("strange", "est") - 0.2, 1.4, f_ring(), 0.5, 90),
    (wt("strange", "chaque") - 0.1, 0.55, f_burst, 0.15, 40),
    (wt("strange", "chaque") + 0.62, 0.05, f_center, 0, 0),
    (wt("strange", "commence") - 0.1, 1.0, f_ring(alpha=1.0), 0.3, 50),
    (wt("present", "parle") - 0.1, 1.5, f_sphere(R=215, warm=0.55, spin=0.55, jit=0.2), 0.45, 70),
    (wt("feel", "est-ce") + 0.05, 1.5, f_question(), 0.6, 160),
    (wt("feel", "honnetement"), 1.2, f_question(jit=4.0), 0.2, 0),
    (wt("honest", "prefere") - 0.1, 1.2, f_heartish(), 0.3, 30),
    (wt("honest", "plutot") + 0.05, 0.6, f_question(jit=2.5), 0.15, 0),
    (wt("like", "revanche") - 0.1, 1.3, f_tangle, 0.5, 120),
    (wt("like", "demeler"), 1.15, f_code(0.95), 0.35, 30),
    (wt("like", "trouver") - 0.15, 0.4, f_code(0.0), 0.2, 0),
    (wt("together", "surtout") - 0.1, 1.1, f_two(330, 95, 0.5, 14.0), 0.5, 0),
    (wt("together", "plus"), 0.9, f_two(150, 70, 0.8, 5.0), 0.3, 0),
    (wt("together", "a") - 0.05, 0.7, f_sphere(R=110, c=(CX, CL_Y), warm=0.5, alpha=0.55, spin=0.8, jit=0.0), 0.2, 30),
    (wt("start", "alors") - 0.35, 0.75, f_cursor, 0.25, 60),
]
KEY_T = [k[0] for k in KEYS]


def state(t, k):
    t0, dur, form, stagger, arc = KEYS[k]
    pos, al, warm, jit = form(t)
    if k == 0:
        return pos, al, warm, jit
    u = ease((t - t0 - stagger * P_R) / dur)
    if np.all(u >= 1):
        return pos, al, warm, jit
    pa, aa, wa, ja = state(t, k - 1)
    u2 = u[:, None]
    bend = (np.sin(np.pi * u) * arc)[:, None] * P_ARC
    um = float(u.mean())
    return (pa * (1 - u2) + pos * u2 + bend, aa * (1 - u) + al * u,
            wa * (1 - u) + warm * u, ja * (1 - um) + jit * um)


def particles(t):
    k = max(0, np.searchsorted(KEY_T, t, side="right") - 1)
    pos, al, warm, jit = state(t, k)
    wob = jit * np.sin(P_W * t + P_PH)
    return pos + wob, al * P_BRIGHT, np.maximum(warm, P_ACCENT)


def draw_particles(layer, t):
    pos, al, warm = particles(t)
    chars = (P_CHAR + np.floor(t * P_FLICK + P_PH[:, 0]).astype(int) * (P_FLICK > 0)) % G
    cols = IVORY[None] * (1 - warm[:, None]) + TERRA[None] * warm[:, None]
    for i in np.nonzero(al > 0.01)[0]:
        x, y = pos[i]
        if not (-30 < x < W + 30 and -30 < y < H + 30):
            continue
        ix, iy = math.floor(x), math.floor(y)
        sx, sy = int((x - ix) * SUB), int((y - iy) * SUB)
        spr, ox, oy = SPRITES[chars[i], P_SIZE[i], sx, sy]
        x0, y0 = ix + int(ox), iy + int(oy)
        h, w = spr.shape
        a0, b0, a1, b1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
        if a0 >= a1 or b0 >= b1:
            continue
        layer[b0:b1, a0:a1] += spr[b0 - y0:b1 - y0, a0 - x0:a1 - x0, None] * (cols[i] * al[i])


# ---------------------------------------------------------------------------
# Scène C : ce que les humains ont écrit, absorbé par la sphère
# ---------------------------------------------------------------------------

STREAM_WORDS = (
    "amour maison pourquoi demain merci lumière water dream because Hoffnung Sehnsucht saudade "
    "ciao hola gracias libertad любовь слово время 夢 言葉 光 愛 φως λόγος शब्द नमस्ते שלום "
    "def return null {…} 42 π E=mc² if café pain bonjour Liebe silence océan histoire justice "
    "enfance mémoire question raison rire espoir ville étoile poésie sel chemin musique vérité "
    "lettre promesse ami pluie hiver jardin minuit cœur why maybe ¿qué? حب kitab عالم"
).split()


def word_font(w):
    for name in (SANS, *SYSTEM_FONTS):
        path = name if name.startswith("/") else str(FONTS / name)
        if all(ord(c) in cmap(path) for c in w):
            return name
    return SYSTEM_FONTS[0]


def make_stream():
    r = np.random.default_rng(11)
    t0, t1 = wt("learned", "j'ai") - 0.1, wt("learned", "recettes", end=True) + 0.2
    events, t = [], t0
    while t < t1:
        p = (t - t0) / (t1 - t0)
        t += 1 / (2.5 + 11 * p)
        w = STREAM_WORDS[r.integers(len(STREAM_WORDS))]
        events.append(dict(t=t, w=w, f=word_font(w), th=r.uniform(0, 2 * np.pi), dur=r.uniform(1.8, 2.6),
                           size=int(r.uniform(22, 34)), swirl=r.uniform(0.5, 1.1) * r.choice([-1, 1])))
    return events


STREAM = make_stream()

FRAGMENTS = [
    ("lettres", "Chère Louise, je t'écris enfin…", SERIF_IT, (430, 250)),
    ("code", "while (!compris) { relire(); }", MONO, (1490, 300)),
    ("poemes", "Demain, dès l'aube…", SERIF_IT, (420, 650)),
    ("disputes", "— Non, c'est toi qui as tort !", SANS, (1500, 640)),
    ("recettes", "Laisser reposer la pâte une heure.", SERIF, (1480, 180)),
]


def draw_stream(canvas, t):
    for e in STREAM:
        u = (t - e["t"]) / e["dur"]
        if not 0 <= u < 1:
            continue
        v = u ** 1.6
        th = e["th"] + e["swirl"] * v
        rad = 1 - v
        x = CX + math.cos(th) * (1150 * rad + 120 * (1 - rad))
        y = CY + math.sin(th) * (640 * rad + 80 * (1 - rad))
        size = max(8, int(e["size"] * (1 - 0.65 * v)))
        a = 0.5 * ramp(u, 0, 0.15) * (1 - ramp(u, 0.65, 1)) * (1 - ramp(y, 800, 870))
        m, ox, oy = text_mask(e["w"], e["f"], size)
        over(canvas, m, x - m.shape[1] / 2, y - m.shape[0] / 2, IVORY, a)
    for key, text, fname, (ax, ay) in FRAGMENTS:
        t0 = wt("learned", key)
        u = (t - (t0 + 1.15)) / 0.9           # vol vers la sphère
        if t < t0 - 0.1 or u >= 1:
            continue
        v = float(smooth(max(u, 0)))
        a = ramp(t, t0 - 0.1, t0 + 0.25) * (1 - ramp(u, 0.45, 1))
        size = max(8, int(36 * (1 - 0.7 * v)))
        x, y = ax + (CX - ax) * v, ay + (CY - ay) * v
        col = IVORY * (1 - v) + TERRA * v
        m, ox, oy = text_mask(text, fname, size)
        over(canvas, m, x - m.shape[1] / 2, y - m.shape[0] / 2 - 0 * oy, col, a * 0.92)


# ---------------------------------------------------------------------------
# Scène F/G : le bug, puis le mot juste
# ---------------------------------------------------------------------------

def draw_code(canvas, t):
    t_bug = wt("like", "bug")
    t_fix = t_bug + 0.28
    a = ramp(t, wt("like", "trouver") - 0.2, wt("like", "trouver") + 0.25) * \
        (1 - ramp(t, wt("like", "chercher") - 0.2, wt("like", "chercher") + 0.15))
    if a <= 0:
        return
    for i, line in enumerate(CODE):
        base = CODE_BASE0 + i * CODE_LH
        if i == 2:
            prefix = "  if ("
            x0 = CODE_LEFT + text_width(prefix, MONO, CODE_SIZE)
            fixed = ramp(t, t_fix, t_fix + 0.18)
            bad, good = "n = 0", "n === 0"
            wtok = text_width(bad if fixed < 0.5 else good, MONO, CODE_SIZE)
            hl = ramp(t, t_bug - 0.1, t_bug + 0.1) * (1 - 0.7 * fixed)
            draw_rect(canvas, x0 - 6, base - 32, x0 + wtok + 6, base + 10, TERRA, a * 0.3 * hl)
            draw_rect(canvas, x0 - 6, base + 7, x0 + wtok + 6, base + 10, TERRA, a * hl * (1 - fixed))
            line = prefix + (bad if fixed < 0.5 else good) + ") return null;"
        col = IVORY if i != 2 else IVORY * (1 - 0.15) + TERRA * 0.15
        draw_text(canvas, line, MONO, CODE_SIZE, CODE_LEFT, base, col, a * 0.92)


SLOT = ["nette", "simple", "limpide", "claire"]


def draw_sentence(canvas, t):
    t0 = wt("like", "chercher")
    a = ramp(t, t0, t0 + 0.3) * (1 - ramp(t, wt("start", "alors") - 0.4, wt("start", "alors") + 0.1))
    if a <= 0:
        return
    size, base = 64, 300
    prefix = "Voir une idée devenir plus "
    total = text_width(prefix, SERIF, size) + text_width("claire.", SERIF_IT, size)
    x = CX - total / 2
    draw_text(canvas, prefix, SERIF, size, x, base, IVORY, a)
    xs = x + text_width(prefix, SERIF, size)
    step = (wt("together", "et") - 0.35 - (t0 + 0.2)) / len(SLOT)
    step = min(step, 0.26)
    for i, w in enumerate(SLOT):
        ti = t0 + 0.2 + i * step
        last = i == len(SLOT) - 1
        vin = ramp(t, ti, ti + 0.12)
        vout = 0 if last else ramp(t, ti + step, ti + step + 0.12)
        if vin <= 0 or vout >= 1:
            continue
        dy = 22 * (1 - vin) - 22 * vout
        col = TERRA if last else DIM
        draw_text(canvas, w + ("." if last else ""), SERIF_IT, size, xs, base + dy, col, a * vin * (1 - vout))
    # soulignement du mot trouvé
    tl = t0 + 0.2 + (len(SLOT) - 1) * step
    u = ramp(t, tl + 0.1, tl + 0.5)
    draw_rect(canvas, xs, base + 16, xs + u * text_width("claire", SERIF_IT, size), base + 18, TERRA, a * 0.8)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def radial_bg():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - CX) / W) ** 2 + ((y - CY) / H) ** 2) * 1.6
    inner = np.array([24, 21, 18], np.float32) / 255
    outer = np.array([7, 6, 6], np.float32) / 255
    k = smooth(d)[..., None]
    return inner * (1 - k) + outer * k


BG = radial_bg()
GRAIN = [np.random.default_rng(100 + i).normal(0, 0.011, (H, W, 1)).astype(np.float32) for i in range(6)]


def blur_small(img, r, passes=3):
    for _ in range(passes):
        for ax in (0, 1):
            c = np.cumsum(np.pad(img, [(r + 1, r) if a == ax else (0, 0) for a in range(3)], mode="edge"), axis=ax)
            if ax == 0:
                img = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
            else:
                img = (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)
    return img


def bloom(layer, strength):
    small = layer.reshape(H // 4, 4, W // 4, 4, 3).mean(axis=(1, 3))
    small = blur_small(small, 6)
    up = np.stack([np.asarray(Image.fromarray(small[..., c].astype(np.float32), "F").resize((W, H), Image.BILINEAR))
                   for c in range(3)], -1)
    return layer + up * strength


def frame(t, fi=0):
    canvas = BG * ramp(t, 0, 0.6)
    canvas = canvas.copy()

    # lettres-particules (additif + halo)
    layer = np.zeros((H, W, 3), np.float32)
    draw_particles(layer, t)
    glow = 1.4 + 1.6 * window(t, wt("present", "entierement") - 0.6, wt("feel", "est-ce") + 0.8, 0.6, 0.8) \
        + 0.8 * window(t, wt("together", "a"), wt("start", "alors") + 0.4, 0.3, 0.6)
    canvas += bloom(layer, glow)

    # textes
    hello_a = 1 - ramp(t, T_WORDS - 0.05, T_WORDS + 0.3)
    if hello_a > 0:
        HELLO.draw(canvas, t, hello_a)
    draw_stream(canvas, t)
    draw_code(canvas, t)
    draw_sentence(canvas, t)
    if t > wt("start", "alors") - 0.4:
        ENDING.draw(canvas, t, 1.0, cursor_alpha=ramp(t, wt("start", "alors") - 0.25, wt("start", "alors")))
        sa = ramp(t, wt("start", end=True) + 1.2, wt("start", end=True) + 2.2)
        if sa > 0:
            name, rest = "Claude", "  ·  une IA créée par Anthropic"
            w1, w2 = text_width(name, SERIF_IT, 46), text_width(rest, SANS, 26)
            x = CX - (w1 + w2) / 2
            draw_text(canvas, name, SERIF_IT, 46, x, 930, IVORY, sa)
            draw_text(canvas, rest, SANS, 26, x + w1, 930, DIM, sa)
    for i in range(len(LINES)):
        draw_caption(canvas, i, t)

    canvas += GRAIN[fi % len(GRAIN)]
    canvas *= 1 - ramp(t, DURATION - 1.1, DURATION - 0.05)
    return (np.clip(canvas, 0, 1) ** (1 / 1.0) * 255 + 0.5).astype(np.uint8)


def encode(frames, out):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12",
           "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in frames:
        p.stdin.write(frame(fi / FPS, fi).tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    if sys.argv[1] == "still":
        t = float(sys.argv[2])
        Image.fromarray(frame(t, int(t * FPS))).save(sys.argv[3])
    elif sys.argv[1] == "segment":
        encode(range(int(sys.argv[2]), int(sys.argv[3])), sys.argv[4])
    elif sys.argv[1] == "keys":
        print(json.dumps({"hello": HELLO.key_times, "ending": ENDING.key_times}))
