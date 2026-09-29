"""Grille temporelle commune à la musique et à l'image.

128 BPM, 4/4, 16 mesures = exactement 30 secondes.

  mesures  0-1   intro      arpège filtré, pulsations graves
  mesures  2-3   montée     charleston, filtre qui s'ouvre, roulement, silence
  mesures  4-11  drop       kick, clap, basse, accords, arpège (+ mélodie dès la mesure 8)
  mesures 12-13  pause      nappe, battement de cœur, compte à rebours
  mesures 14-15  final      impact, titre
"""
BPM = 128
BEAT = 60 / BPM            # 0.46875 s
BAR = 4 * BEAT             # 1.875 s
SIX = BEAT / 4             # double croche
BARS = 16
DURATION = BARS * BAR      # 30.0 s


def t(bar, beat=0.0):
    """Instant (s) d'une mesure et d'un temps (temps comptés à partir de 0)."""
    return bar * BAR + beat * BEAT


def beat_of(time):
    return time / BEAT


# fa mineur : Fa m – Ré♭ – La♭ – Mi♭, une mesure chacun
CHORDS = [
    # (fondamentale basse, accord en voicing medium)
    (41, [65, 68, 72]),    # Fa m
    (37, [65, 68, 73]),    # Ré♭
    (44, [63, 68, 72]),    # La♭
    (39, [63, 67, 70]),    # Mi♭
]


def chord_at(bar):
    return CHORDS[bar % 4]


# sections (mesures de début, de fin exclue)
INTRO = (0, 2)
BUILD = (2, 4)
DROP = (4, 12)
BREAK = (12, 14)
FINAL = (14, 16)


def kicks():
    out = []
    for bar in range(BARS):
        if BUILD[0] <= bar < BUILD[1]:
            out += [t(bar, 0), t(bar, 2)] if bar == 2 else [t(bar, 0), t(bar, 1), t(bar, 2)]
        elif DROP[0] <= bar < DROP[1] or bar == 14:
            out += [t(bar, b) for b in range(4)]
    out.append(t(15, 0))
    return out


def claps():
    out = []
    for bar in list(range(*DROP)) + [14]:
        out += [t(bar, 1), t(bar, 3)]
    return out


def heartbeats():
    return [t(bar, b) for bar in range(*BREAK) for b in (0, 2)]


IMPACTS = [t(4), t(14), t(15)]          # entrée du drop, retour, coup final
GAP = (t(3, 3), t(4))                   # silence juste avant le drop
