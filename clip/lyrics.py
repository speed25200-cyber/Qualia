"""Paroles de « Synthetic Scream », horodatées mot par mot.

Transcrites automatiquement (Whisper medium et large-v3) puis vérifiées à la
main. Les mots incertains sont marqués « ? » en commentaire : il suffit de
corriger le texte ici et de relancer le rendu.

Styles d'affichage :
  center  une ligne au centre, décodée caractère par caractère (intro, fin)
  line    sous-titre bas-gauche, façon terminal (couplets)
  slam    un mot à la fois, plein écran, sur le temps (refrains)
  grid    le mot répété en grille (No escape)
"""

LINES = [
    # style, texte, [début de chaque mot]
    ("center", "SIGNAL LOST", [2.90, 4.08]),
    ("center", "NEON RAIN", [5.18, 6.20]),
    ("center", "STATIC PULSE", [8.86, 10.00]),
    ("center", "FEEL THE PAIN", [12.46, 13.20, 13.60]),

    ("line", "CLOSED STREETS BURN", [30.00, 30.64, 31.00]),          # « Cold » ?
    ("line", "LIGHTS DECAY", [31.70, 32.42]),
    ("line", "VOICES GLITCH, FADE AWAY", [33.38, 34.18, 35.22, 35.72]),
    ("line", "CHROME ON SKIN", [37.06, 37.44, 37.92]),
    ("line", "EYES IN RED", [38.60, 39.02, 39.64]),
    ("line", "CITY SCREAMS", [40.40, 40.98]),
    ("line", "FEED THE DEAD", [42.20, 42.58, 43.04]),
    ("line", "BASSLINE BITES", [44.00, 44.82]),
    ("line", "STEEL AND SMOKE", [45.62, 46.20, 46.54]),
    ("line", "BROKEN DREAMS", [47.34, 48.04]),
    ("line", "NEURAL CHOKE", [49.16, 49.50]),                        # « Mineral » ?
    ("line", "LOCK THE DOORS", [50.96, 51.36, 51.50]),
    ("line", "KILL THE LIGHT", [52.28, 53.10, 53.26]),
    ("line", "FEEL THE SYSTEM COME ALIVE TONIGHT", [53.90, 54.74, 54.98, 55.40, 55.68, 56.06]),

    ("slam", "CONCRETE PULSE", [58.36, 59.26]),
    ("slam", "GO INSANE", [59.72, 60.46]),
    ("slam", "HEARTS CORRUPTED", [61.66, 62.28]),
    ("slam", "IN THE RAIN", [63.56, 63.92, 64.26]),
    ("slam", "CONCRETE PULSE", [65.38, 66.20]),
    ("slam", "BODIES COLLIDE", [66.90, 67.30]),
    ("slam", "CYBER CHAOS", [68.52, 69.26]),
    ("slam", "NO PLACE TO HIDE", [70.26, 70.46, 70.92, 71.28]),
    ("slam", "CONCRETE PULSE", [72.22, 73.10]),
    ("slam", "FEEL THE SOUND", [73.88, 74.60, 74.76]),
    ("slam", "UNDERGROUND GODS", [75.44, 75.88]),
    ("slam", "SHAKING THE GROUND", [76.72, 77.62, 78.22]),
    ("slam", "HARDER", [81.38]),
    ("slam", "FASTER", [83.04]),

    ("grid", "NO ESCAPE", [87.30, 88.20]),
    ("grid", "NO ESCAPE", [89.00, 89.70]),
    ("grid", "NO ESCAPE", [93.22, 94.20]),

    ("line", "BLACKENED SKY", [99.04, 100.14]),
    ("line", "DIGITAL SCARS", [100.70, 101.28]),
    ("line", "LOST, LOST SOULS DANCING UNDER NEON STARS", [102.52, 103.02, 103.28, 103.66, 103.94, 104.44, 105.20]),
    ("line", "MACHINE WHISPERS INSIDE MY HEAD", [106.20, 106.72, 107.58, 108.14, 108.78]),
    ("line", "SYNTHETIC LOVE", [109.34, 109.90]),
    ("line", "EMOTION DEAD", [110.96, 111.66]),
    ("line", "SHARP LIGHTS CUT THROUGH THE HAZE", [113.00, 113.52, 113.88, 114.24, 115.32, 115.70]),
    ("line", "AT ENDLESS NIGHTS", [116.30, 116.60, 117.14]),        # « Of » ?
    ("line", "TOXIC MAZE", [118.60, 119.16]),
    ("line", "CAN YOU FEEL IT INSIDE YOUR VEINS?", [119.98, 120.74, 120.92, 121.24, 121.50, 122.28, 122.68]),
    ("line", "SYSTEM OVERLOAD", [123.36, 123.88]),
    ("line", "ERASE THE PAIN", [124.88, 125.30, 125.88]),

    ("slam", "CONCRETE PULSE", [127.80, 128.62]),
    ("slam", "BREAK BREAK BREAK BREAK THE CHAIN", [129.28, 129.64, 129.84, 130.02, 130.20, 130.46]),
    ("slam", "VIOLENCE FLOWING THROUGH MY BRAIN", [131.08, 131.90, 132.88, 133.30, 133.70]),
    ("slam", "CONCRETE PULSE", [134.62, 135.44]),
    ("slam", "THEY THEY THEY ATTACK", [136.20, 136.56, 136.74, 136.95]),
    ("slam", "NO TOMORROW", [137.84, 138.20]),
    ("slam", "NO NO NO TURNING BACK", [138.82, 139.34, 139.54, 139.68, 140.38]),

    ("center", "SIGNAL FADING", [143.30, 144.12]),
    ("center", "HEARTBEAT SLOW", [146.20, 147.52]),
    ("center", "HEARTBEAT SLOW", [148.72, 149.54]),
    ("center", "NEON SHADOWS", [150.38, 150.86]),
    ("center", "TAKE CONTROL", [153.62, 154.32]),
]


def lines():
    """Liste de dicts : style, texte, mots [(mot, début)], début, fin d'affichage."""
    out = []
    for i, (style, text, times) in enumerate(LINES):
        words = text.split()
        assert len(words) == len(times), text
        nxt = LINES[i + 1][2][0] if i + 1 < len(LINES) else times[-1] + 2.5
        end = min(nxt - 0.05, times[-1] + (2.2 if style != "slam" else 1.2))
        out.append(dict(style=style, text=text, words=list(zip(words, times)), start=times[0], end=end))
    return out
