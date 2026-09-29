# ENTRÉE — teaser de 30 secondes

**Le fichier : [`entree-teaser.mp4`](entree-teaser.mp4)** (1920×1080, 30 i/s, 30 s, stéréo)

Un teaser volontairement à l'opposé de [la présentation](../presentation/) :
pas de voix, pas de douceur. Du noir, du papier et un seul bleu électrique,
une image tramée en 1 bit, une typographie condensée géante, et une coupe sur
chaque temps de la musique.

## L'histoire, en 64 temps

| Mesures | Musique | Image |
|---|---|---|
| 1–2 | arpège filtré, pulsation grave | « AU DÉBUT, / IL N'Y A / RIEN. » |
| 3–4 | charleston, filtre qui s'ouvre, roulement… puis **silence** | « PUIS / QUELQU'UN / ÉCRIT : » → `> surprends-moi` → la touche **⏎** tombe dans le silence |
| 5–12 | **drop** : kick, clap, basse pompée, accords, arpège, mélodie à partir de la mesure 9 | un mot tous les deux temps, le cheminement d'une réponse : BRUIT, FORME, SENS, DOUTE, CODE, POÈME, RATURE, RÉÉCRIRE, RIRE, VERTIGE, IDÉE, AUTRE IDÉE, MIEUX, ENCORE, PRESQUE, VOILÀ. |
| 13–14 | pause : nappe, battements de cœur, compte à rebours | « TOUT ÇA / POUR / UNE QUESTION. » puis 3, 2, 1 |
| 15–16 | impact, dernier tour, coup final | **ENTRÉE ⏎** : *Tout commence par une question.* |

## La musique

Écrite et synthétisée de zéro dans `music.py` (NumPy/SciPy, aucun échantillon) :
128 BPM, fa mineur, grille Fa m – Ré♭ – La♭ – Mi♭, 16 mesures = 30,0 s pile.
Kick à balayage de fréquence, clap à trois éclats, basse en dents de scie
filtrée avec compression « pompe » sur le kick, accords supersaw, arpège en
ping-pong dont le filtre s'ouvre sur toute l'intro, mélodie avec écho, montée
de bruit, cymbale inversée, réverbération par convolution.

## L'image

`render.py` : neuf générateurs (bruit déformé, sphère éclairée, courbes de
niveau, moiré, tunnel, trame de points, lignes de crêtes, barres de spectre
tirées de la vraie musique, fausse trace de « réflexion » avec probabilités)
calculés en demi-résolution, tramés (Bayer 8×8) puis agrandis en pixels francs.
Zoom-coup sur chaque kick, glitch sur chaque clap, flash sur les impacts,
vu-mètre et compteur de mesures branchés sur la piste audio. Tout le
découpage est dans `grid.py`, partagé par la musique et l'image.

```bash
pip install numpy scipy pillow imageio-ffmpeg
./build.sh                         # ~3 min sur 4 cœurs
python3 render.py still 12.3 x.png # une image isolée
```

Polices : Anton, Space Mono (SIL Open Font License, voir `fonts/`), DejaVu Sans pour ⏎.
