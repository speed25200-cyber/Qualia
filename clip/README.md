# Synthetic Scream — clip

**Le fichier : [`synthetic-scream.mp4`](synthetic-scream.mp4)** (1920×1080, 30 i/s, 3 min 05, version web
~90 Mo ; le master à 33 Mbit/s, 770 Mo, se reconstruit avec `./build.sh`)

Clip réalisé sur le morceau « Synthetic Scream » (darksynth / industriel,
~136 BPM, fa mineur). Tout part de l'écoute du morceau et de ses paroles :
une androïde dans une ville cyberpunk en feu, sous une pluie de néons, qui
« s'allume », descend dans une rave souterraine, perd pied, et finit par
crier puis reprendre le contrôle.

## Les images

Trois images seulement, générées avec Higgsfield (modèle GPT Image 2.5,
2,5 crédits au total), le reste est entièrement calculé :

| Fichier | Rôle |
|---|---|
| `assets/city.png` | la ville : avenue inondée, néons rouges et cyan, feux, fumée |
| `assets/android.png` | l'androïde : visage mi-chair mi-chrome, yeux rouges, pluie |
| `assets/scream.png` | le cri : même visage, chrome fissuré de lumière rouge (généré à partir du portrait pour garder le personnage) |

## Le découpage

| Temps | Partie | Ce qu'on voit |
|---|---|---|
| 0:00 | démarrage | écran noir, terminal « SYNTHETIC_OS… SIGNAL SEARCHING », parasites |
| 0:03 | intro (*Signal lost / Neon rain…*) | la ville à travers la neige télé, paroles décodées au centre |
| 0:16 | montée instrumentale | lente plongée dans l'avenue, zoom sur chaque kick, éclairs subliminaux des yeux rouges |
| 0:30 | couplet 1 | un plan par ligne : rues en feu, néons qui meurent, *Chrome on skin* (gros plan chrome), *Eyes in red* (réticules sur les yeux), *Broken dreams* (image brisée), *Neural choke* (pixels triés), *Kill the light* (noir, seuls les yeux brillent), *Feel the system come alive* (démarrage du système) |
| 0:57 | refrain 1 (*Concrete pulse…*) | coupe à chaque temps entre yeux, ville, lasers, chrome ; un mot plein écran par temps ; *Harder / Faster* en croches |
| 1:25 | *No escape* | cadres imbriqués sans fin, grille « NO ESCAPE » |
| 1:39 | couplet 2 | ciel noirci, cicatrices numériques, étoiles de néon, *Machine whispers* (visage recomposé avec ces mots), *Emotion dead* (les yeux s'éteignent), faisceaux, labyrinthe, électrocardiogramme, *System overload* (surchauffe, négatif), fondu au blanc |
| 2:06 | refrain 2 | coupe franche sur **le cri** ; chaque *break / they / no* répété brise l'image |
| 2:23 | *Signal fading / Heartbeat slow* | le cœur ralentit, parasites, *Take control* : les yeux se verrouillent |
| 2:35 | final (cris) | montage frénétique autour du cri, « SYNTHETIC / SCREAM », croches à la fin |
| 2:58 | sortie | le signal se dégrade, extinction d'écran cathodique, titre |

## Le rendu « cinéma »

- **Caméra 2,5D** : une carte de profondeur par image (Depth Anything V2) fait
  glisser le premier plan devant le fond ; travelling avant réel dans
  l'avenue, mises au point qui basculent (*Chrome on skin*, *Lights decay*).
- **Optique** : format 2.39:1 qui s'ouvre en plein cadre sur les refrains,
  reflets anamorphiques sur les néons et les yeux, halo, aberration
  chromatique, légère distorsion, flottement de pellicule, courbe film, grain.
- **Matière** : pluie sur trois plans de profondeur, gouttes sur l'objectif qui
  réfractent l'image, caméra à l'épaule, filé sur les coupes rapides.
- **Netteté** : la ville (générée en 1K) est agrandie ×4 par Real-ESRGAN.

## Fabrication

- `features.py` : analyse du morceau image par image (énergie, sub, basses,
  médiums, aigus, kicks, caisses claires, temps).
- `lyrics.py` : paroles transcrites (Whisper medium + large-v3), vérifiées et
  horodatées mot par mot. Trois mots restent incertains, marqués « ? »
  (*Closed/Cold*, *Neural/Mineral*, *At/Of*) : corriger le texte et relancer.
- `fx.py` : caméra sur image fixe, étalonnage, glitchs (RGB, tranches, blocs,
  tri de pixels, éclats), pluie, neige télé, halo, grain, lignes, texte.
- `cine.py` : caméra 2,5D, profondeur de champ, pluie, gouttes, optique et pellicule.
- `depth.py`, `upscale.py` : cartes de profondeur et agrandissement IA (ONNX, sur CPU).
- `render.py` : le découpage ci-dessus, plan par plan.

```bash
pip install numpy scipy pillow librosa imageio-ffmpeg opencv-python-headless onnxruntime
cp "Synthetic Scream.wav" build/song.wav   # le morceau n'est pas versionné
./build.sh                                  # ~45 min sur 4 cœurs, reprenable
python3 render.py still 38.9 image.png      # une image isolée
```

Polices : Chakra Petch, Share Tech Mono (SIL Open Font License, voir `fonts/`).
