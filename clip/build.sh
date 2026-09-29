#!/usr/bin/env bash
# Reconstruit le clip « Synthetic Scream ».
# Pré-requis : le morceau dans build/song.wav (non versionné), les images dans assets/.
# Le rendu est découpé en morceaux de 150 images : relancer le script reprend là où il s'était arrêté.
set -euo pipefail
cd "$(dirname "$0")"
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
JOBS=${JOBS:-4}
CHUNK=150

[[ -f build/features.npz ]] || python3 features.py
FRAMES=$(python3 -c "import numpy as np, math; print(math.ceil(float(np.load('build/features.npz')['duration']) * 30))")
mkdir -p build/chunks

render_chunk() {
  a=$1; b=$(( a + CHUNK < FRAMES ? a + CHUNK : FRAMES ))
  out=$(printf "build/chunks/c%05d.mp4" "$a")
  [[ -f $out ]] && return 0
  python3 render.py segment "$a" "$b" "$out.part.mp4" && mv "$out.part.mp4" "$out"
}
export -f render_chunk
export CHUNK FRAMES
seq 0 $CHUNK $((FRAMES - 1)) | xargs -P "$JOBS" -I{} bash -c 'render_chunk {} || echo "ÉCHEC morceau {}"'

n_expected=$(( (FRAMES + CHUNK - 1) / CHUNK ))
n_done=$(ls build/chunks/ | grep -v part | grep -c "^c.*mp4$" || true)
if [[ $n_done -ne $n_expected ]]; then
  echo "Morceaux manquants ($n_done/$n_expected) : relancer ./build.sh"; exit 1
fi
ls build/chunks/ | grep -v part | grep "^c.*mp4$" | sed "s|^|file 'chunks/|; s|$|'|" > build/segments.txt
"$FFMPEG" -y -loglevel error -f concat -safe 0 -i build/segments.txt -i build/song.wav \
  -map 0:v -map 1:a -c:v libx264 -preset slow -crf 21 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 256k -shortest synthetic-scream.mp4
echo "OK -> synthetic-scream.mp4"
