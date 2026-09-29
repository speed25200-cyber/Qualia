#!/usr/bin/env bash
# Reconstruit le clip « Synthetic Scream ».
# Pré-requis : le morceau dans build/song.wav (non versionné), les 3 images dans assets/.
set -euo pipefail
cd "$(dirname "$0")"
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
JOBS=${JOBS:-4}

python3 features.py                                    # build/features.npz
FRAMES=$(python3 -c "import numpy as np, math; print(math.ceil(float(np.load('build/features.npz')['duration']) * 30))")

step=$(( (FRAMES + JOBS - 1) / JOBS ))
rm -f build/seg*.mp4 build/segments.txt
for ((j = 0; j < JOBS; j++)); do
  a=$(( j * step )); b=$(( a + step < FRAMES ? a + step : FRAMES ))
  python3 render.py segment $a $b build/seg$j.mp4 &
  echo "file 'seg$j.mp4'" >> build/segments.txt
done
wait

"$FFMPEG" -y -loglevel error -f concat -safe 0 -i build/segments.txt -i build/song.wav \
  -map 0:v -map 1:a -c:v libx264 -preset slow -crf 21 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 256k -shortest synthetic-scream.mp4
echo "OK -> synthetic-scream.mp4"
