#!/usr/bin/env bash
# Reconstruit « ENTRÉE » : musique, 900 images, montage.
set -euo pipefail
cd "$(dirname "$0")"
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
JOBS=${JOBS:-4}
FRAMES=900

python3 music.py                      # build/music.wav

step=$(( (FRAMES + JOBS - 1) / JOBS ))
rm -f build/seg*.mp4 build/segments.txt
for ((j = 0; j < JOBS; j++)); do
  a=$(( j * step )); b=$(( a + step < FRAMES ? a + step : FRAMES ))
  python3 render.py segment $a $b build/seg$j.mp4 &
  echo "file 'seg$j.mp4'" >> build/segments.txt
done
wait

"$FFMPEG" -y -loglevel error -f concat -safe 0 -i build/segments.txt -i build/music.wav \
  -map 0:v -map 1:a -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 256k -shortest entree-teaser.mp4
echo "OK -> entree-teaser.mp4"
