#!/usr/bin/env bash
# Reconstruit la vidéo de bout en bout.
#   VOICE=chemin/vers/fr_FR-siwis-medium.onnx ./build.sh          (tout, voix comprise)
#   ./build.sh                                                     (réutilise build/voice.wav)
set -euo pipefail
cd "$(dirname "$0")"
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
JOBS=${JOBS:-4}
FRAMES=1800

if [[ -n "${VOICE:-}" ]]; then
  python3 tts.py "$VOICE"      # build/voice.wav + build/timeline.json
  python3 align.py             # build/words.json
fi
python3 render.py keys > build/keys.json
python3 music.py               # build/mix.wav

step=$(( (FRAMES + JOBS - 1) / JOBS ))
rm -f build/seg*.mp4 build/segments.txt
for ((j = 0; j < JOBS; j++)); do
  a=$(( j * step )); b=$(( a + step < FRAMES ? a + step : FRAMES ))
  python3 render.py segment $a $b build/seg$j.mp4 &
  echo "file 'seg$j.mp4'" >> build/segments.txt
done
wait

# léger débruitage : le grain de pellicule coûte très cher en débit sinon
"$FFMPEG" -y -loglevel error -f concat -safe 0 -i build/segments.txt -i build/mix.wav \
  -map 0:v -map 1:a -vf hqdn3d=3:2:4:3 -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 192k -shortest claude-presentation.mp4
echo "OK -> claude-presentation.mp4"
