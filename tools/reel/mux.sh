#!/usr/bin/env bash
# Mux a frame directory and the soundtrack into irys-reel.mp4.
#   ./mux.sh FRAMES_DIR [out/soundtrack.wav] [out/irys-reel.mp4]
# Frames are %05d.png or %05d.jpg, numbered from 0; FPS defaults to 30.
# The video is cut to the soundtrack's length; the audio is never trimmed to
# the frames, so a short frame set ends on a held last frame.
set -euo pipefail
FRAMES=${1:?usage: mux.sh FRAMES_DIR [soundtrack.wav] [out.mp4]}
AUDIO=${2:-out/soundtrack.wav}
OUT=${3:-out/irys-reel.mp4}
FPS=${FPS:-30}
EXT=png; [ -e "$FRAMES/00000.png" ] || EXT=jpg
[ -e "$FRAMES/00000.$EXT" ] || { echo "no $FRAMES/00000.png or .jpg" >&2; exit 1; }
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
ffmpeg -v error -y -framerate "$FPS" -i "$FRAMES/%05d.$EXT" -i "$AUDIO" \
  -vf "tpad=stop_mode=clone:stop_duration=$DUR" -t "$DUR" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -c:a aac -b:a ${ABR:-192k} \
  -movflags +faststart "$OUT"
echo "$OUT"
