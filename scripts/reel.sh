#!/usr/bin/env bash
# Phase 5b: frame sequences for a reel, 1920 x 1080 at 30 fps, JPEG q92,
# outside the repo ($REEL/frames/<sequence>/NNNN.jpg + frames.json), and
# $REEL/manifest.json (path, frames, fps, caption, the per-frame mean
# spectrum in 40 bins over 380-780 nm).
#
#   scripts/reel.sh [sequences]   default: all of
#     classroom-keyframed pavilion-dolly pavilion-orbit fit rose head swatch
#
# Env: REEL (../irys-assets/reel), EXPORTS (../irys-assets/blender-demos/export5b),
# SPP_CLASSROOM (64), SPP_PAVILION (48), MIN_FREE_GB (15), CARGO_TARGET_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
REEL=${REEL:-../irys-assets/reel}
EXPORTS=${EXPORTS:-../irys-assets/blender-demos/export5b}
SEQS=${1:-classroom-keyframed,pavilion-dolly,pavilion-orbit,fit,rose,head,swatch}
MIN_FREE_GB=${MIN_FREE_GB:-15}
T=${CARGO_TARGET_DIR:-target}
cargo build --release -p irys-diff --features gpu,assets --example blend_look --example reel
mkdir -p "$REEL/frames" "$REEL/cams"

free_gb() { df -g /System/Volumes/Data | awk 'NR==2 {print $4}'; }
check_disk() {
  # a 1080p q92 frame is ~0.3-0.8 MB; stop if a 200-frame sequence would
  # take the disk under the floor
  if [ "$(free_gb)" -lt $((MIN_FREE_GB + 1)) ]; then
    echo "disk: $(free_gb) GB free, under ${MIN_FREE_GB}+1 GB: stopping" >&2
    exit 3
  fi
}

python3 scripts/reel_cameras.py "$EXPORTS/classroom" keyframed "$REEL/cams/classroom-keyframed.json"
python3 scripts/reel_cameras.py "$EXPORTS/pavilion" dolly "$REEL/cams/pavilion-dolly.json" --seconds 5 --distance 4 --rise 0.3 --distance-to-target 25
python3 scripts/reel_cameras.py "$EXPORTS/pavilion" orbit "$REEL/cams/pavilion-orbit.json" --seconds 5 --distance 18 --degrees 18 --rise 0.4

for s in ${SEQS//,/ }; do
  check_disk
  case $s in
    classroom-keyframed) "$T/release/examples/blend_look" "$EXPORTS/classroom" --sequence "$REEL/cams/$s.json" --outdir "$REEL/frames/$s" \
        --spp "${SPP_CLASSROOM:-64}" --bounces 7 \
        --caption "Blender's classroom on irys's OpenPBR GPU path: the file's own animated camera, resampled to 30 fps" \
        --credit "classroom by Christophe Seux, CC0, from blender.org's demo files" ;;
    pavilion-dolly|pavilion-orbit) "$T/release/examples/blend_look" "$EXPORTS/pavilion" --sequence "$REEL/cams/$s.json" --outdir "$REEL/frames/$s" \
        --spp "${SPP_PAVILION:-48}" --bounces 7 \
        --caption "The Barcelona Pavilion on irys's OpenPBR GPU path: an authored ${s#pavilion-} (the file's camera does not move)" \
        --credit "Barcelona Pavillion by eMirage, CC-BY, from blender.org's demo files" ;;
    fit) "$T/release/examples/reel" fit "$REEL/frames/fit" ;;
    rose) "$T/release/examples/reel" rose "$REEL/frames/rose" ;;
    head) "$T/release/examples/reel" head "$REEL/frames/head" ;;
    swatch) "$T/release/examples/reel" swatch "$REEL/frames/swatch" --env "$EXPORTS/pavilion" ;;
    *) echo "unknown sequence $s" >&2; exit 2 ;;
  esac
done
"$T/release/examples/reel" manifest "$REEL"
