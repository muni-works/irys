#!/usr/bin/env bash
# Phase 5b: the equal-time benchmark on Blender's demo files with true
# materials: Cycles renders the .blend, irys its OpenPBR import on the GPU
# scene kernel (examples/look_bench). Phase 3's protocol: 480 x 270,
# max_bounces 7, 32768-spp references for both renderers, sweeps to 1024
# spp, fastest of two, each renderer against its own reference, budgets at
# Cycles' 64 / 256 / 1024-spp times. `<scene>-clamp` clamps indirect samples
# at CLAMP in both (Cycles' sample_clamp_indirect, irys's per-lane clamp).
#
#   scripts/demos_bench.sh [classroom,pavilion,pavilion-clamp]
#
# Env: OUT (out/bench5b), EXPORTS (../irys-assets/blender-demos/export5b),
# REF_SPP, MAX_SPP, W (480), CLAMP (10), BLENDER, CARGO_TARGET_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
SCENES=${1:-classroom,pavilion,pavilion-clamp}
OUT=${OUT:-out/bench5b}
ASSETS=${ASSETS:-../irys-assets/blender-demos}
EXPORTS=${EXPORTS:-$ASSETS/export5b}
REF_SPP=${REF_SPP:-32768}
MAX_SPP=${MAX_SPP:-1024}
W=${W:-480}
CLAMP=${CLAMP:-10}
BLENDER=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
T=${CARGO_TARGET_DIR:-target}
cargo build --release -p irys-bench
cargo build --release -p irys-diff --features gpu,assets --example look_bench

for t in ${SCENES//,/ }; do
  s=${t%-clamp}
  clamp=0
  [ "$t" != "$s" ] && clamp=$CLAMP
  case $s in
    classroom) file="$ASSETS/classroom/classroom/classroom.blend"; sargs=() ;;
    pavilion) file="$ASSETS/pabellon/3d/pavillon_barcelone_v1.2.blend"; sargs=(--scene "1- time: midday") ;;
    *) echo "unknown scene $s" >&2; exit 2 ;;
  esac
  mkdir -p "$OUT/$t"
  if [ ! -f "$OUT/$t/cycles.json" ]; then
    "$BLENDER" -b -Y "$file" -P scripts/cycles_blend.py -- "$OUT/$t" "$EXPORTS/$s" ${sargs[@]+"${sargs[@]}"} \
      --ref-spp "$REF_SPP" --max-spp "$MAX_SPP" --width "$W" --clamp "$clamp" 2>&1 | grep -E "^cycles|Error|Traceback" || true
  fi
  "$T/release/examples/look_bench" "$EXPORTS/$s" "$OUT/$t" --width "$W" --ref-spp "$REF_SPP" --max-spp "$MAX_SPP" \
    --clamp "$clamp" --variant openpbr
done
"$T/release/irys-bench" report --out "$OUT" --scenes "$SCENES" --variants openpbr --docs docs --prefix demos-bench
