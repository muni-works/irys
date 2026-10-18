#!/usr/bin/env bash
# Phase 5a: the equal-time benchmark on Blender's demo files.
#
#   scripts/scenes_bench.sh [scenes] [variants]     e.g. scripts/scenes_bench.sh classroom,pavilion
#
# For each scene, two benchmarks:
# - `<scene>`: true materials. Cycles renders the .blend itself
#   (scripts/cycles_blend.py); irys renders its import of it.
# - `<scene>-grey`: equal materials. irys's import with every non-emitter a
#   grey Lambertian (0.8), exported by irys-bench and rebuilt in Cycles
#   (scripts/cycles_bench.py, instanced), so both render one scene.
# The .blend files are not in the repo: ASSETS (../irys-assets/blender-demos)
# holds them and their ATTRIBUTION.md; exports go to $ASSETS/export.
#
# Env: OUT (out/scenes), REF_SPP (32768), MAX_SPP (1024), W, H (480 x 270),
# BLENDER, CARGO_TARGET_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
SCENES=${1:-classroom,pavilion}
VARIANTS=${2:-sobol+spectral,sobol+spectral+guide}
OUT=${OUT:-out/scenes}
REF_SPP=${REF_SPP:-32768}
MAX_SPP=${MAX_SPP:-1024}
W=${W:-480}
H=${H:-270}
ASSETS=${ASSETS:-../irys-assets/blender-demos}
BLENDER=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
export IRYS_SCENES=${IRYS_SCENES:-$ASSETS/export}
cargo build --release -p irys-bench
BENCH=${CARGO_TARGET_DIR:-target}/release/irys-bench

blend_of() {
  case $1 in
    classroom) echo "$ASSETS/classroom/classroom/classroom.blend|" ;;
    pavilion) echo "$ASSETS/pabellon/3d/pavillon_barcelone_v1.2.blend|1- time: midday" ;;
    *) echo "unknown scene $1" >&2; exit 2 ;;
  esac
}

all=()
for s in ${SCENES//,/ }; do
  IFS='|' read -r file scene <<<"$(blend_of "$s")"
  sargs=()
  [ -n "$scene" ] && sargs=(--scene "$scene")
  if [ ! -f "$IRYS_SCENES/$s/irys.json" ]; then
    "$BLENDER" -b -Y "$file" -P scripts/blend_export.py -- "$IRYS_SCENES/$s" ${sargs[@]+"${sargs[@]}"} --width "$W" 2>&1 | grep -E "^irys export|Error|Traceback" || true
  fi
  # true materials
  mkdir -p "$OUT/$s"
  if [ ! -f "$OUT/$s/cycles.json" ]; then
    "$BLENDER" -b -Y "$file" -P scripts/cycles_blend.py -- "$OUT/$s" "$IRYS_SCENES/$s" ${sargs[@]+"${sargs[@]}"} \
      --ref-spp "$REF_SPP" --max-spp "$MAX_SPP" 2>&1 | grep -E "^cycles|Error|Traceback" || true
  fi
  # grey
  g="$s-grey"
  "$BENCH" export --scene "$g" --out "$OUT" --w "$W" --h "$H"
  if [ ! -f "$OUT/$g/cycles.json" ]; then
    "$BLENDER" -b --factory-startup -P scripts/cycles_bench.py -- "$OUT/$g" \
      --ref-spp "$REF_SPP" --max-spp "$MAX_SPP" 2>&1 | grep -E "^cycles|Error|Traceback" || true
  fi
  for t in "$s" "$g"; do
    for v in ${VARIANTS//,/ }; do
      "$BENCH" irys --scene "$t" --out "$OUT" --variant "$v" --ref-spp "$REF_SPP" --max-spp "$MAX_SPP" --w "$W" --h "$H"
    done
    all+=("$t")
  done
done
list=$(IFS=,; echo "${all[*]}")
"$BENCH" report --out "$OUT" --scenes "$list" --variants "$VARIANTS" --docs docs --prefix scenes
