#!/usr/bin/env bash
# Phase 3's equal-time benchmark, irys vs Cycles. See crates/irys-bench.
#
#   scripts/bench.sh [scenes] [variants]      e.g. scripts/bench.sh pair,room base,restir
#
# Env: OUT (out/bench), REF_SPP (32768), MAX_SPP (1024), BLENDER
# (/Applications/Blender.app/Contents/MacOS/Blender), CARGO_TARGET_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
SCENES=${1:-pair,room}
VARIANTS=${2:-base}
OUT=${OUT:-out/bench}
REF_SPP=${REF_SPP:-32768}
MAX_SPP=${MAX_SPP:-1024}
BLENDER=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
cargo build --release -p irys-bench
BENCH=${CARGO_TARGET_DIR:-target}/release/irys-bench
"$BENCH" export --scene "$SCENES" --out "$OUT"
for s in ${SCENES//,/ }; do
  if [ ! -f "$OUT/$s/cycles.json" ]; then
    "$BLENDER" -b --factory-startup -P scripts/cycles_bench.py -- "$OUT/$s" \
      --ref-spp "$REF_SPP" --max-spp "$MAX_SPP" 2>&1 | grep -E "^cycles|Error|Traceback" || true
  fi
  for v in ${VARIANTS//,/ }; do
    "$BENCH" irys --scene "$s" --out "$OUT" --variant "$v" --ref-spp "$REF_SPP" --max-spp "$MAX_SPP"
  done
done
"$BENCH" report --out "$OUT" --scenes "$SCENES" --variants "$VARIANTS" --docs docs
