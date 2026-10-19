#!/bin/sh
# Build irys-ffi and the Swift harness, and run it. Args: an optional .ply.
set -e
here=$(cd "$(dirname "$0")" && pwd)
target=${CARGO_TARGET_DIR:-$here/../../../target}
cargo build --release -p irys-ffi
mkdir -p "$target/irys-ffi-harness"
swiftc -O "$here/main.swift" -import-objc-header "$here/../include/irys.h" \
    -L "$target/release" -lirys_ffi -o "$target/irys-ffi-harness/harness"
DYLD_LIBRARY_PATH="$target/release" "$target/irys-ffi-harness/harness" "$@"
