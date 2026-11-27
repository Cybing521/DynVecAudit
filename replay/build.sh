#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
MARCH="${QBAM_MARCH:-native}"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQBAM_BUILD_TESTS=ON \
  -DQBAM_MARCH="$MARCH"

cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "Built qbam_replay at: $BUILD_DIR/qbam_replay"
