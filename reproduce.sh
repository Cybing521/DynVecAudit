#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== [1/4] DynVecAudit Hygiene Audit Self-Test ==="
python3 hygiene/audit_hygiene.py --self-test

echo "=== [2/4] Build & Test Counterfactual Replay Simulator (qbam_replay) ==="
if command -v cmake >/dev/null 2>&1; then
  ./replay/build.sh
  ctest --test-dir replay/build --output-on-failure
else
  echo "cmake not found; skipping C++ replay engine build."
fi

echo "=== [3/4] Paper Analysis, Tables & Figures ==="
if [[ -d "${RESULTS_DIR:-../results}" ]]; then
  echo "Detected results tree at ${RESULTS_DIR:-../results}."
  echo "Regenerate flip matrix, decompositions, floor sensitivity, LaTeX tables/macros, and paper figures via:"
  echo "  OMP_NUM_THREADS=1 python3 scripts/run_r3_exp1_orthogonal_decomp.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/run_r3_exp2_real_streams_cost_phase.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/analyze_floor_sensitivity.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/compute_flip_matrix.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/compute_protocol_reaudit_and_pareto.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/build_paper_tables.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/build_story_tables.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/generate_paper_figures.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/generate_story_figures.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/build_fig1_framework.py"
else
  echo "No local results/ tree found inside standalone checkout."
  echo "See README.md for NUMA-pinned single-core engine execution, pacing generality, and replay commands."
fi

echo "=== [4/4] Script Syntax & Separation Sanity Check ==="
python3 -m py_compile scripts/*.py
if grep -rnE "hybrid|inplace|freeze" replay/cpp/ replay/CMakeLists.txt replay/third_party/hnswlib/hnswlib/ >/dev/null 2>&1; then
  echo "ERROR: Unexpected symbol found in replay/."
  exit 1
fi
echo "Script compile & check PASS (0 hits). DynVecAudit verification complete."
