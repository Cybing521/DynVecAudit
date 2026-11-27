#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== [1/4] DynVecAudit Four-Gate Hygiene Self-Test ==="
python3 hygiene/audit_hygiene.py --self-test

echo "=== [2/4] Build & Test Counterfactual Replay Simulator (qbam_replay) ==="
if command -v cmake >/dev/null 2>&1; then
  ./replay/build.sh
  ctest --test-dir replay/build --output-on-failure
else
  echo "cmake not found; skipping C++ replay engine build."
fi

echo "=== [3/4] Unified Ranking-Flip Matrix & Paper Artifacts ==="
if [[ -d "${RESULTS_DIR:-../results}" ]]; then
  echo "Detected results tree at ${RESULTS_DIR:-../results}."
  echo "Regenerate all 80 verified flip-matrix rows (local_qbam=49, exp6_real_engine=18, colab_1m_audit=7, exp7_gpu_1m_arxiv=6),"
  echo "review audits (A/B/C), LaTeX tables, and paper figures via:"
  echo "  OMP_NUM_THREADS=1 python3 scripts/compute_flip_matrix.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/compute_protocol_reaudit_and_pareto.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/build_paper_tables.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/generate_paper_figures.py"
  echo "  OMP_NUM_THREADS=1 python3 scripts/build_fig1_framework.py"
  echo ""
  echo "Additional sprint_v2, real-engine & audit suites supported by scripts/:"
  echo "  - scripts/run_mimo_sprint_v2.py                (canon25, p0_2_long_seasonal, sift1m_n12)"
  echo "  - scripts/run_llm01_real_engine_traps.py       (exp6_real_engine_traps: A1+D, A2, A3, A4)"
  echo "  - scripts/run_gpu_1m_and_arxiv.py              (exp7_gpu_1m_arxiv: B1 1M SIFT & B3 arXiv MiniLM-384)"
  echo "  - scripts/run_heldout_schedule_audit.py        (Audit A: held-out t_sw Trap 2 split vs clock check)"
  echo "  - scripts/run_milp_and_horizon_sensitivity.py  (Audit B: Prop. 1 cost lambda & Horizon H sensitivity)"
else
  echo "No local results/ tree found inside standalone checkout."
  echo "See README.md for NUMA-pinned single-core engine execution, sprint_v2, and replay commands."
fi

echo "=== [4/4] Script Syntax & Route-A Separation Sanity Check ==="
python3 -m py_compile scripts/*.py
if grep -rnE "hybrid|inplace|freeze" replay/cpp/ replay/CMakeLists.txt replay/third_party/hnswlib/hnswlib/ >/dev/null 2>&1; then
  echo "ERROR: Unexpected Route-A symbol found in replay/."
  exit 1
fi
echo "Script compile & Route-A check PASS (0 hits). DynVecAudit verification complete."
