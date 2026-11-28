# DynVecAudit

Official reproducibility artifact for the **PVLDB Volume 20 (VLDB 2027)** Experiment, Analysis & Benchmark (EA&B) submission:

> **When Dynamic Vector Index Benchmarks Mislead: Recall Floors, Placement, and Budget Pacing in Maintenance Evaluation [Experiment, Analysis \& Benchmark]**

---

## Overview

**DynVecAudit** is an open-source measurement-hygiene suite, 8-backend dynamic lifecycle harness (`7` codebases), and deterministic counterfactual replay simulator for dynamic approximate nearest neighbor (ANN) vector indexes under shifting workloads, mixed CRUD/runbook churn, and real-timestamp streams (`5,462` bare-metal runs + `2,400` counterfactual replay traces across `100K–10M` vectors and `128–960` dimensions).

### Five Audit Checks (`hygiene/audit_hygiene.py`)

1. **Quality Floor (`Recall@10 >= 0.965` in `>= 95%` of windows per run)**: Filters sub-floor configurations (`†`) before comparing engine cost, preventing mean-recall masking of late-window collapses.
2. **Check P — Placement & Symmetric Contention (`check_contamination`)**: Enforces single-thread execution (`OMP_NUM_THREADS=1`), physical core pinning within one NUMA node, zero cross-batch pooling, $\text{wall}/\text{CPU} \le 1.18$, and a co-scheduled 17.6 ms L2-GEMV probe slowdown $\le 1.25\times$ (catching silent hyper-thread contention where $\text{wall}/\text{CPU} \approx 1.00$), with paired-bootstrap confidence intervals (`paired_bootstrap_ci`).
3. **Check C — Executed Cadence Bracket (`check_cadence_bracket`)**: Requires every budgeted policy spend $M(\pi)$ to be bracketed between two *executed* adjacent integer cadences ($M(c{+}1) \le M(\pi) \le M(c)$), preventing linear-interpolation distortion across concave cost frontiers.
4. **Check S — Split & Clock Leakage (`check_clock_leak`)**: Requires train/test splits that randomize both hotspot shards and switch times ($t_{\text{sw}}$), alongside Spearman rank correlation filtering ($|\rho_{\text{Spearman}}(f, j)| < 0.70$), quarantining leaky counters from causal comparisons.
5. **Check H — End-of-Trace Counterfactual Horizon & Tombstone Semantics**: Accumulates counterfactual credit to the end of the trace ($H = \infty$) and separates tombstone-reclaiming `refresh` from append-compacting `merge` under deletions.

---

## Repository Layout

- `hygiene/` — Standalone, dependency-free measurement-hygiene verifier (`audit_hygiene.py --self-test`).
- `engines/` — Unified multi-engine lifecycle harness (`window_harness.py`, `qbam_io.py`, `diskann_driver.cpp`, `backends/{faiss_ivf,hnswlib_mut,annoy_rebuild,usearch_mut}.py`, plus `quake` and `diskann`).
- `replay/` — Deterministic C++17 counterfactual replay simulator (`qbam_replay` + 8 core unit tests, CMake + `build.sh`).
- `rust/` — Zero-FFI Rust harness (`dynvec-bench`, `dynvec-audit`) covering `usearch_rust` and `hnsw_rs`.
- `scripts/` — Orchestration, pacing & orthogonal decomposition (`run_exp11a_pacing_generality.py`, `run_exp11d_adaptive_pacing.py`, `run_r3_exp1_orthogonal_decomp.py`, `run_r3_exp2_real_streams_cost_phase.py`), live 16-shard engine bridges (`run_llm01_exp11b_paced_bridge.py`, `run_mimo_supplement_real_engines.py`), bare-metal 768-d & floor calibration (`run_llm01_exp11_b1_parallel.py`, `run_mimo_exp11e_f_quake_glove.py`, `analyze_floor_sensitivity.py`), Ranking-Flip Matrix (`compute_flip_matrix.py`), hygiene recleaning (`reclean_and_summarize.py`), and paper table/figure builders (`build_paper_tables.py`, `build_story_tables.py`, `generate_paper_figures.py`, `generate_story_figures.py`, `build_fig1_framework.py`).
- `requirements.txt` — Python dependencies for engines, predictors, MILP optimization, and plotting.
- `reproduce.sh` — End-to-end self-test, C++ simulator build, syntax verification, and paper artifact reproduction entrypoint.

---

## Quick Start

### 1. Python Environment & Hygiene Self-Test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Zero-dependency audit verifier self-test
python3 hygiene/audit_hygiene.py --self-test
```

### 2. Build & Test the Counterfactual Replay Simulator (`qbam_replay`)

```bash
chmod +x replay/build.sh reproduce.sh
./replay/build.sh
# Runs CMake Release build (-O3) and executes all 8 deterministic CTest suites
```

To build a portable binary across x86-64 hosts:

```bash
QBAM_MARCH=x86-64-v3 ./replay/build.sh
```

---

## Core & NUMA Pinning Protocol (Check P Compliance)

Dynamic ANN timings and tail latencies ($p_{95}, p_{99}$) are sensitive to hyper-thread sibling sharing, cross-socket NUMA traffic, and multi-thread runtime oversubscription. All citable CPU timing runs **must** obey:

1. **Disable multi-threaded BLAS/OpenMP pools** (except explicit multi-threaded concurrency arms):
   ```bash
   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1
   ```
2. **Pin each worker to an isolated physical core on a single NUMA node** (avoiding hyper-thread siblings; inspect topology via `lscpu -e` or `numactl --hardware`):
   ```bash
   numactl --cpunodebind=0 --membind=0 \
     taskset -c 0 nice -n 19 \
     python3 engines/window_harness.py \
       --system faiss_ivf \
       --trace abrupt_s05 \
       --data-root data/sift100k_u10_q2500_v2 \
       --pin-cpu 0 \
       --bl-args '{"nlist":256,"nprobe":16,"retrain_every":0}' \
       --out results/faiss_np16_abrupt_s05.jsonl
   ```
3. **Audit every completed JSONL log immediately**:
   ```bash
   python3 hygiene/audit_hygiene.py \
     --audit-jsonl results/faiss_np16_abrupt_s05.jsonl \
     --max-ratio 1.18 \
     --max-load 2.0
   ```

---

## Reproduction Commands

### Paper Tables, Figures, and Decomposition Analyses

```bash
# Exact 2x2 orthogonal decomposition (Where / When / Foresight) & real-stream/phase-boundary replay
python3 scripts/run_r3_exp1_orthogonal_decomp.py
python3 scripts/run_r3_exp2_real_streams_cost_phase.py
python3 scripts/analyze_floor_sensitivity.py

# Regenerate unified flip matrix, protocol re-audits, LaTeX tables/macros, and paper figures
python3 scripts/compute_flip_matrix.py
python3 scripts/compute_protocol_reaudit_and_pareto.py
python3 scripts/build_paper_tables.py
python3 scripts/build_story_tables.py
python3 scripts/generate_paper_figures.py
python3 scripts/generate_story_figures.py
python3 scripts/build_fig1_framework.py
```

### Multi-Engine Lifecycle & Live Pacing Bridges

```bash
# Six-engine lifecycle benchmark (FAISS-IVF, hnswlib, FreshDiskANN, Quake, Annoy, USearch)
OMP_NUM_THREADS=1 python3 scripts/run_exp5_multi_engine_audit.py --pin-cpu 8 --max-load 2.0

# Dual-socket Xeon real-engine placement, cadence, CRUD, and floor suites
OMP_NUM_THREADS=1 python3 scripts/run_llm01_real_engine_traps.py --suite ALL

# Live 16-shard paced scheduling bridge (FAISS-IVF, USearch, hnswlib)
OMP_NUM_THREADS=1 python3 scripts/run_llm01_exp11b_paced_bridge.py

# Bare-metal 768-d Wikipedia (140K-1M) and floor-calibrated Quake / GloVe-100 runs
OMP_NUM_THREADS=1 python3 scripts/run_llm01_exp11_b1_parallel.py
OMP_NUM_THREADS=1 python3 scripts/run_mimo_exp11e_f_quake_glove.py --mode exp11e_quake
```

### Counterfactual Replay & Pacing Generality (2,400 Traces)

```bash
# Pacing generality & adaptive pacing rules across 6 drift families x 2 couplings x R in {16, 64}
python3 scripts/run_exp11a_pacing_generality.py
python3 scripts/run_exp11d_adaptive_pacing.py

# Predictability spectrum, clock-leak audit, and executed cadence brackets
OMP_NUM_THREADS=1 python3 scripts/run_exp2_predictors_and_cadence.py \
  --data-root data/sift100k_u10_q2500_v2 \
  --binary ./replay/build/qbam_replay \
  --cores 0,2,4,6,8,10,12,14

# Sprint v2 suites (canon25, long seasonal, 1M scale) & review audits (A/B/C)
OMP_NUM_THREADS=1 python3 scripts/run_mimo_sprint_v2.py --suite all --binary ./replay/build/qbam_replay
python3 scripts/run_heldout_schedule_audit.py
python3 scripts/run_milp_and_horizon_sensitivity.py
```

---

## License

Released under the [MIT License](LICENSE).
