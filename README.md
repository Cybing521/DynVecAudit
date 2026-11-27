# DynVecAudit

Official reproducibility artifact for the **PVLDB Volume 20 (VLDB 2027)** Experiment, Analysis & Benchmark (EA&B) submission:

> **When Dynamic Vector Index Benchmarks Mislead: Measurement Hygiene, Predictability Limits, and Cross-Engine Maintenance Auditing [Experiment, Analysis & Benchmark]**

---

## Overview

**DynVecAudit** is an open-source measurement-hygiene framework, 8-backend dynamic lifecycle harness (`7` codebases), and deterministic counterfactual replay suite for dynamic approximate nearest neighbor (ANN) vector indexes under shifting workloads, mixed CRUD/runbook churn, and real-timestamp streams.

### Four Quantitative Measurement-Hygiene Gates (`hygiene/audit_hygiene.py`)

1. **Gate 1 — Executed Cadence Bracket (`check_cadence_bracket`)**: Requires every budgeted policy spend $M(\pi)$ to be bracketed between two *executed* adjacent integer cadences ($M(c{+}1) \le M(\pi) \le M(c)$ for $c \in \{1..8\}$ or `noop`), preventing linear-interpolation illusions across non-linear cost frontiers.
2. **Gate 2 — Split & Clock Leakage (`check_clock_leak`)**: Enforces two-part admission requiring held-out switch-schedule ($t_{\text{sw}}$) train/test splits alongside Spearman rank correlation filtering ($|\rho_{\text{Spearman}}(f, j)| < 0.85$), quarantining leaky predictors from causal comparisons.
3. **Gate 3 — Counterfactual Attribution & Tombstone Semantics**: Evaluates counterfactual actions over full multi-window horizons ($H=4$ blocks) rather than truncated 1-block snapshots, and isolates tombstone-reclaiming `refresh` from append-compacting `merge` under deletions.
4. **Gate 4 — Contention & Cross-NUMA Isolation (`check_contamination`)**: Enforces single-thread execution (`OMP_NUM_THREADS=1`), physical core pinning within one NUMA node, zero cross-batch pooling, and $\text{wall}/\text{CPU} \le 1.18$ with paired-bootstrap confidence intervals (`paired_bootstrap_ci`).

---

## Repository Layout

- `hygiene/` — Standalone, dependency-free four-gate measurement-hygiene verifier (`audit_hygiene.py --self-test`).
- `engines/` — Unified six-engine lifecycle harness (`window_harness.py`, `qbam_io.py`, `diskann_driver.cpp`, `backends/{faiss_ivf,hnswlib_mut,annoy_rebuild,usearch_mut}.py`, plus built-in `quake` and `diskann`).
- `replay/` — Deterministic C++17 counterfactual replay simulator (`qbam_replay` + 8 core unit tests, CMake + `build.sh`).
- `scripts/` — Canonical orchestration, Ranking-Flip Matrix (`compute_flip_matrix.py`), statistical verification (`analyze_p2a_global_dp_and_stats.py`), hygiene recleaning (`reclean_and_summarize.py`), and paper table/figure builders (`build_paper_tables.py`, `generate_paper_figures.py`, `build_fig1_framework.py`).
- `requirements.txt` — Python dependencies for engines, predictors, MILP optimization, and plotting.
- `reproduce.sh` — End-to-end self-test and C++ simulator build entrypoint.

---

## Quick Start

### 1. Python Environment & Hygiene Self-Test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Zero-dependency 4-gate verifier self-test
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

## Core & NUMA Pinning Protocol (Gate 4 Compliance)

Dynamic ANN timings and tail latencies ($p_{95}, p_{99}$) are highly sensitive to hyper-thread sibling sharing, cross-socket NUMA traffic, and multi-thread runtime oversubscription. All citable CPU timing runs **must** obey:

1. **Disable multi-threaded BLAS/OpenMP pools** (except explicit multi-threaded scaling arms):
   ```bash
   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1
   ```
2. **Pin each worker to an isolated physical core on a single NUMA node** (avoiding hyper-thread siblings; inspect topology via `lscpu -e` or `numactl --hardware`):
   ```bash
   # Single-core NUMA-0 pinned execution (recommended for citable timings)
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
4. **Counterfactual replay with physical core pinning**:
   ```bash
   OMP_NUM_THREADS=1 taskset -c 8 nice -n 19 \
     ./replay/build/qbam_replay --config configs/abrupt_s05_past_qmass.json --out out.jsonl
   ```

---

## Reproduction Commands

### Unified Ranking-Flip Matrix (80 Verified Suites)

```bash
python3 scripts/compute_flip_matrix.py
# Emits results/paper_tables/ranking_flip_matrix.{csv,json} (80 rows, 0 pending)
```

`compute_flip_matrix.py` computes pairwise flip percentage, Kendall $\tau$, and Top-1 winner flip counts across **80 audited benchmark suites** spanning four source groups (`local_qbam`: 49 rows, `exp6_real_engine`: 18 rows, `colab_1m_audit`: 7 rows, `exp7_gpu_1m_arxiv`: 6 rows), cleanly separating multi-thread **Concurrency Regimes** (`6` rows) from **Trap 4** contention/NUMA noise:

| Trap / Category | Source Group | Covered Suites & Comparisons | Rows |
| :--- | :--- | :--- | ---: |
| **Trap 1 (Interp. Baseline)** | `local_qbam` | `exp2_sift25`, `sift100k_canon25`, `wikimedia_210d`, pooled across all 56 traces (`151/4080 = 3.70%`, $\tau = 0.9278$) | 4 |
| **Trap 1 (Interp. Baseline)** | `exp6_real_engine` | `A2` real-engine executed cadence bracket across 5 engines (`exp6_a2_real_engine_cadence`, 7 configs) | 1 |
| **Trap 1 (Interp. Baseline)** | `exp7_gpu_1m_arxiv` | `B3` arXiv MiniLM-384 executed cadence bracket vs linear interpolation (`exp7_arxiv_minilm384_exact_ops_and_dp_trap1_cadence`) | 1 |
| **Trap 2 (Split & Clock Leakage)** | `local_qbam` | `canon25`, `exp2_575`, `leakage_2x2` (`fixedhub`, `randhub`), `crossmodal` (`rag384`, `clip512`), `clinc150`, `wikimedia_210d`, plus `core_canon_leakage_exp2` & all-local pooled (`exact_scan_ops` & `total_cpu_ms`) | 19 |
| **Trap 2 (Split & Clock Leakage)** | `exp7_gpu_1m_arxiv` | `B3` arXiv MiniLM-384 leaky vs causal predictors under exact-ops & uncapped regimes (`1/1` top-1 flip uncapped) | 2 |
| **Trap 3 (CF Horizon & Tombstone)** | `local_qbam` | `track_cf` (`crud30_abrupt_s05`, `crud50_abrupt_s05`, `u10_abrupt_s05`, `u10_gradual_s06`, `u10_uniform_s05`, pooled `20.33%`) and `tombstone_sweep` (`crud00`–`crud50`, pooled `7.78%`) | 12 |
| **Trap 3 (CF Horizon & Tombstone)** | `exp6_real_engine` | `A3` CRUD deletion & tombstone sweep (`exp6_a3_crud_cpu_vs_attainment`, `60.0%` pairwise, `6/6` top-1 flip) | 1 |
| **Trap 4 (Contention & NUMA)** | `local_qbam` | `dirty_vs_clean` (`phase_uxq_15grids`, `abrupt_tsw_3grids`, pooled `82/504 = 16.27%`), dual-socket Xeon (`ht_sibling_2w`, `same_socket_4w/8w`, `cross_socket_16w`, pooled CPU & $p_{99}$), and `wikimedia_cpu_vs_ops` | 14 |
| **Trap 4 (Contention & NUMA)** | `exp6_real_engine` | `A1+D` co-location (`ht_sibling_2w`, `same_numa_4w`, `cross_numa_8w` across CPU/wall/$p_{99}$) + `A4` unmatched Recall@10 floor (`63.24%`) | 10 |
| **Trap 4 (Contention & NUMA)** | `colab_1m_audit` | T4 semantic/SIFT-1M VM SMT inflation and floor-omission splits across `140K`–`1M` | 7 |
| **Trap 4 (Contention & NUMA)** | `exp7_gpu_1m_arxiv` | `B3` `arxiv_cs_minilm384` (`40.0%`, `1/1` top-1) & `B1` `sift1m_crud30` (`57.14%`) / `sift1m_drift` (`58.33%`) CPU-only vs Recall@10 attainment ranking | 3 |
| **Concurrency Regime** | `exp6_real_engine` | Multi-thread deployment comparisons (`mt4_vs_iso1t`, `mt8_vs_iso1t`, `mt8_vs_mt4` on CPU/wall) separated from noise traps | 6 |

### Multi-Engine Lifecycle Audit (6 Open-Source Engines)

```bash
# Single-core NUMA-pinned lifecycle benchmark across FAISS-IVF, hnswlib,
# FreshDiskANN, Quake, Annoy, and USearch
OMP_NUM_THREADS=1 python3 scripts/run_exp5_multi_engine_audit.py --pin-cpu 8 --max-load 2.0

# Dual-socket Xeon real-engine traps (A1+D HT/NUMA/MT, A2 cadence, A3 CRUD, A4 floor)
OMP_NUM_THREADS=1 python3 scripts/run_llm01_real_engine_traps.py --suite ALL

# 1M-scale SIFT (B1) & real-timestamp arXiv MiniLM-384 (B3) suite
python3 scripts/run_gpu_1m_and_arxiv.py --phase all
```

### Extended Replay Suites (`sprint_v2`), Predictors & Cadence Brackets

```bash
# Predictability spectrum, clock-leak audit, and executed cadence brackets
OMP_NUM_THREADS=1 python3 scripts/run_exp2_predictors_and_cadence.py \
  --data-root data/sift100k_u10_q2500_v2 \
  --binary ./replay/build/qbam_replay \
  --cores 0,2,4,6,8,10,12,14

# Sprint v2 suites: canon25 (25 traces), p0_2_long_seasonal (horizon transfer),
# and sift1m_n12 (1M scale transfer)
OMP_NUM_THREADS=1 python3 scripts/run_mimo_sprint_v2.py --suite all --binary ./replay/build/qbam_replay
```

### Review Audits, Paper Tables, Statistical Tests & Figures

```bash
python3 scripts/run_heldout_schedule_audit.py          # Audit A: held-out t_sw Trap 2 check
python3 scripts/run_milp_and_horizon_sensitivity.py    # Audit B: Prop. 1 lambda & Track-CF Horizon H
python3 scripts/compute_protocol_reaudit_and_pareto.py # Audit C: protocol re-audit & QPS/CRUD Pareto
python3 scripts/build_paper_tables.py
python3 scripts/generate_paper_figures.py
python3 scripts/build_fig1_framework.py
```

---

## License

Released under the [MIT License](LICENSE).
