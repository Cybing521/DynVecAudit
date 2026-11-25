# DynVecAudit

Official reproducibility repository for the PVLDB Volume 20 (VLDB 2027) Experiments, Analysis & Benchmarks (EA&B) submission:

> **When Dynamic Vector Index Benchmarks Mislead: Measurement Hygiene, Predictability Limits, and Cross-Engine Maintenance Auditing [Experiment, Analysis & Benchmark]**

---

## Overview

**DynVecAudit** is an open-source measurement-hygiene framework and cross-engine lifecycle benchmark for dynamic approximate nearest neighbor (ANN) vector indexes under shifting workloads, CRUD churn, and real-timestamp streams.

### Four Quantitative Measurement-Hygiene Gates

1. **Executed Cadence Bracket Gate**: Requires bounding budgeted maintenance policies against executed adjacent integer cadences rather than linearly interpolated sparse baselines.
2. **Clock/Step Leakage Gate**: Detects monotonic insert-counter and step-index leakage in learned maintenance schedulers (`|rho| >= 0.85`).
3. **Counterfactual Attribution Gate**: Separates in-block savings from full-trace amortized counterfactual accounting and isolates tombstone-reclaiming `refresh` from `merge`.
4. **Contention & Cross-NUMA Gate**: Audits single-core/single-socket pinning and quantifies CPU and tail-latency (`p99`) inflation under parallel and cross-NUMA contention.

### Audited Vector Search Engines

DynVecAudit evaluates six open-source vector search engines under one unified, leakage-free harness alongside deterministic counterfactual replay across 128-d to 512-d embeddings, mixed CRUD/runbook churn, and real-timestamp streams:

- `FAISS-IVF`
- `hnswlib`
- `FreshDiskANN`
- `Quake`
- `Annoy`
- `USearch`

---

## Planned Artifact Layout

- `hygiene/` — Four-gate measurement-hygiene audit suite and contamination detectors
- `engines/` — Unified six-engine lifecycle wrappers and recall-matched evaluation runners
- `workloads/` — Synthetic drift, CRUD/runbook, and real-timestamp stream trace builders
- `scripts/` — End-to-end reproduction, statistical auditing, and figure/table generation scripts
- `docs/` — Hardware/NUMA pinning instructions and step-by-step replication guide

---

## Release Status

This public repository hosts the official **DynVecAudit** reproducibility artifact. Complete benchmark implementations, workload generation scripts, and step-by-step replication instructions are being finalized here prior to the **December 1, 2026** PVLDB Volume 20 full-paper submission deadline.

## License

Released under the [MIT License](LICENSE).
