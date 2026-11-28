#!/usr/bin/env python3
"""Tier 1 #1: 5M/10M Recall-Matched `ef` Calibration on Xeon-8352V.

Single-build multi-ef sweeps (ef in {96, 192, 256}) for `usearch-rust` and `hnswlib`
across 5M and 10M BigANN Drift and CRUD-20% workloads (8 threads, cores 0-7, batch_query=True).
Designed to run cleanly after Suite S2 finishes via wait_and_run_s2_ef_calib.sh.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_llm01_rust_and_10m_scale import (  # noqa: E402
    BIGANN_SRC,
    BIGANN_WL_ROOT,
    RESULTS_DIR,
    log,
    run_py_bigann_list,
    run_rust_bench_list,
    wait_quiet_host,
    write_csv,
)


def main() -> int:
    log("=== Starting 5M/10M Recall-Matched ef Calibration (ef=96,192,256) ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "s2_5m_10m_ef_calibration.csv"
    rows: List[Dict[str, Any]] = []

    workloads = [
        ("5M", "drift", "bigann_5m_drift"),
        ("5M", "crud20", "bigann_5m_crud20"),
        ("10M", "drift", "bigann_10m_drift"),
        ("10M", "crud20", "bigann_10m_crud20"),
    ]
    ef_list = [96, 192, 256]

    for stag, wl_type, wl in workloads:
        # 1. USearch-Rust single-build multi-ef
        wait_quiet_host()
        log(f"[EF-Calib] Running {wl} usearch-rust ef={ef_list}...")
        r_list = run_rust_bench_list(
            mode="bigann",
            engine="usearch-rust",
            data_dir=BIGANN_WL_ROOT / wl,
            ef_list=ef_list,
            src_dir=BIGANN_SRC,
            ef_c=100,
            threads=8,
            cores="0-7",
            batch_query=True,
        )
        for r in r_list:
            r["suite"] = "S2-EF-Calib"
            r["scale"] = stag
            r["wl_type"] = wl_type
            r["config"] = f"ef{r['ef']}"
            rows.append(r)
            log(f"[EF-Calib] {wl} usearch_rust {r['config']}: build_w={r['build_wall_s']:.2f}s dyn_w={r['dyn_wall_s']:.2f}s rec={r['mean_recall']:.4f} att={r['att_0965_pct']}% p99={r['p99_us']}us")
        write_csv(rows, out_csv)

        # 2. hnswlib single-build multi-ef
        wait_quiet_host()
        knobs = [{"tag": f"ef{e}", "ef": e} for e in ef_list]
        log(f"[EF-Calib] Running {wl} hnswlib ef={ef_list}...")
        h_list = run_py_bigann_list(
            wl_name=wl,
            system="hnswlib",
            params={"M": 16, "ef_construction": 100},
            knobs=knobs,
            threads=8,
            cores="0-7",
            batch_query=True,
        )
        for r in h_list:
            r["suite"] = "S2-EF-Calib"
            r["scale"] = stag
            r["wl_type"] = wl_type
            rows.append(r)
            log(f"[EF-Calib] {wl} hnswlib {r['config']}: build_w={r['build_wall_s']:.2f}s dyn_w={r['dyn_wall_s']:.2f}s rec={r['mean_recall']:.4f} att={r['att_0965_pct']}% p99={r['p99_us']}us")
        write_csv(rows, out_csv)

    log("ALL 5M/10M ef calibration runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
