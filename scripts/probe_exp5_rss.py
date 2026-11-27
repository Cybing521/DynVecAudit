#!/usr/bin/env python3
"""Lightweight Peak RSS probe for the 5 Python-harness engines in Exp 5A.

Waits passively until `run_exp2_predictors_and_cadence.py` completes and host
1-minute loadavg <= 1.5, then runs 1 representative trace (`abrupt_s05` on
`sift100k_u10_q2500_v2`) across the 9 Python-harness configurations on an
isolated core to collect `peak_rss_mb` without perturbing active Core-8 jobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from run_exp5_multi_engine_audit import (  # noqa: E402
    DATA_BASE,
    EXT_SRC,
    VENV_PY,
    prepare_overlay_harness,
    read_load1m,
    summarize_jsonl,
)


def _exp2_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "run_exp2_predictors_and_cadence.py"], capture_output=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Peak RSS for Exp 5A Python-harness engines")
    ap.add_argument("--work-dir", default="/workspace/pvldb")
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--max-load", type=float, default=1.5)
    ap.add_argument("--no-wait-exp2", action="store_true")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = work_dir / "results" / "exp5_multi_engine"
    probe_dir = out_dir / "rss_probe_runs"
    probe_dir.mkdir(parents=True, exist_ok=True)
    wh_script = prepare_overlay_harness(work_dir)

    if not args.no_wait_exp2:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for run_exp2_predictors_and_cadence.py to finish...", flush=True)
        while _exp2_running():
            time.sleep(60)

    while read_load1m() > args.max_load:
        time.sleep(20)

    pydeps = work_dir / "pydeps"
    env = os.environ.copy()
    quake_lib = str(EXT_SRC / "build" / "quake_pkg" / "quake")
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "QUAKE_DISABLE_AFFINITY": "1",
        "QBAM_EXTERNAL_ROOT": str(EXT_SRC),
        "LD_LIBRARY_PATH": f"{quake_lib}:{old_ld}" if old_ld else quake_lib,
        "PYTHONPATH": f"{pydeps}:{EXT_SRC / 'tools'}:{EXT_SRC / 'build' / 'quake_pkg'}",
    })

    data_root = DATA_BASE / "sift100k_u10_q2500_v2"
    trace = "abrupt_s05"
    specs = [
        ("faiss_ivf", "np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}, []),
        ("faiss_ivf", "np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}, []),
        ("hnswlib_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}, []),
        ("hnswlib_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}, []),
        ("quake", "rt097", {"nlist": 256, "recall_target": 0.97}, ["--nlist", "256", "--recall-target", "0.97"]),
        ("annoy_rebuild", "sk8000_re4", {"search_k": 8000, "n_trees": 5, "rebuild_every": 4}, []),
        ("annoy_rebuild", "sk8000_re1", {"search_k": 8000, "n_trees": 5, "rebuild_every": 1}, []),
        ("usearch_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}, []),
        ("usearch_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}, []),
    ]

    rows = []
    for sys_name, ptag, bl, extra_cli in specs:
        while read_load1m() > args.max_load:
            time.sleep(15)
        out_jsonl = probe_dir / f"{trace}__{sys_name}_{ptag}.jsonl"
        cmd = [
            "taskset", "-c", str(args.core), "nice", "-n", "19",
            str(VENV_PY), str(wh_script),
            "--system", sys_name,
            "--trace", trace,
            "--data-root", str(data_root),
            "--bl-args", json.dumps(bl),
            *extra_cli,
            "--pin-cpu", str(args.core),
            "--force",
            "--out", str(out_jsonl),
        ]
        proc = subprocess.run(cmd, cwd=str(EXT_SRC), env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[FAIL] {sys_name}_{ptag}: {proc.stderr[-200:]}", flush=True)
            continue
        s = summarize_jsonl(out_jsonl)
        rows.append({
            "workload": "sift128_drift",
            "trace": trace,
            "system": sys_name,
            "param_tag": ptag,
            "peak_rss_mb": s["peak_rss_mb"],
            "total_cpu_ms": s["total_cpu_ms"],
            "mean_recall": s["mean_recall"],
            "loadavg_end": s["loadavg_end"],
        })
        print(f"[OK] {sys_name}_{ptag} peak_rss_mb={s['peak_rss_mb']}MB", flush=True)

    if rows:
        csv_path = out_dir / "exp5_rss_probe.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[DONE] Wrote {len(rows)} rows -> {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
