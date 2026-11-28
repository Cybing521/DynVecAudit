#!/usr/bin/env python3
"""Exp 11-B1: run the Exp 10-B1 bare-metal 768-d workloads in parallel on isolated NUMA-1 cores.

Re-uses the unchanged Exp 10-B1 worker (`--worker-json {"mode": "b1", ...}`) so results are
directly comparable; only the scheduling across cores differs (one single-threaded job per
physical core, HT sibling idle). Results go to a separate CSV so the serial Exp 10-B queue on
core 0 is not touched.
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

HERE = Path(__file__).resolve().parent
EXP10B = HERE / "run_llm01_exp10b_full_audit.py"
REAL768_WL = HERE / "real768_workloads"

WLS = [("real768_140k_drift", "140K_Drift"), ("real768_500k_drift", "500K_Drift"),
       ("real768_500k_crud20", "500K_CRUD20"), ("real768_1m_drift", "1M_Drift")]
ENGINES = [
    ("faiss_ivf", "np32", {"nlist": 512, "nprobe": 32}),
    ("faiss_ivf", "np64", {"nlist": 512, "nprobe": 64}),
    ("faiss_hnsw", "ef64", {"M": 16, "ef_c": 100, "ef": 64}),
    ("hnswlib", "ef64", {"M": 16, "ef_c": 100, "ef": 64}),
    ("hnswlib", "ef128", {"M": 16, "ef_c": 100, "ef": 128}),
    ("usearch", "ef64", {"M": 16, "ef_c": 100, "ef": 64}),
    ("diskann", "L64", {"R": 32, "Lbuild": 100, "search_L": 64}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", default="52,56,58,60")
    ap.add_argument("--out-dir", default=str(HERE / "results_exp11b1"))
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    jobs = []
    # seed-major order so every workload gets n=1 before n=3
    for sd in [int(s) for s in args.seeds.split(",")]:
        for prefix, wl in WLS:
            npz = f"{prefix}_s{sd}.npz"
            if not (REAL768_WL / npz).exists():
                continue
            for sysn, tag, prm in ENGINES:
                if wl == "500K_CRUD20" and sysn == "faiss_hnsw":
                    continue
                key = f"{wl}__s{sd}__{sysn}__{tag}"
                if (out / "cache" / f"{key}.json").exists():
                    continue
                jobs.append((key, {"mode": "b1", "npz_name": npz, "workload": wl, "seed": sd,
                                   "system": sysn, "param_tag": tag, "params": prm}))
    cores = [c for c in args.cores.split(",") if c]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running = {}
    log = (out / "exp11b1.log").open("a", encoding="utf-8")
    while jobs or running:
        for c in cores:
            if c not in running and jobs:
                key, cfg = jobs.pop(0)
                cmd = ["taskset", "-c", c, sys.executable, str(EXP10B), "--worker-json", json.dumps(cfg)]
                running[c] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env), key, time.time())
        time.sleep(5.0)
        for c in list(running):
            p, key, t0 = running[c]
            if p.poll() is None:
                if time.time() - t0 > 7200:
                    p.kill()
                    log.write(f"[{time.strftime('%F %T')}] TIMEOUT core={c} {key}\n")
                    del running[c]
                continue
            so, se = p.communicate()
            del running[c]
            res = next((json.loads(l.split("=", 1)[1]) for l in reversed(so.splitlines()) if l.startswith("WORKER_JSON=")), None)
            if res is None:
                log.write(f"[{time.strftime('%F %T')}] FAIL core={c} {key}: {se[-300:]}\n")
            else:
                res["core"] = c
                (out / "cache" / f"{key}.json").write_text(json.dumps(res) + "\n", encoding="utf-8")
                log.write(f"[{time.strftime('%F %T')}] core={c} {key} cpu={res.get('total_cpu_s')} rec={res.get('mean_recall')} att={res.get('att_0965_pct')} w/C={res.get('wall_cpu_ratio')}\n")
            log.flush()
    rows = [json.loads(p.read_text()) for p in sorted((out / "cache").glob("*.json"))]
    if rows:
        with (out / "exp11b1_baremetal_real768.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    (out / "EXP11B1_DONE.flag").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
