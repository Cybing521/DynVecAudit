#!/usr/bin/env python3
"""Exp 11-B1-ext: missing floor-calibrated graph knobs on bare-metal 768-d Wikipedia.

Runs on isolated NUMA-1 physical cores (idle HT siblings), reusing Exp 10-B1's worker
in run_llm01_exp10b_full_audit.py so all metrics/Gate-4 checks match Exp 11-B1.
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

WLS = [
    ("real768_500k_drift", "500K_Drift"),
    ("real768_500k_crud20", "500K_CRUD20"),
    ("real768_1m_drift", "1M_Drift"),
]
ENGINES = [
    ("faiss_hnsw", "ef128", {"M": 16, "ef_c": 100, "ef": 128}),
    ("usearch", "ef128", {"M": 16, "ef_c": 100, "ef": 128}),
    ("diskann", "L128", {"R": 32, "Lbuild": 100, "search_L": 128}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", default="52,56,58,60")
    ap.add_argument("--out-dir", default=str(HERE / "results_exp11b1_ext"))
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    jobs = []
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
    cores = [c.strip() for c in args.cores.split(",") if c.strip()]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running = {}
    log = (out / "exp11b1_ext.log").open("a", encoding="utf-8")
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
        with (out / "exp11b1_ext_graph_knobs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    (out / "EXP11B1_EXT_DONE.flag").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
