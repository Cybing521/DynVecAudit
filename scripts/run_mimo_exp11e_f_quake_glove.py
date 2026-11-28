#!/usr/bin/env python3
"""Exp 11-E (Quake floor-calibrated targets) + Exp 11-F (GloVe-100 high-knob check) on Ryzen-9950X Core-8.

Fixed schema ('mean_recall' fallback), fixed GloVe path ('glove100_u10_q2500'),
and adds co-scheduled L2-GEMV probe logging (isolated Core-8 baseline = 5.973 ms).
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
from typing import Any, Dict, List

import numpy as np

WORK_DIR = Path(os.environ.get("DYNVECAUDIT_ROOT", str(Path(__file__).resolve().parents[1])))
EXT_SRC = Path(os.environ.get("DYNVECAUDIT_ENGINES_DIR", str(Path(__file__).resolve().parents[1] / "engines")))
DATA_BASE = Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data"))
BASE_PROBE_MS = 5.973


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_load1m() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def wait_quiet_host(max_load: float = 2.0, min_checks: int = 2, sleep_s: float = 10.0) -> None:
    ok = 0
    while ok < min_checks:
        ld = read_load1m()
        if ld <= max_load:
            ok += 1
            if ok >= min_checks:
                return
        else:
            ok = 0
        time.sleep(sleep_s)


def run_micro_probe_ms(iters: int = 160) -> float:
    rng = np.random.default_rng(12345)
    a = rng.standard_normal((256, 128), dtype=np.float32)
    b = rng.standard_normal((128, 128), dtype=np.float32)
    _ = a @ b
    t0 = time.perf_counter_ns()
    acc = 0.0
    for _ in range(iters):
        c = a @ b
        acc += float(c[0, 0])
    if acc == 1e30:
        print("", end="")
    return (time.perf_counter_ns() - t0) / 1e6


def summarize_jsonl(jsonl_path: Path, wall_s: float) -> Dict[str, Any]:
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    hdr = next((r for r in rows if r.get("type") in ("run", "header")), {})
    wins = [r for r in rows if r.get("type") == "window"]
    recs = [float(w.get("mean_recall", w.get("recall_at_10", 0.0))) for w in wins]
    p99s = [float(w["p99_us"]) for w in wins if float(w.get("p99_us", 0)) > 0]
    q_ms = sum(float(w.get("query_cpu_ms", 0)) for w in wins)
    m_ms = sum(float(w.get("maint_cpu_ms", 0)) for w in wins)
    b_ms = float(hdr.get("build_cpu_ms", 0.0) or 0.0)
    tot_ms = q_ms + m_ms
    w_c = wall_s / max((tot_ms + b_ms) / 1000.0, 1e-6)
    att = 100.0 * sum(1 for r in recs if r >= 0.965) / max(len(recs), 1)
    n_w = len(recs)
    t1, t2 = max(1, n_w // 3), max(2, (2 * n_w) // 3)
    return {
        "n_windows": n_w,
        "build_cpu_ms": round(b_ms, 2),
        "query_cpu_ms": round(q_ms, 2),
        "maint_cpu_ms": round(m_ms, 2),
        "total_cpu_ms": round(tot_ms, 2),
        "early_recall": round(sum(recs[:t1]) / max(len(recs[:t1]), 1), 5) if recs else 0.0,
        "mid_recall": round(sum(recs[t1:t2]) / max(len(recs[t1:t2]), 1), 5) if recs else 0.0,
        "late_recall": round(sum(recs[t2:]) / max(len(recs[t2:]), 1), 5) if recs else 0.0,
        "mean_recall": round(sum(recs) / max(n_w, 1), 5),
        "min_recall": round(min(recs) if recs else 0.0, 5),
        "att_0965_pct": round(att, 1),
        "mean_p99_us": round(sum(p99s) / max(len(p99s), 1), 2) if p99s else -1.0,
        "peak_rss_mb": float(hdr.get("peak_rss_mb", 0.0) or 0.0),
        "wall_s": round(wall_s, 3),
        "wall_cpu_ratio": round(w_c, 4),
        "loadavg_end": round(read_load1m(), 2),
        "status": "ok" if wins else "empty",
    }


def build_jobs(mode: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    if mode == "exp11e_quake":
        drifts = ["abrupt", "gradual", "hotspot", "periodic", "uniform"]
        seeds = ["s05", "s06"]
        targets = [("rt098", 0.98), ("rt099", 0.99), ("rt0995", 0.995)]
        for d in drifts:
            for s in seeds:
                tr = f"{d}_{s}"
                if not (DATA_BASE / "sift100k_u10_q2500_v2" / "traces" / tr / "gt.bin").exists():
                    continue
                for ptag, rt in targets:
                    jobs.append({
                        "suite": "exp11e_quake_calib", "workload": "sift100k_u10",
                        "data_root": DATA_BASE / "sift100k_u10_q2500_v2", "trace": tr,
                        "system": "quake", "param_tag": ptag,
                        "bl": {"nlist": 256, "recall_target": rt},
                    })
    elif mode == "exp11f_glove":
        traces = ["abrupt_s00", "hotspot_s00", "gradual_s00"]
        engines = [
            ("faiss_ivf", "np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}),
            ("faiss_ivf", "np64", {"nlist": 256, "nprobe": 64, "retrain_every": 0}),
            ("faiss_ivf", "np96", {"nlist": 256, "nprobe": 96, "retrain_every": 0}),
            ("faiss_ivf", "np128", {"nlist": 256, "nprobe": 128, "retrain_every": 0}),
            ("hnswlib_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}),
            ("hnswlib_mut", "ef256", {"ef": 256, "M": 16, "ef_construction": 200}),
            ("hnswlib_mut", "ef512", {"ef": 512, "M": 16, "ef_construction": 200}),
            ("usearch_mut", "ef256", {"ef": 256, "M": 16, "ef_construction": 200}),
            ("diskann", "L128", {"search_L": 128, "R": 32, "Lbuild": 100}),
            ("diskann", "L192", {"search_L": 192, "R": 32, "Lbuild": 100}),
        ]
        for tr in traces:
            if not (DATA_BASE / "glove100_u10_q2500" / "traces" / tr / "gt.bin").exists():
                continue
            for sys_n, ptag, bl in engines:
                jobs.append({
                    "suite": "exp11f_glove_calib", "workload": "glove100",
                    "data_root": DATA_BASE / "glove100_u10_q2500", "trace": tr,
                    "system": sys_n, "param_tag": ptag, "bl": bl,
                })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exp11e_quake", "exp11f_glove"], required=True)
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--max-load", type=float, default=2.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = WORK_DIR / "results" / ("exp11e_quake_calib" if args.mode == "exp11e_quake" else "exp11f_glove_calib")
    raw_dir = out_dir / "raw_jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_name = "exp11e_quake_clean" if args.mode == "exp11e_quake" else "exp11f_glove_clean"

    if args.merge:
        rows = []
        for p in sorted(out_dir.glob(f"{out_name}_shard*.csv")):
            with p.open(encoding="utf-8") as f:
                rows.extend(list(csv.DictReader(f)))
        if rows:
            with (out_dir / f"{out_name}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        flag = "EXP11E_DONE.flag" if args.mode == "exp11e_quake" else "EXP11F_DONE.flag"
        (out_dir / flag).write_text("ok\n", encoding="utf-8")
        log(f"MERGED {len(rows)} rows -> {out_dir / f'{out_name}.csv'}")
        return 0

    all_jobs = build_jobs(args.mode)
    jobs = [j for idx, j in enumerate(all_jobs) if idx % args.nshards == args.shard]
    wh_script = WORK_DIR / "external_harness" / "window_harness.py"
    pydeps = WORK_DIR / "pydeps"
    quake_lib = str(EXT_SRC / "build" / "quake_pkg" / "quake")
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    env = os.environ.copy()
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
    py_exe = os.environ.get("DYNVECAUDIT_PYTHON", sys.executable)
    rows = []
    for i, job in enumerate(jobs, 1):
        wait_quiet_host(max_load=args.max_load)
        jsonl = raw_dir / f"{job['workload']}__{job['trace']}__{job['system']}__{job['param_tag']}_clean.jsonl"
        cmd = [
            "taskset", "-c", str(args.core), py_exe, str(wh_script),
            "--system", job["system"], "--trace", job["trace"],
            "--data-root", str(job["data_root"]), "--bl-args", json.dumps(job["bl"]),
            "--pin-cpu", str(args.core), "--force", "--out", str(jsonl),
        ]
        if job["system"] == "quake":
            cmd += ["--recall-target", str(job["bl"]["recall_target"])]
        elif job["system"] == "diskann":
            cmd += ["--R", str(job["bl"]["R"]), "--Lbuild", str(job["bl"]["Lbuild"]),
                    "--search-L", str(job["bl"]["search_L"]), "--build-threads", "1"]

        p0 = run_micro_probe_ms()
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
        p1 = run_micro_probe_ms()
        probe_mean = round(0.5 * (p0 + p1), 3)
        probe_sd = round(probe_mean / BASE_PROBE_MS, 4)

        if proc.returncode != 0 or not jsonl.exists():
            log(f"[{i}/{len(jobs)}] FAIL {jsonl.name}: {proc.stderr[-300:]}")
            continue
        summ = summarize_jsonl(jsonl, wall_s)
        flagged = bool(summ["wall_cpu_ratio"] > 1.18 or probe_sd > 1.25)
        row = {
            "suite": job["suite"], "workload": job["workload"], "trace": job["trace"],
            "system": job["system"], "param_tag": job["param_tag"], **summ,
            "probe_ms_mean": probe_mean, "probe_slowdown": probe_sd,
            "gate4_flagged": flagged,
        }
        rows.append(row)
        shard_csv = out_dir / f"{out_name}_shard{args.shard}.csv"
        with shard_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"[{i}/{len(jobs)}] OK {jsonl.name}: cpu={row['total_cpu_ms']/1000:.2f}s rec={row['mean_recall']} att={row['att_0965_pct']}% w/C={row['wall_cpu_ratio']} probe={probe_sd}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
