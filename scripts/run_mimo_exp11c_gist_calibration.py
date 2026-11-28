#!/usr/bin/env python3
"""Exp 10-A (Ryzen-9950X Core-8 Pure-CPU, CUDA_VISIBLE_DEVICES=""):
1. 100K CRUD-30% & CRUD-50% Recall-Floor-Matched (>=95%) Sweep + Early/Mid/Late 5-pt Recall-QPS Pareto:
   - Workloads: sift100k_u10_q2500_crud30v2 & crud50v2 (abrupt_s05, hotspot_s05, gradual_s05)
   - Engines & >=5 knobs:
     * faiss_ivf: np in {16, 24, 32, 48, 64} (+ ret10 variant for np32/np48)
     * hnswlib_mut: ef in {64, 96, 128, 192, 256}
     * usearch_mut: ef in {64, 96, 128, 192, 256}
   Reports per-run & per-(Early W0-12 / Mid W13-26 / Late W27-39) Recall@10, QPS, updates/s, p99, CPU, Peak RSS.
2. Multi-Metric / High-Dim Bare-Metal Core-8 Audit (GIST-960d & GloVe-100d, 3 seeds each):
   - Engines: faiss_ivf (np32, np64), hnswlib_mut (ef64, ef128), usearch_mut (ef64, ef128), diskann (L64, L96)
Strictly Core 8 pinned, OMP=1, GPU disabled, load <= 2.5 gate.
"""
from __future__ import annotations

import os

for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_ev] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
EXT_SRC = Path(os.environ.get("DYNVECAUDIT_ENGINES_DIR", str(Path(__file__).resolve().parents[1] / "engines")))
VENV_PY = EXT_SRC / "venv" / "bin" / "python"
DATA_BASE = Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data"))
WORK_DIR = Path(os.environ.get("DYNVECAUDIT_ROOT", str(Path(__file__).resolve().parents[1])))
OUT_DIR = WORK_DIR / "results" / "exp11c_gist_calibration"
RAW_DIR = OUT_DIR / "raw_jsonl"

sys.path.insert(0, str(THIS_DIR))
import audit_hygiene as ah  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    except Exception:
        return 0.0


def wait_quiet_host(max_load: float = 2.5, min_checks: int = 2, sleep_s: float = 15.0) -> None:
    ok = 0
    while ok < min_checks:
        if read_load1m() <= max_load:
            ok += 1
            if ok < min_checks:
                time.sleep(3.0)
        else:
            ok = 0
            time.sleep(sleep_s)


def summarize_phases(jsonl_path: Path, hdr_wall_s: float) -> Dict[str, Any]:
    wins: List[Dict[str, Any]] = []
    hdr: Dict[str, Any] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "window":
            wins.append(obj)
        elif obj.get("type") == "run":
            hdr = obj

    ok_wins = [w for w in wins if w.get("window_status", "ok") == "ok"]
    eval_w = ok_wins if ok_wins else wins
    win_status = ",".join(sorted({str(w.get("window_status", "ok")) for w in wins}))

    q_cpu_ms = sum(float(w.get("query_cpu_ms", 0.0)) for w in eval_w)
    m_cpu_ms = sum(float(w.get("maint_cpu_ms", 0.0)) for w in eval_w)
    tot_cpu_ms = q_cpu_ms + m_cpu_ms
    build_cpu_ms = float(hdr.get("build_cpu_ms") or 0.0)
    n_q_tot = sum(int(w.get("n_queries", 0)) for w in eval_w)
    # Count dynamic CRUD ops (insert + update + erase if available, fallback n_insert)
    n_crud_tot = sum(
        int(w.get("n_insert", 0)) + int(w.get("n_update", 0)) + int(w.get("n_erase", 0))
        for w in eval_w
    )
    if n_crud_tot == 0:
        n_crud_tot = len(eval_w) * 10000

    recs = [float(w["mean_recall"]) for w in eval_w if w.get("mean_recall") is not None]
    p95s = [float(w["p95_us"]) for w in eval_w if w.get("p95_us") is not None]
    p99s = [float(w["p99_us"]) for w in eval_w if w.get("p99_us") is not None]
    att_0965 = sum(1 for r in recs if r >= 0.965)
    att_0950 = sum(1 for r in recs if r >= 0.950)

    def _phase_slice(lo: int, hi: int) -> Tuple[float, float, float]:
        sub = eval_w[lo:hi]
        if not sub:
            return 0.0, 0.0, 0.0
        mr = statistics.fmean(float(w["mean_recall"]) for w in sub)
        qc = sum(float(w.get("query_cpu_ms", 0.0)) for w in sub)
        nq = sum(int(w.get("n_queries", 0)) for w in sub)
        qps = nq / max(qc / 1000.0, 1e-6)
        p99 = statistics.fmean(float(w["p99_us"]) for w in sub if w.get("p99_us") is not None)
        return round(mr, 5), round(qps, 1), round(p99, 2)

    n = len(eval_w)
    e_rec, e_qps, e_p99 = _phase_slice(0, max(1, n // 3))
    m_rec, m_qps, m_p99 = _phase_slice(max(1, n // 3), max(2, (2 * n) // 3))
    l_rec, l_qps, l_p99 = _phase_slice(max(2, (2 * n) // 3), n)

    peak_rss = hdr.get("peak_rss_mb")
    if peak_rss is None and hdr.get("rss_hwm_kb"):
        peak_rss = round(float(hdr["rss_hwm_kb"]) / 1024.0, 2)
    if peak_rss is None:
        drv = jsonl_path.with_suffix(".driver.jsonl")
        if drv.exists():
            for line in drv.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                d_obj = json.loads(line)
                if d_obj.get("type") == "run" and d_obj.get("rss_hwm_kb"):
                    peak_rss = round(float(d_obj["rss_hwm_kb"]) / 1024.0, 2)
                    break

    overall_qps = n_q_tot / max(q_cpu_ms / 1000.0, 1e-6)
    updates_per_s = n_crud_tot / max(m_cpu_ms / 1000.0, 1e-6)
    tot_all_s = (tot_cpu_ms + build_cpu_ms) / 1000.0
    wc = round(hdr_wall_s / max(tot_all_s, 1e-6), 4)

    return {
        "n_windows": len(wins),
        "n_ok_windows": len(ok_wins),
        "window_status": win_status,
        "build_cpu_ms": round(build_cpu_ms, 2),
        "query_cpu_ms": round(q_cpu_ms, 2),
        "maint_cpu_ms": round(m_cpu_ms, 2),
        "total_cpu_ms": round(tot_cpu_ms, 2),
        "query_pct": round(100.0 * q_cpu_ms / max(tot_cpu_ms, 1e-6), 2),
        "maint_pct": round(100.0 * m_cpu_ms / max(tot_cpu_ms, 1e-6), 2),
        "mean_recall": round(statistics.fmean(recs), 5) if recs else 0.0,
        "min_recall": round(min(recs), 5) if recs else 0.0,
        "att_0965_pct": round(100.0 * att_0965 / max(len(recs), 1), 1),
        "att_0950_pct": round(100.0 * att_0950 / max(len(recs), 1), 1),
        "qps_1core": round(overall_qps, 1),
        "updates_per_s": round(updates_per_s, 1),
        "mean_p95_us": round(statistics.fmean(p95s), 2) if p95s else None,
        "mean_p99_us": round(statistics.fmean(p99s), 2) if p99s else None,
        "early_recall": e_rec,
        "early_qps": e_qps,
        "early_p99_us": e_p99,
        "mid_recall": m_rec,
        "mid_qps": m_qps,
        "mid_p99_us": m_p99,
        "late_recall": l_rec,
        "late_qps": l_qps,
        "late_p99_us": l_p99,
        "peak_rss_mb": peak_rss,
        "wall_s": round(hdr_wall_s, 3),
        "wall_cpu_ratio": wc,
        "loadavg_end": wins[-1].get("loadavg") if wins else None,
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_jobs() -> List[Dict[str, Any]]:
    """Exp 11-C: GIST-960 knob calibration. Exp 10-A ran graphs only at ef<=128 / L<=96, where
    late-window recall decayed (0.977 -> 0.878) and no graph met the floor. Here graphs get
    ef in {192, 256, 384, 512} and DiskANN L in {128, 192, 256}; IVF adds np48 between np32/np64."""
    jobs: List[Dict[str, Any]] = []
    gist_traces = ["abrupt_s05", "hotspot_s05", "gradual_s05"]
    engines = [
        ("faiss_ivf", "np48", {"nlist": 256, "nprobe": 48, "retrain_every": 0}),
        ("hnswlib_mut", "ef192", {"ef": 192, "M": 16, "ef_construction": 200}),
        ("hnswlib_mut", "ef256", {"ef": 256, "M": 16, "ef_construction": 200}),
        ("hnswlib_mut", "ef384", {"ef": 384, "M": 16, "ef_construction": 200}),
        ("hnswlib_mut", "ef512", {"ef": 512, "M": 16, "ef_construction": 200}),
        ("usearch_mut", "ef192", {"ef": 192, "M": 16, "ef_construction": 200}),
        ("usearch_mut", "ef256", {"ef": 256, "M": 16, "ef_construction": 200}),
        ("usearch_mut", "ef384", {"ef": 384, "M": 16, "ef_construction": 200}),
        ("diskann", "L128", {"search_L": 128, "R": 32, "Lbuild": 100}),
        ("diskann", "L192", {"search_L": 192, "R": 32, "Lbuild": 100}),
        ("diskann", "L256", {"search_L": 256, "R": 32, "Lbuild": 100}),
    ]
    for tr in gist_traces:
        if not (DATA_BASE / "gist960_u10" / "traces" / tr / "gt.bin").exists():
            continue
        for sys_name, ptag, bl in engines:
            jobs.append({"suite": "gist_calibration", "workload": "gist960", "data_root": DATA_BASE / "gist960_u10", "trace": tr, "system": sys_name, "param_tag": ptag, "bl": bl})
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp 10-A Core-8 Pure-CPU Supplement on Ryzen-9950X")
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--max-load", type=float, default=2.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wh_script = WORK_DIR / "external_harness" / "window_harness.py"
    pydeps = WORK_DIR / "pydeps"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{pydeps}:{EXT_SRC / 'tools'}"
    for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[_ev] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""

    jobs = [j for i, j in enumerate(build_jobs()) if i % args.nshards == args.shard]
    log(f"Exp 10-A: queued {len(jobs)} Core-{args.core} jobs (CUDA disabled, max_load={args.max_load}).")
    out_csv = OUT_DIR / f"exp11c_summary_shard{args.shard}.csv"
    rows: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs, 1):
        stem = f"{job['workload']}__{job['trace']}__{job['system']}__{job['param_tag']}"
        out_jsonl = RAW_DIR / f"{stem}.jsonl"
        log_f = RAW_DIR / f"{stem}.log"
        if out_jsonl.exists() and out_jsonl.stat().st_size > 200:
            prev_wall = 0.0
            if log_f.exists():
                first_l = log_f.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
                if first_l and first_l[0].startswith("WALL_S="):
                    try:
                        prev_wall = float(first_l[0].split("=", 1)[1])
                    except ValueError:
                        prev_wall = 0.0
            s = summarize_phases(out_jsonl, hdr_wall_s=prev_wall)
            g4_prev = ah.check_contamination(out_jsonl, max_wall_cpu_ratio=1.18, max_load=args.max_load)
            is_flag_prev = (not g4_prev["passed"]) or (prev_wall > 0 and s["wall_cpu_ratio"] > 1.18)
            if not is_flag_prev and s["n_ok_windows"] >= 1:
                rows.append({
                    "idx": idx, "suite": job["suite"], "workload": job["workload"], "trace": job["trace"],
                    "system": job["system"], "param_tag": job["param_tag"], **s,
                    "gate4_flagged": False, "status": "ok",
                })
                write_csv(rows, out_csv)
                log(f"[{idx}/{len(jobs)}] RESUME-OK {stem}: cpu={s['total_cpu_ms']/1000:.2f}s rec={s['mean_recall']:.4f} att={s['att_0965_pct']}% w/c={s['wall_cpu_ratio']}")
                continue

        wait_quiet_host(args.max_load)
        extra_cli: List[str] = []
        if job["system"] == "diskann":
            extra_cli = [
                "--R", str(job["bl"].get("R", 32)),
                "--Lbuild", str(job["bl"].get("Lbuild", 100)),
                "--search-L", str(job["bl"].get("search_L", 64)),
                "--build-threads", "1",
            ]
        cmd = [
            "taskset", "-c", str(args.core), "nice", "-n", "19",
            str(VENV_PY), str(wh_script),
            "--system", job["system"],
            "--trace", job["trace"],
            "--data-root", str(job["data_root"]),
            "--bl-args", json.dumps(job["bl"]),
            *extra_cli,
            "--pin-cpu", str(args.core),
            "--force",
            "--out", str(out_jsonl),
        ]
        t0 = time.perf_counter()
        pr = subprocess.run(cmd, cwd=str(EXT_SRC), env=env, capture_output=True, text=True, timeout=3600)
        wall_s = time.perf_counter() - t0
        log_f.write_text(f"WALL_S={wall_s:.3f}\n" + pr.stdout + "\n--- STDERR ---\n" + pr.stderr, encoding="utf-8")
        if pr.returncode != 0 or not out_jsonl.exists():
            log(f"[{idx}/{len(jobs)}] FAIL {stem} rc={pr.returncode}: {pr.stderr[-220:]}")
            continue

        s = summarize_phases(out_jsonl, wall_s)
        g4 = ah.check_contamination(out_jsonl, max_wall_cpu_ratio=1.18, max_load=args.max_load)
        is_flagged = (not g4["passed"]) or (s["wall_cpu_ratio"] > 1.18)
        rows.append({
            "idx": idx,
            "suite": job["suite"],
            "workload": job["workload"],
            "trace": job["trace"],
            "system": job["system"],
            "param_tag": job["param_tag"],
            **s,
            "gate4_flagged": is_flagged,
            "status": "ok" if not is_flagged else "dirty_gate4",
        })
        write_csv(rows, out_csv)
        log(f"[{idx}/{len(jobs)}] OK {stem}: cpu={s['total_cpu_ms']/1000:.2f}s rec={s['mean_recall']:.4f} att={s['att_0965_pct']}% qps={s['qps_1core']} upd/s={s['updates_per_s']} w/c={s['wall_cpu_ratio']}")

    (OUT_DIR / f"EXP11C_DONE_shard{args.shard}.flag").write_text("ok\n", encoding="utf-8")
    log("EXP11C_SHARD_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
