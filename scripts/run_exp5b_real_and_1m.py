#!/usr/bin/env python3
"""Exp 5B: 6-Engine Open-Source Evaluation on Real-Timestamp Wikimedia & 1M Scale.

Reuses the non-invasive overlay harness at `engines/window_harness.py`
on idle `Ryzen-9950X` (Core 8 pinned, nice -n 19, load-gated <= 2.2, 2400s per-run timeout).

Suites:
  1. wikimedia_w2real : `sift100k_u10_q2500_w2real` (`real_s00`) x 6 engines (10 configs)
  2. sift1m_scale     : `sift1m_n600k_u02_q2500_v2` (`abrupt_s07`, `hotspot_s07`) x 6 engines (9 configs each)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

EXT_SRC = Path(os.environ.get("DYNVECAUDIT_ENGINES_DIR", str(Path(__file__).resolve().parents[1] / "engines")))
VENV_PY = EXT_SRC / "venv" / "bin" / "python"
DATA_BASE = Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return 0.0


def wait_quiet_host(max_load: float, min_quiet_checks: int = 2, sleep_s: int = 20) -> None:
    ok_streak = 0
    while ok_streak < min_quiet_checks:
        lv = read_load1m()
        if lv <= max_load:
            ok_streak += 1
            if ok_streak < min_quiet_checks:
                time.sleep(5)
        else:
            ok_streak = 0
            time.sleep(sleep_s)


def check_usearch_available(pydeps: Path) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{pydeps}:{EXT_SRC / 'tools'}"
    r = subprocess.run([str(VENV_PY), "-c", "import usearch; print(usearch.__version__)"], env=env, capture_output=True, text=True)
    return r.returncode == 0


def build_job_matrix(include_usearch: bool = True) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    w2_dir = DATA_BASE / "sift100k_u10_q2500_w2real"
    w2_engines = [
        ("faiss_ivf", "np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}),
        ("faiss_ivf", "np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}),
        ("hnswlib_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}),
        ("hnswlib_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}),
        ("quake", "rt097", {"nlist": 256, "recall_target": 0.97}),
        ("diskann", "L64", {"search_L": 64, "R": 32, "Lbuild": 100}),
        ("diskann", "L96", {"search_L": 96, "R": 32, "Lbuild": 100}),
        ("annoy_rebuild", "sk8000_re4", {"search_k": 8000, "n_trees": 5, "rebuild_every": 4}),
    ]
    if include_usearch:
        w2_engines.extend([
            ("usearch_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}),
            ("usearch_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}),
        ])
    if (w2_dir / "traces").exists():
        for tdir in sorted((w2_dir / "traces").iterdir()):
            if (tdir / "gt.bin").exists():
                for sys_name, ptag, bl in w2_engines:
                    jobs.append({
                        "workload": "wikimedia_w2real",
                        "data_root": w2_dir,
                        "trace": tdir.name,
                        "system": sys_name,
                        "param_tag": ptag,
                        "bl": bl,
                    })

    s1m_dir = DATA_BASE / "sift1m_n600k_u02_q2500_v2"
    s1m_engines = [
        ("faiss_ivf", "np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}),
        ("faiss_ivf", "np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}),
        ("hnswlib_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}),
        ("hnswlib_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}),
        ("quake", "rt097", {"nlist": 256, "recall_target": 0.97}),
        ("diskann", "L64", {"search_L": 64, "R": 32, "Lbuild": 100}),
        ("annoy_rebuild", "sk8000_re4", {"search_k": 8000, "n_trees": 5, "rebuild_every": 4}),
    ]
    if include_usearch:
        s1m_engines.extend([
            ("usearch_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}),
            ("usearch_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}),
        ])
    for tr in ("abrupt_s07", "hotspot_s07"):
        if (s1m_dir / "traces" / tr / "gt.bin").exists():
            for sys_name, ptag, bl in s1m_engines:
                jobs.append({
                    "workload": "sift1m_scale",
                    "data_root": s1m_dir,
                    "trace": tr,
                    "system": sys_name,
                    "param_tag": ptag,
                    "bl": bl,
                })

    return jobs


def summarize_jsonl(path: Path, target_recall: float = 0.965) -> Dict[str, Any]:
    wins, hdr = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "window":
            wins.append(obj)
        elif obj.get("type") == "run":
            hdr = obj
    win_statuses = sorted(set(str(w.get("window_status", "ok")) for w in wins))
    ok = [w for w in wins if w.get("window_status", "ok") == "ok"]
    eval_wins = ok if ok else wins
    q_cpu = sum(float(w.get("query_cpu_ms", 0.0)) for w in eval_wins)
    m_cpu = sum(float(w.get("maint_cpu_ms", 0.0)) for w in eval_wins)
    recs = [float(w["mean_recall"]) for w in eval_wins if w.get("mean_recall") is not None]
    p95s = [float(w["p95_us"]) for w in eval_wins if w.get("p95_us") is not None]
    p99s = [float(w["p99_us"]) for w in eval_wins if w.get("p99_us") is not None]
    attained = sum(1 for r in recs if r >= target_recall)
    peak_rss = hdr.get("peak_rss_mb")
    if peak_rss is None and hdr.get("rss_hwm_kb"):
        peak_rss = round(float(hdr["rss_hwm_kb"]) / 1024.0, 2)
    if peak_rss is None:
        drv_path = path.with_suffix(".driver.jsonl")
        if drv_path.exists():
            for line in drv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                d_obj = json.loads(line)
                if d_obj.get("type") == "run" and d_obj.get("rss_hwm_kb"):
                    peak_rss = round(float(d_obj["rss_hwm_kb"]) / 1024.0, 2)
                    break
    return {
        "n_windows": len(wins),
        "n_ok_windows": len(ok),
        "window_status": ",".join(win_statuses) if win_statuses else "empty",
        "query_cpu_ms": round(q_cpu, 2),
        "maint_cpu_ms": round(m_cpu, 2),
        "total_cpu_ms": round(q_cpu + m_cpu, 2),
        "build_cpu_ms": round(float(hdr.get("build_cpu_ms", 0.0) or 0.0), 2),
        "mean_recall": round(statistics.fmean(recs), 4) if recs else None,
        "min_recall": round(min(recs), 4) if recs else None,
        "attained_0965": f"{attained}/{len(recs)}" if recs else "0/0",
        "attainment_pct": round(100.0 * attained / len(recs), 1) if recs else 0.0,
        "mean_p95_us": round(statistics.fmean(p95s), 1) if p95s else None,
        "mean_p99_us": round(statistics.fmean(p99s), 1) if p99s else None,
        "peak_rss_mb": peak_rss,
        "loadavg_end": hdr.get("loadavg_end"),
        "wall_s": hdr.get("wall_s"),
    }


def _resolve_row_status(s: Dict[str, Any], default_status: str) -> str:
    if s["n_windows"] > 0 and s["n_ok_windows"] == 0 and "unsupported_erase" in s["window_status"]:
        return "unsupported_erase"
    return default_status


def write_summary(summary_rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not summary_rows:
        return
    keys = [
        "idx", "workload", "system", "param_tag", "trace", "n_windows", "n_ok_windows", "window_status",
        "query_cpu_ms", "maint_cpu_ms", "total_cpu_ms", "build_cpu_ms", "mean_recall", "min_recall",
        "attained_0965", "attainment_pct", "mean_p95_us", "mean_p99_us", "peak_rss_mb", "loadavg_end",
        "wall_s", "status", "bl",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="PVLDB Exp 5B Real-Timestamp & 1M 6-Engine Benchmark")
    ap.add_argument("--work-dir", default=os.environ.get("DYNVECAUDIT_ROOT", "."))
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--max-load", type=float, default=2.2)
    ap.add_argument("--timeout-s", type=int, default=2400)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = work_dir / "results" / "exp5b_real_and_1m"
    log_dir = out_dir / "logs"
    pydeps = work_dir / "pydeps"
    wh_script = work_dir / "external_harness" / "window_harness.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    has_usearch = True if args.dry_run else check_usearch_available(pydeps)
    jobs = build_job_matrix(include_usearch=has_usearch)

    if args.dry_run:
        by_wl: Dict[str, int] = {}
        for j in jobs:
            by_wl[j["workload"]] = by_wl.get(j["workload"], 0) + 1
        print(json.dumps({"dry_run": True, "total_jobs": len(jobs), "by_workload": by_wl, "sample": jobs[:3]}, indent=2, default=str))
        return 0

    if not wh_script.exists():
        raise FileNotFoundError(f"Missing overlay harness: {wh_script}")

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

    csv_path = out_dir / "exp5b_summary.csv"
    summary_rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs, 1):
        stem = f"{job['workload']}__{job['system']}_{job['param_tag']}__{job['trace']}"
        out_jsonl = out_dir / f"{stem}.jsonl"
        run_log = log_dir / f"{stem}.log"

        if out_jsonl.exists() and out_jsonl.stat().st_size > 200:
            s = summarize_jsonl(out_jsonl)
            if s["n_windows"] >= 39 and (s["loadavg_end"] is None or float(s["loadavg_end"]) <= args.max_load + 1.0):
                summary_rows.append({"idx": idx, **job, "bl": json.dumps(job["bl"]), **s, "status": _resolve_row_status(s, "skipped_clean")})
                write_summary(summary_rows, csv_path)
                continue

        wait_quiet_host(args.max_load)
        extra_cli: List[str] = []
        if job["system"] == "quake":
            extra_cli = [
                "--nlist", str(job["bl"].get("nlist", 256)),
                "--recall-target", str(job["bl"].get("recall_target", 0.97)),
            ]
        elif job["system"] == "diskann":
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
        try:
            proc = subprocess.run(cmd, cwd=str(EXT_SRC), env=env, capture_output=True, text=True, timeout=args.timeout_s)
        except subprocess.TimeoutExpired as exc:
            wall_s = time.perf_counter() - t0
            run_log.write_text(f"TIMEOUT={args.timeout_s}s\n" + (exc.stdout or "") + "\n--- STDERR ---\n" + (exc.stderr or ""), encoding="utf-8")
            log(f"[{idx}/{len(jobs)}] TIMEOUT {stem} after {wall_s:.1f}s")
            summary_rows.append({"idx": idx, **job, "bl": json.dumps(job["bl"]), "wall_s": round(wall_s, 3), "status": "timeout"})
            write_summary(summary_rows, csv_path)
            continue

        wall_s = time.perf_counter() - t0
        run_log.write_text(f"WALL_S={wall_s:.3f}\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            log(f"[{idx}/{len(jobs)}] FAIL {stem} rc={proc.returncode}: {proc.stderr[-240:]}")
            summary_rows.append({"idx": idx, **job, "bl": json.dumps(job["bl"]), "wall_s": round(wall_s, 3), "status": f"fail_rc_{proc.returncode}"})
            write_summary(summary_rows, csv_path)
            continue

        s = summarize_jsonl(out_jsonl)
        summary_rows.append({
            "idx": idx,
            "workload": job["workload"],
            "system": job["system"],
            "param_tag": job["param_tag"],
            "trace": job["trace"],
            "bl": json.dumps(job["bl"]),
            "wall_s": round(wall_s, 3),
            **s,
            "status": _resolve_row_status(s, "ok"),
        })
        write_summary(summary_rows, csv_path)
        log(f"[{idx}/{len(jobs)}] OK {stem} tot_cpu={s['total_cpu_ms']}ms minR={s['min_recall']} att={s['attained_0965']} rss={s['peak_rss_mb']}MB lv={s['loadavg_end']}")

    log(f"ALL DONE: wrote {len(summary_rows)} rows -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
