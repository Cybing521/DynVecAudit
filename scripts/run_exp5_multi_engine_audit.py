#!/usr/bin/env python3
"""Exp 5A: 6-Engine Open-Source Lifecycle & Hygiene Benchmark for PVLDB EA&B (DynVecAudit).

Benchmarks 6 independent open-source vector engines under one leakage-free, single-core
load-gated harness on an isolated Ryzen 9 9950X host:
  1. faiss_ivf      (FAISS inverted lists: append / periodic retrain)
  2. hnswlib_mut    (hnswlib single mutable HNSW with markDelete / replace_deleted)
  3. diskann        (FreshDiskANN C++ dynamic memory index)
  4. quake          (Quake adaptive multi-level partitioning)
  5. annoy_rebuild  (Annoy static forest + scheduled rebuild cadence)
  6. usearch_mut    (USearch single mutable HNSW, f32 l2sq)

Workloads covered:
  - SIFT-128 canonical drifts (abrupt, gradual, hotspot, periodic, uniform; seeds s05, s06)
  - SIFT-128 CRUD 30% & CRUD 50% (abrupt, hotspot; s05, s07)
  - BigANN deletion runbooks (clustered, sliding; s05)
  - Real-timestamp Wikimedia 210d SIFT-128 stream (win1, win2, win3 if present)

Every run is gated on 1-minute host load <= max_load, pinned to Core 8 with nice -n 19,
and audited immediately via `audit_hygiene.check_contamination()`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import audit_hygiene as ah  # noqa: E402

EXT_SRC = Path("/workspace/qbam/external")
VENV_PY = EXT_SRC / "venv" / "bin" / "python"
DATA_BASE = Path("/workspace/qbam/data")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return 0.0


def wait_quiet_host(max_load: float, min_quiet_checks: int = 2, sleep_s: int = 20) -> None:
    """Wait until host loadavg <= max_load for `min_quiet_checks` consecutive checks."""
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


def prepare_overlay_harness(work_dir: Path) -> Path:
    """Create non-invasive harness overlay in work_dir with usearch_mut plugin."""
    hdir = work_dir / "external_harness"
    bdir = hdir / "backends"
    bdir.mkdir(parents=True, exist_ok=True)
    wh_dst = hdir / "window_harness.py"
    if not wh_dst.exists() and (EXT_SRC / "harness" / "window_harness.py").exists():
        shutil.copy2(EXT_SRC / "harness" / "window_harness.py", wh_dst)
    if wh_dst.exists():
        txt = wh_dst.read_text(encoding="utf-8")
        if "peak_rss_mb" not in txt:
            if "import resource" not in txt:
                txt = txt.replace("import json\n", "import json\nimport resource\n", 1)
            target_line = '    header["wall_s"] = round(time.time() - t_start, 3)'
            if target_line in txt:
                txt = txt.replace(
                    target_line,
                    target_line + '\n    header["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)',
                    1,
                )
            wh_dst.write_text(txt, encoding="utf-8")
    for bp in (EXT_SRC / "harness" / "backends").glob("*.py"):
        dst = bdir / bp.name
        if not dst.exists():
            shutil.copy2(bp, dst)
    shutil.copy2(THIS_DIR / "backends" / "usearch_mut.py", bdir / "usearch_mut.py")
    return wh_dst


def check_usearch_available(pydeps: Path) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{pydeps}:{EXT_SRC / 'tools'}"
    r = subprocess.run([str(VENV_PY), "-c", "import usearch; print(usearch.__version__)"], env=env, capture_output=True, text=True)
    if r.returncode == 0:
        log(f"usearch available: {r.stdout.strip()}")
        return True
    log("usearch not yet installed; attempting isolated pip install into pydeps...")
    pydeps.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PY), "-m", "pip", "install", "--quiet",
        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
        "--target", str(pydeps), "--no-deps", "usearch", "numkong", "simsimd",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    r2 = subprocess.run([str(VENV_PY), "-c", "import usearch; print(usearch.__version__)"], env=env, capture_output=True, text=True)
    ok = r2.returncode == 0
    log(f"usearch install check={ok}: {r2.stdout.strip() or r2.stderr[-200:]}")
    return ok


def build_job_matrix(include_usearch: bool) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    # 1. SIFT-128 Drifting Inserts @ matched recall tiers
    sift_dir = DATA_BASE / "sift100k_u10_q2500_v2"
    sift_traces = [
        f"{d}_{s}"
        for s in ("s05", "s06")
        for d in ("abrupt", "gradual", "hotspot", "periodic", "uniform")
        if (sift_dir / "traces" / f"{d}_{s}" / "gt.bin").exists()
    ]
    insert_engines = [
        ("faiss_ivf", "np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}),
        ("faiss_ivf", "np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}),
        ("hnswlib_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}),
        ("hnswlib_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}),
        ("quake", "rt097", {"nlist": 256, "recall_target": 0.97}),
        ("diskann", "L64", {"search_L": 64, "R": 32, "Lbuild": 100}),
        ("diskann", "L96", {"search_L": 96, "R": 32, "Lbuild": 100}),
        ("annoy_rebuild", "sk8000_re4", {"search_k": 8000, "n_trees": 5, "rebuild_every": 4}),
        ("annoy_rebuild", "sk8000_re1", {"search_k": 8000, "n_trees": 5, "rebuild_every": 1}),
    ]
    if include_usearch:
        insert_engines.extend([
            ("usearch_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}),
            ("usearch_mut", "ef128", {"ef": 128, "M": 16, "ef_construction": 200}),
        ])
    for tr in sift_traces:
        for sys_name, ptag, bl in insert_engines:
            jobs.append({
                "workload": "sift128_drift",
                "data_root": sift_dir,
                "trace": tr,
                "system": sys_name,
                "param_tag": ptag,
                "bl": bl,
            })

    # 2. CRUD 30% & 50% + BigANN Runbooks (engines supporting updates/deletes)
    churn_suites = [
        ("crud30v2", DATA_BASE / "sift100k_u10_q2500_crud30v2", ["abrupt_s05", "hotspot_s05", "abrupt_s07", "hotspot_s07"]),
        ("crud50v2", DATA_BASE / "sift100k_u10_q2500_crud50v2", ["abrupt_s05", "hotspot_s05", "abrupt_s07", "hotspot_s07"]),
        ("runbook", DATA_BASE / "sift100k_u10_q2500_runbook", ["clustered_s05", "sliding_s05"]),
    ]
    churn_engines = [
        ("faiss_ivf", "np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}),
        ("faiss_ivf", "np16_ret10", {"nlist": 256, "nprobe": 16, "retrain_every": 10}),
        ("hnswlib_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}),
        ("annoy_rebuild", "sk8000_re4", {"search_k": 8000, "n_trees": 5, "rebuild_every": 4}),
    ]
    if include_usearch:
        churn_engines.append(("usearch_mut", "ef64", {"ef": 64, "M": 16, "ef_construction": 200}))

    for wname, ddir, tr_list in churn_suites:
        if not ddir.exists():
            continue
        for tr in tr_list:
            if not (ddir / "traces" / tr / "gt.bin").exists():
                continue
            for sys_name, ptag, bl in churn_engines:
                jobs.append({
                    "workload": wname,
                    "data_root": ddir,
                    "trace": tr,
                    "system": sys_name,
                    "param_tag": ptag,
                    "bl": bl,
                })

    # 3. Wikimedia real-timestamp SIFT-128 stream (if present)
    for wtag in ("wikimedia_w1", "wikimedia_w2"):
        for cand in (DATA_BASE / wtag, DATA_BASE / f"sift100k_{wtag}"):
            if (cand / "traces").exists():
                for tdir in sorted((cand / "traces").iterdir()):
                    if (tdir / "gt.bin").exists():
                        for sys_name, ptag, bl in insert_engines:
                            jobs.append({
                                "workload": wtag,
                                "data_root": cand,
                                "trace": tdir.name,
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


def main() -> int:
    ap = argparse.ArgumentParser(description="PVLDB Exp 5A 6-Engine Benchmark")
    ap.add_argument("--work-dir", default="/workspace/pvldb")
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--max-load", type=float, default=2.2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resummarize-only", action="store_true", help="Re-parse existing JSONLs and rewrite CSV without running jobs")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = work_dir / "results" / "exp5_multi_engine"
    log_dir = out_dir / "logs"
    pydeps = work_dir / "pydeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    wh_script = prepare_overlay_harness(work_dir)
    has_usearch = True if (args.dry_run or args.resummarize_only) else check_usearch_available(pydeps)
    jobs = build_job_matrix(include_usearch=has_usearch)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "total_jobs": len(jobs), "sample": jobs[:3]}, indent=2, default=str))
        return 0

    if args.resummarize_only:
        summary_rows: List[Dict[str, Any]] = []
        for idx, job in enumerate(jobs, 1):
            stem = f"{job['workload']}__{job['system']}_{job['param_tag']}__{job['trace']}"
            out_jsonl = out_dir / f"{stem}.jsonl"
            if out_jsonl.exists() and out_jsonl.stat().st_size > 200:
                s = summarize_jsonl(out_jsonl)
                row_status = _resolve_row_status(s, "ok")
                summary_rows.append({"idx": idx, **job, "bl": json.dumps(job["bl"]), **s, "status": row_status})
        if summary_rows:
            csv_path = out_dir / "exp5_multi_engine_summary.csv"
            keys = ["idx", "workload", "system", "param_tag", "trace", "n_windows", "n_ok_windows", "window_status",
                    "query_cpu_ms", "maint_cpu_ms", "total_cpu_ms", "build_cpu_ms", "mean_recall", "min_recall",
                    "attained_0965", "attainment_pct", "mean_p95_us", "mean_p99_us", "peak_rss_mb", "loadavg_end",
                    "wall_s", "status", "bl"]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(summary_rows)
            log(f"RESUMMARIZE DONE: wrote {len(summary_rows)} rows -> {csv_path}")
        return 0

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

    summary_rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs, 1):
        stem = f"{job['workload']}__{job['system']}_{job['param_tag']}__{job['trace']}"
        out_jsonl = out_dir / f"{stem}.jsonl"
        run_log = log_dir / f"{stem}.log"

        if out_jsonl.exists() and out_jsonl.stat().st_size > 200:
            s = summarize_jsonl(out_jsonl)
            if s["n_windows"] >= 39 and (s["loadavg_end"] is None or float(s["loadavg_end"]) <= args.max_load + 1.0):
                summary_rows.append({"idx": idx, **job, "bl": json.dumps(job["bl"]), **s, "status": _resolve_row_status(s, "skipped_clean")})
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
        proc = subprocess.run(cmd, cwd=str(EXT_SRC), env=env, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
        run_log.write_text(f"WALL_S={wall_s:.3f}\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            log(f"[{idx}/{len(jobs)}] FAIL {stem} rc={proc.returncode}: {proc.stderr[-240:]}")
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
        log(f"[{idx}/{len(jobs)}] OK {stem} tot_cpu={s['total_cpu_ms']}ms minR={s['min_recall']} att={s['attained_0965']} lv={s['loadavg_end']}")

    if summary_rows:
        csv_path = out_dir / "exp5_multi_engine_summary.csv"
        keys = ["idx", "workload", "system", "param_tag", "trace", "n_windows", "n_ok_windows", "window_status",
                "query_cpu_ms", "maint_cpu_ms", "total_cpu_ms", "build_cpu_ms", "mean_recall", "min_recall",
                "attained_0965", "attainment_pct", "mean_p95_us", "mean_p99_us", "peak_rss_mb", "loadavg_end",
                "wall_s", "status", "bl"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        log(f"ALL DONE: wrote {len(summary_rows)} rows -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
