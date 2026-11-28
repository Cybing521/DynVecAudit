#!/usr/bin/env python3
"""Exp 10-A2 (Ryzen-9950X Core-8 Pure-CPU Follow-Up, queued cleanly after EXP10A_DONE.flag):
  1. M3 Q/u Crossover Midpoints (resolving R2-W2):
     - Runs Q/u in {3.75, 5.00, 6.25, 7.50} via exact `_worker_m3` signature (`dataset`, `u_pct`, `q_per_win`,
       `query_repeat`) over available traces (up to 3) x 3 engines (FAISS-IVF np32, hnswlib ef128, USearch ef128).
     - Directly brackets empirical (Q/u)* without unmeasured linear interpolation across [2.5, 10.0].
  2. M1 hnswlib Symmetry Completion (resolving R2-W4):
     - Completes remaining 2 traces (periodic_s05, uniform_s05) + missing policies (fixed_c4, online_milp, oracle_1blk)
       for 16-shard hnswlib so FAISS-IVF and hnswlib are 100% symmetric (n=5 traces, all 9 policies).
"""
from __future__ import annotations

import os

for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_ev] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import run_mimo_supplement_real_engines as m9  # noqa: E402

OUT_DIR = Path(os.environ.get("DYNVECAUDIT_OUT_DIR", "./results/exp10a2_followup"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def discover_common_traces(datasets: List[str], preferred: List[str], max_n: int = 3) -> List[str]:
    sets = []
    for ds in datasets:
        tdir = m9.DATA_BASE / ds / "traces"
        if not tdir.exists():
            continue
        sets.append({p.name for p in tdir.iterdir() if p.is_dir() and (p / "gt.bin").exists()})
    if not sets:
        return ["hotspot_s07"]
    common = set.intersection(*sets)
    picked = [t for t in preferred if t in common]
    for t in sorted(common):
        if len(picked) >= max_n:
            break
        if t not in picked:
            picked.append(t)
    return picked[:max_n] if picked else ["hotspot_s07"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-load", type=float, default=2.5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: M3 Midpoints (Q/u in {3.75, 5.00, 6.25, 7.50})
    m3_cells: List[Tuple[int, int, str, int]] = [
        (2, 2500, "sift100k_u02_q2500_v2", 3),  # Q=7500,  Q/u=3.75
        (2, 5000, "sift100k_u02_q5000_v2", 2),  # Q=10000, Q/u=5.00
        (2, 2500, "sift100k_u02_q2500_v2", 5),  # Q=12500, Q/u=6.25
        (2, 5000, "sift100k_u02_q5000_v2", 3),  # Q=15000, Q/u=7.50
    ]
    traces = discover_common_traces(
        ["sift100k_u02_q2500_v2", "sift100k_u02_q5000_v2"],
        preferred=["hotspot_s07", "abrupt_s05", "gradual_s05", "abrupt_s07", "gradual_s07"],
        max_n=3,
    )
    log(f"=== Exp 10-A2 Part 1: M3 Q/u Midpoints on traces={traces} ===")
    m3_rows: List[Dict[str, Any]] = []
    m3_csv = OUT_DIR / "m3_qu_midpoints.csv"
    m3_engines = [
        ("faiss_ivf", "np32"),
        ("hnswlib", "ef128"),
        ("usearch_f32", "ef128"),
    ]
    for tr in traces:
        for u_pct, q_pw, ds, rep in m3_cells:
            for sys_n, ptag in m3_engines:
                m9.wait_quiet_host(args.max_load)
                try:
                    res = m9._invoke_self_worker(
                        "m3",
                        {
                            "dataset": ds,
                            "trace": tr,
                            "system": sys_n,
                            "param_tag": ptag,
                            "u_pct": u_pct,
                            "q_per_win": q_pw,
                            "query_repeat": rep,
                        },
                        timeout_s=900,
                    )
                    m3_rows.append(res)
                    write_csv(m3_rows, m3_csv)
                    log(
                        f"  [M3-Mid] {tr} Q/u={res['q_to_u_ratio']} {sys_n}: "
                        f"cpu={res['total_cpu_s']}s rec={res['mean_recall']} att={res['att_0965_pct']}% w/C={res['wall_cpu_ratio']}"
                    )
                except Exception as exc:
                    log(f"  [M3-Mid] FAIL {tr} {ds} rep={rep} {sys_n}: {exc}")

    # Part 2: M1 16-Shard hnswlib Symmetry Completion
    log("=== Exp 10-A2 Part 2: M1 16-Shard hnswlib Symmetry Completion ===")
    m1_rows: List[Dict[str, Any]] = []
    m1_csv = OUT_DIR / "m1_hnswlib_completion.csv"
    all_pols = ["noop", "fixed_c2", "fixed_c4", "past_qmass", "ewma_a03", "gbrt_causal", "gbrt_leaky", "online_milp", "oracle_1blk"]
    jobs_m1: List[Tuple[str, str]] = []
    for tr in ("abrupt_s05", "gradual_s05", "hotspot_s05"):
        for pol in ("fixed_c4", "online_milp", "oracle_1blk"):
            jobs_m1.append((tr, pol))
    for tr in ("periodic_s05", "uniform_s05"):
        for pol in all_pols:
            jobs_m1.append((tr, pol))

    pred_cache: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    for tr, pol in jobs_m1:
        drift = tr.rsplit("_", 1)[0]
        preds_map: Dict[str, List[float]] = {}
        if pol in ("gbrt_causal", "gbrt_leaky"):
            if drift not in pred_cache:
                pred_cache[drift] = m9._build_gbrt_predictions(drift, target_seed="s05")
            preds_map = {str(k): v for k, v in pred_cache[drift][pol].items()}
        m9.wait_quiet_host(args.max_load)
        try:
            res = m9._invoke_self_worker(
                "m1",
                {
                    "engine": "sharded_hnswlib",
                    "trace": tr,
                    "policy": pol,
                    "budget": 5,
                    "preds_map": preds_map,
                },
                timeout_s=900,
            )
            m1_rows.append(res)
            write_csv(m1_rows, m1_csv)
            log(
                f"  [M1-HNSW] {tr} {pol}: cpu={res['total_cpu_s']}s "
                f"gcr={res['gcr_ops_pct']}% rec={res['mean_recall']} w/C={res['wall_cpu_ratio']}"
            )
        except Exception as exc:
            log(f"  [M1-HNSW] FAIL {tr} {pol}: {exc}")

    (OUT_DIR / "EXP10A2_DONE.flag").write_text("ok\n", encoding="utf-8")
    log("EXP10A2_ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
