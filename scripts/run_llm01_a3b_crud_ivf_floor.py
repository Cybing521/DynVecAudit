#!/usr/bin/env python3
"""Suite A3b: floor-calibrated FAISS-IVF on 100K CRUD-30%/50% (review round 2, N6).

Completes the A3 CRUD comparison, where FAISS-IVF was only run at nprobe=16
(22.5%/38.8% attainment). Runs nprobe in {24, 32, 48} on the same crud30v2 /
crud50v2 traces (abrupt_s05, hotspot_s05) through the unchanged A3 worker,
single-threaded and pinned to an isolated physical core (default 40 on NUMA-1,
HT sibling 104 idle) with memory bound to the same node.
Gate 4: runs with wall/CPU > 1.18 are retried up to 3 times and flagged otherwise.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_llm01_real_engine_traps as rt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="40")
    ap.add_argument("--out-dir", default="results_a3b")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    knobs = [(f"np{p}", {"nlist": 256, "nprobe": p, "retrain_every": 0}) for p in (24, 32, 48)]
    rows = []
    for wl in ("crud30v2", "crud50v2"):
        for tr in ("abrupt_s05", "hotspot_s05"):
            for tag, params in knobs:
                res = None
                for attempt in range(3):
                    res = rt.invoke_worker(
                        cache_dir, wl, rt.DATA_ROOTS_LLM01[wl], tr, "faiss_ivf", tag, params,
                        cores=args.core, threads=1, batch_query=False, max_windows=None,
                        reuse_cache=(attempt == 0), enforce_clean_iso=True,
                    )
                    if float(res["wall_cpu_ratio"]) <= 1.18:
                        break
                    rt.log(f"[A3b retry {attempt + 1}] {wl}/{tr} {tag} w/C={res['wall_cpu_ratio']}")
                    time.sleep(5.0)
                res["gate4_flag"] = int(float(res["wall_cpu_ratio"]) > 1.18)
                rows.append({"suite": "A3b", **res})
                rt.log(f"[A3b] {wl}/{tr} {tag}: total={float(res['total_cpu_ms'])/1000:.1f}s "
                       f"att={res['attainment_0965_pct']} w/C={res['wall_cpu_ratio']}")
                with (out_dir / "a3b_crud_ivf_floor.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    (out_dir / "A3B_DONE.flag").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
