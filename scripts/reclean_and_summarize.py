#!/usr/bin/env python3
"""Post-Completion Hygiene Cleaner & Statistical Re-Aggregator for DynVecAudit / Paper 2A.

Scans benchmark results tree (excluding stage1_hygiene where contention is intentional),
audits:
  1. Missing WALL_MS= in .log (--require-wall)
  2. WALL_MS / total_cpu_ms > max_ratio
  3. Bad pin_cpu in --exclude-pins (e.g. colliding cores like 22 or 27 on EPYC-7343)
  4. Trace-level w0 invariant-window canary (w0_query_cpu_ms / min_w0 > max_w0_ratio)

Optionally re-executes ONLY contaminated `.config.json` files cleanly on isolated CPU core(s)
when host load permits, writes fresh `clean_summary.csv` per directory, and emits `hygiene_audit_report.json`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return 0.0


def _split_trace_arm(stem: str) -> Tuple[str, str]:
    if "__" in stem:
        tr, arm = stem.split("__", 1)
        return tr, arm
    return stem, "unknown"


def audit_one_run(
    jsonl_path: Path,
    max_ratio: float,
    require_wall: bool = True,
    exclude_pins: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    log_path = jsonl_path.with_suffix(".log")
    cfg_path = jsonl_path.with_suffix(".config.json")
    wins, blks = [], []
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            rt = obj.get("record_type") or obj.get("type")
            if rt == "window":
                wins.append(obj)
            elif rt == "block":
                blks.append(obj)
    except Exception as exc:
        tr, arm = _split_trace_arm(jsonl_path.stem)
        return {
            "jsonl": str(jsonl_path),
            "cfg": str(cfg_path),
            "has_cfg": cfg_path.exists(),
            "parent_dir": str(jsonl_path.parent),
            "trace": tr,
            "arm": arm,
            "pin_cpu": None,
            "w0_query_cpu_ms": 0.0,
            "passed": False,
            "reasons": [f"corrupt_json:{exc}"],
            "reason": f"corrupt_json:{exc}",
        }

    tr, arm = _split_trace_arm(jsonl_path.stem)
    if not wins:
        return {
            "jsonl": str(jsonl_path),
            "cfg": str(cfg_path),
            "has_cfg": cfg_path.exists(),
            "parent_dir": str(jsonl_path.parent),
            "trace": tr,
            "arm": arm,
            "pin_cpu": None,
            "w0_query_cpu_ms": 0.0,
            "passed": False,
            "reasons": ["no_windows"],
            "reason": "no_windows",
        }

    pin_cpu = None
    if cfg_path.exists():
        try:
            pin_cpu = json.loads(cfg_path.read_text(encoding="utf-8")).get("pin_cpu")
        except Exception:
            pin_cpu = None

    q_cpu = sum(float(w.get("query_cpu_ms", 0.0)) for w in wins)
    u_cpu = sum(float(w.get("update_cpu_ms", 0.0)) for w in wins)
    m_cpu = sum(float(b.get("maintenance_cpu_ms", 0.0)) for b in blks)
    cpu_ms = q_cpu + u_cpu + m_cpu
    w0_q_cpu = float(wins[0].get("query_cpu_ms", 0.0))
    n_actions = sum(1 for b in blks if str(b.get("action", "noop")) != "noop")

    recalls = [float(w["recall_at_10"]) for w in wins if w.get("recall_at_10") is not None]
    mean_rec = sum(recalls) / len(recalls) if recalls else None
    min_rec = min(recalls) if recalls else None

    exact_scan_ops = None
    if all("shard_q_accesses" in w and "shard_delta_n" in w for w in wins):
        ops_sum = 0.0
        for w in wins:
            qa = w["shard_q_accesses"]
            dn = w["shard_delta_n"]
            ops_sum += sum(float(a) * float(b) for a, b in zip(qa, dn))
        exact_scan_ops = ops_sum

    wall_ms = None
    if log_path.exists():
        first = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        if first and first[0].startswith("WALL_MS="):
            try:
                wall_ms = float(first[0].split("=", 1)[1])
            except ValueError:
                pass
    ratio = (wall_ms / cpu_ms) if (wall_ms is not None and cpu_ms > 0) else None

    reasons: List[str] = []
    if require_wall and wall_ms is None:
        reasons.append("missing_wall_ms")
    if ratio is not None and ratio > max_ratio:
        reasons.append(f"wall_cpu_ratio={ratio:.4f}>{max_ratio}")
    if exclude_pins and pin_cpu in exclude_pins:
        reasons.append(f"bad_pin_cpu={pin_cpu}")

    passed = len(reasons) == 0
    return {
        "jsonl": str(jsonl_path),
        "cfg": str(cfg_path),
        "has_cfg": cfg_path.exists(),
        "parent_dir": str(jsonl_path.parent),
        "trace": tr,
        "arm": arm,
        "pin_cpu": pin_cpu,
        "n_windows": len(wins),
        "n_actions": n_actions,
        "exact_scan_ops": round(exact_scan_ops, 1) if exact_scan_ops is not None else "",
        "query_cpu_ms": round(q_cpu, 2),
        "update_cpu_ms": round(u_cpu, 2),
        "maint_cpu_ms": round(m_cpu, 2),
        "total_cpu_ms": round(cpu_ms, 2),
        "mean_recall": round(mean_rec, 5) if mean_rec is not None else "",
        "min_recall": round(min_rec, 5) if min_rec is not None else "",
        "w0_query_cpu_ms": round(w0_q_cpu, 2),
        "w0_ratio": None,
        "wall_ms": round(wall_ms, 2) if wall_ms is not None else None,
        "wall_cpu_ratio": round(ratio, 4) if ratio is not None else None,
        "passed": passed,
        "reasons": reasons,
        "reason": ";".join(reasons),
    }


def apply_w0_canary(records: List[Dict[str, Any]], max_w0_ratio: float) -> None:
    by_trace: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("w0_query_cpu_ms", 0.0) > 0:
            by_trace[(r["parent_dir"], r["trace"])].append(r)

    for grp in by_trace.values():
        min_w0 = min(r["w0_query_cpu_ms"] for r in grp)
        if min_w0 <= 0:
            continue
        for r in grp:
            w0_ratio = r["w0_query_cpu_ms"] / min_w0
            r["w0_ratio"] = round(w0_ratio, 4)
            if w0_ratio > max_w0_ratio:
                msg = f"w0_ratio={w0_ratio:.3f}>{max_w0_ratio}"
                if msg not in r["reasons"]:
                    r["reasons"].append(msg)
                r["passed"] = False
                r["reason"] = ";".join(r["reasons"])


def scan_and_audit(
    all_jsonls: List[Path],
    max_ratio: float,
    max_w0_ratio: float,
    require_wall: bool,
    exclude_pins: Set[int],
) -> List[Dict[str, Any]]:
    records = [audit_one_run(jp, max_ratio, require_wall=require_wall, exclude_pins=exclude_pins) for jp in all_jsonls]
    apply_w0_canary(records, max_w0_ratio=max_w0_ratio)
    return records


def write_clean_summaries(records: List[Dict[str, Any]]) -> int:
    by_dir: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if "__" in Path(r["jsonl"]).stem and "n_windows" in r:
            by_dir[r["parent_dir"]].append(r)

    fieldnames = [
        "trace", "arm", "pin_cpu", "n_windows", "n_actions", "exact_scan_ops",
        "query_cpu_ms", "update_cpu_ms", "maint_cpu_ms", "total_cpu_ms",
        "mean_recall", "min_recall", "w0_query_cpu_ms", "w0_ratio",
        "wall_ms", "wall_cpu_ratio", "hygiene_passed",
    ]
    written = 0
    for dpath, grp in sorted(by_dir.items()):
        out_csv = Path(dpath) / "clean_summary.csv"
        grp_sorted = sorted(grp, key=lambda x: (x["trace"], x["arm"]))
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in grp_sorted:
                w.writerow({
                    "trace": r["trace"],
                    "arm": r["arm"],
                    "pin_cpu": r.get("pin_cpu") if r.get("pin_cpu") is not None else "",
                    "n_windows": r.get("n_windows", 0),
                    "n_actions": r.get("n_actions", 0),
                    "exact_scan_ops": r.get("exact_scan_ops", ""),
                    "query_cpu_ms": r.get("query_cpu_ms", ""),
                    "update_cpu_ms": r.get("update_cpu_ms", ""),
                    "maint_cpu_ms": r.get("maint_cpu_ms", ""),
                    "total_cpu_ms": r.get("total_cpu_ms", ""),
                    "mean_recall": r.get("mean_recall", ""),
                    "min_recall": r.get("min_recall", ""),
                    "w0_query_cpu_ms": r.get("w0_query_cpu_ms", ""),
                    "w0_ratio": r.get("w0_ratio") if r.get("w0_ratio") is not None else "",
                    "wall_ms": r.get("wall_ms") if r.get("wall_ms") is not None else "",
                    "wall_cpu_ratio": r.get("wall_cpu_ratio") if r.get("wall_cpu_ratio") is not None else "",
                    "hygiene_passed": bool(r.get("passed", False)),
                })
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit & re-clean contaminated qbam_replay JSONLs")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--cores", default="8", help="Comma-separated isolated cores for clean re-runs")
    ap.add_argument("--max-ratio", type=float, default=1.18)
    ap.add_argument("--max-w0-ratio", type=float, default=1.18)
    ap.add_argument("--max-load", type=float, default=12.0)
    ap.add_argument("--require-wall", action="store_true", default=True)
    ap.add_argument("--no-require-wall", dest="require_wall", action="store_false")
    ap.add_argument("--exclude-pins", default="", help="Comma-separated contaminated CPU IDs to force re-run")
    ap.add_argument("--include-subdirs", default="", help="Optional comma-separated relative subdir prefixes to filter")
    ap.add_argument("--rerun-dirty", action="store_true")
    args = ap.parse_args()

    root = Path(args.results_root)
    binary = Path(args.binary)
    cores = [int(x) for x in args.cores.split(",") if x.strip()]
    exclude_pins = {int(x) for x in args.exclude_pins.split(",") if x.strip()}
    inc_subdirs = [x.strip().strip("/") for x in args.include_subdirs.split(",") if x.strip()]

    def _keep(jp: Path) -> bool:
        if any(bad in jp.parts for bad in ("stage1_hygiene", "smoke", "track_cf")):
            return False
        if "reclean" in jp.name or jp.name.endswith(".tmp"):
            return False
        if inc_subdirs:
            rel = str(jp.relative_to(root))
            if not any(rel == s or rel.startswith(s + "/") for s in inc_subdirs):
                return False
        return True

    all_jsonls = sorted(p for p in root.rglob("*.jsonl") if _keep(p))
    initial_records = scan_and_audit(
        all_jsonls,
        max_ratio=args.max_ratio,
        max_w0_ratio=args.max_w0_ratio,
        require_wall=args.require_wall,
        exclude_pins=exclude_pins,
    )
    initial_dirty = [r for r in initial_records if not r["passed"]]
    clean_count = len(initial_records) - len(initial_dirty)
    print(f"[Audit] Scanned {len(all_jsonls)} JSONLs: clean={clean_count}, dirty={len(initial_dirty)}", flush=True)

    if args.rerun_dirty and initial_dirty:
        valid_dirty = [d for d in initial_dirty if d["has_cfg"]]

        def _rerun(idx_item: Tuple[int, Dict[str, Any]]) -> str:
            idx, item = idx_item
            pin = cores[idx % len(cores)]
            while read_load1m() > args.max_load:
                time.sleep(15)
            jp = Path(item["jsonl"])
            cp = Path(item["cfg"])
            lp = jp.with_suffix(".log")
            cfg = json.loads(cp.read_text(encoding="utf-8"))
            cfg["pin_cpu"] = pin
            cp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            tmp = jp.with_suffix(".jsonl.reclean.tmp")
            if tmp.exists():
                tmp.unlink()
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            t0 = time.perf_counter()
            proc = subprocess.run(
                ["taskset", "-c", str(pin), "nice", "-n", "19", str(binary), "--config", str(cp), "--out", str(tmp)],
                env=env, capture_output=True, text=True,
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if proc.returncode == 0 and tmp.exists():
                lp.write_text(f"WALL_MS={wall_ms:.2f}\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
                tmp.replace(jp)
            return str(jp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cores)) as ex:
            for done_i, _ in enumerate(ex.map(_rerun, enumerate(valid_dirty)), 1):
                if done_i % 25 == 0 or done_i == len(valid_dirty):
                    print(f"[Reclean] Progress {done_i}/{len(valid_dirty)}", flush=True)

    final_records = scan_and_audit(
        all_jsonls,
        max_ratio=args.max_ratio,
        max_w0_ratio=args.max_w0_ratio,
        require_wall=args.require_wall,
        exclude_pins=exclude_pins,
    )
    still_dirty = [r for r in final_records if not r["passed"]]
    reran_fixed = max(0, len(initial_dirty) - len(still_dirty))
    n_csv = write_clean_summaries(final_records)
    print(
        f"[Reclean] Final audit: clean={len(final_records)-len(still_dirty)}/{len(final_records)}, "
        f"fixed={reran_fixed}, remaining_dirty={len(still_dirty)}, clean_summary_csvs={n_csv}",
        flush=True,
    )

    by_dir_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "clean": 0, "dirty": 0})
    for r in final_records:
        rel_d = str(Path(r["parent_dir"]).relative_to(root))
        by_dir_counts[rel_d]["total"] += 1
        if r["passed"]:
            by_dir_counts[rel_d]["clean"] += 1
        else:
            by_dir_counts[rel_d]["dirty"] += 1

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results_root": str(root),
        "max_ratio": args.max_ratio,
        "max_w0_ratio": args.max_w0_ratio,
        "require_wall": args.require_wall,
        "exclude_pins": sorted(exclude_pins),
        "include_subdirs": inc_subdirs,
        "total_scanned": len(all_jsonls),
        "initial_clean": clean_count,
        "initial_dirty_count": len(initial_dirty),
        "reran_fixed_count": reran_fixed,
        "remaining_dirty_count": len(still_dirty),
        "clean_summary_csvs_written": n_csv,
        "by_dir": dict(sorted(by_dir_counts.items())),
        "remaining_dirty": [
            {k: d[k] for k in ("jsonl", "pin_cpu", "w0_ratio", "wall_cpu_ratio", "reason")}
            for d in still_dirty[:50]
        ],
    }
    rep_name = "hygiene_audit_report.json" if not inc_subdirs else f"hygiene_audit_report_{'_'.join(s.replace('/', '_') for s in inc_subdirs)}.json"
    out_rep = root / rep_name
    out_rep.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[Audit] Wrote {out_rep}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
