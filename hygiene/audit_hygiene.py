#!/usr/bin/env python3
"""DynVecAudit hygiene verification library & CLI.

Implements the four quantitative measurement-hygiene gates for PVLDB EA&B:
  1. check_contamination   : wall/CPU inflation and host-load audit over JSONL runs.
  2. check_clock_leak      : Spearman rank correlation detector catching monotonic
                             step/insert-counter leakage in ML maintenance predictors.
  3. check_cadence_bracket : verifies budgeted selection spend is strictly bracketed by
                             two executed integer cadences (preventing interpolation illusions).
  4. paired_bootstrap_ci   : deterministic paired-bootstrap 95% CI and win counts.

Pure Python stdlib (no required third-party dependencies); passes `--self-test` anywhere.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _average_ranks(values: Sequence[float]) -> List[float]:
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Spearman rank correlation with exact tie-averaging (pure Python)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))


def check_contamination(
    jsonl_path: str | Path,
    max_wall_cpu_ratio: float = 1.18,
    max_load: Optional[float] = None,
    max_probe_slowdown: Optional[float] = 1.25,
) -> Dict[str, Any]:
    """Audit a JSONL run log for wall/CPU inflation, host load spikes, or silent SMT pressure.

    Symmetric hyper-thread contention keeps wall/CPU near 1.00 (Exp 10-B4: <= 1.002 while CPU
    inflates 1.52x-1.78x), so Gate 4 also checks the slowdown of a co-scheduled 17.6 ms GEMV
    probe when the log records it (``probe_slowdown``; clean <= 1.05x, contended >= 1.46x).
    """
    p = Path(jsonl_path)
    if not p.exists():
        return {"passed": False, "path": str(p), "reasons": ["missing_file"]}

    win_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rtype = obj.get("type")
            if rtype == "window":
                win_rows.append(obj)
            elif rtype == "run":
                run_rows.append(obj)
            else:
                meta_rows.append(obj)

    reasons: List[str] = []
    if not win_rows:
        reasons.append("no_window_rows")

    cpu_ms = sum(
        float(r.get("query_cpu_ms", 0.0))
        + float(r.get("update_cpu_ms", 0.0))
        + float(r.get("maint_cpu_ms", 0.0))
        for r in win_rows
    )
    # External harness logs wall via summary/meta or per-window p50/wall fields; qbam_replay logs run row
    wall_s = None
    for row in reversed(run_rows + meta_rows):
        if "wall_s" in row and row["wall_s"] is not None:
            wall_s = float(row["wall_s"])
            break
        if "total_wall_ms" in row and row["total_wall_ms"] is not None:
            wall_s = float(row["total_wall_ms"]) / 1e3
            break

    ratio = None
    if wall_s is not None and cpu_ms > 0.0:
        ratio = wall_s / (cpu_ms / 1e3)
        if ratio > max_wall_cpu_ratio:
            reasons.append(f"wall_cpu_ratio={ratio:.4f}>{max_wall_cpu_ratio}")

    loads: List[float] = []
    for row in win_rows + run_rows + meta_rows:
        for lk in ("load", "load_1m", "loadavg", "load_end", "load_start"):
            if lk in row and row[lk] is not None:
                try:
                    loads.append(float(row[lk]))
                except (TypeError, ValueError):
                    pass
    max_load_seen = max(loads) if loads else None
    if max_load is not None and max_load_seen is not None and max_load_seen > max_load:
        reasons.append(f"max_load={max_load_seen:.2f}>{max_load}")

    probes = [float(r["probe_slowdown"]) for r in win_rows + run_rows + meta_rows if r.get("probe_slowdown") is not None]
    max_probe_seen = max(probes) if probes else None
    if max_probe_slowdown is not None and max_probe_seen is not None and max_probe_seen > max_probe_slowdown:
        reasons.append(f"probe_slowdown={max_probe_seen:.3f}>{max_probe_slowdown}")

    return {
        "passed": len(reasons) == 0,
        "path": str(p),
        "n_windows": len(win_rows),
        "cpu_ms": round(cpu_ms, 3),
        "wall_s": round(wall_s, 3) if wall_s is not None else None,
        "wall_cpu_ratio": round(ratio, 4) if ratio is not None else None,
        "max_load_seen": round(max_load_seen, 3) if max_load_seen is not None else None,
        "max_probe_slowdown": round(max_probe_seen, 3) if max_probe_seen is not None else None,
        "reasons": reasons,
    }


def _to_column_dict(features_df: Any) -> Dict[str, List[float]]:
    if isinstance(features_df, Mapping):
        return {str(k): [float(x) for x in v] for k, v in features_df.items()}
    if hasattr(features_df, "to_dict"):
        raw = features_df.to_dict(orient="list")
        return {str(k): [float(x) for x in v] for k, v in raw.items()}
    rows = list(features_df)
    if not rows:
        return {}
    cols: Dict[str, List[float]] = {str(k): [] for k in rows[0].keys()}
    for r in rows:
        for k in cols:
            cols[k].append(float(r[k]))
    return cols


def check_clock_leak(
    features_df: Any,
    step_col: str,
    threshold: float = 0.70,
    ignore_cols: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Flag features whose |Spearman rho| against monotonic step_col meets threshold."""
    cols = _to_column_dict(features_df)
    if step_col not in cols:
        raise KeyError(f"step_col {step_col!r} not found in features")
    steps = cols[step_col]
    ignored = set(ignore_cols or ()) | {step_col}
    flagged: Dict[str, float] = {}
    all_rhos: Dict[str, float] = {}
    for name, vals in cols.items():
        if name in ignored:
            continue
        rho = spearman_rho(vals, steps)
        all_rhos[name] = round(rho, 4)
        if abs(rho) >= threshold:
            flagged[name] = round(rho, 4)
    return {
        "passed": len(flagged) == 0,
        "step_col": step_col,
        "threshold": threshold,
        "flagged_features": flagged,
        "all_rhos": all_rhos,
    }


def check_cadence_bracket(
    policy_spend_ms: float,
    fixed_spends_dict: Mapping[Any, float],
) -> Dict[str, Any]:
    """Check whether policy_spend_ms is bracketed by two executed fixed cadences."""
    if not fixed_spends_dict:
        return {"bracketed": False, "reason": "empty_fixed_spends"}
    ordered: List[Tuple[Any, float]] = sorted(
        ((k, float(v)) for k, v in fixed_spends_dict.items()),
        key=lambda kv: kv[1],
    )
    lo_entry = None
    hi_entry = None
    for idx in range(len(ordered) - 1):
        k1, s1 = ordered[idx]
        k2, s2 = ordered[idx + 1]
        if s1 - 1e-9 <= policy_spend_ms <= s2 + 1e-9:
            lo_entry = (k1, s1)
            hi_entry = (k2, s2)
            break
    if lo_entry is not None and hi_entry is not None:
        return {
            "bracketed": True,
            "policy_spend_ms": round(float(policy_spend_ms), 3),
            "lower_cadence": lo_entry[0],
            "lower_spend_ms": round(lo_entry[1], 3),
            "upper_cadence": hi_entry[0],
            "upper_spend_ms": round(hi_entry[1], 3),
            "extrapolation_risk": False,
        }
    side = "below_min" if policy_spend_ms < ordered[0][1] else "above_max"
    return {
        "bracketed": False,
        "policy_spend_ms": round(float(policy_spend_ms), 3),
        "min_executed": (ordered[0][0], round(ordered[0][1], 3)),
        "max_executed": (ordered[-1][0], round(ordered[-1][1], 3)),
        "extrapolation_risk": True,
        "reason": side,
    }


def paired_bootstrap_ci(
    diffs: Sequence[float],
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute mean, SD, wins (<0 / >0), and 95% paired bootstrap CI over trace diffs."""
    vals = [float(x) for x in diffs]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": [None, None], "neg_wins": "0/0", "pos_wins": "0/0"}
    mean_val = sum(vals) / n
    sd_val = statistics.stdev(vals) if n > 1 else 0.0
    rng = random.Random(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo_idx = max(0, min(n_boot - 1, int(math.floor(0.025 * (n_boot - 1)))))
    hi_idx = max(0, min(n_boot - 1, int(math.ceil(0.975 * (n_boot - 1)))))
    neg = sum(1 for v in vals if v < 0.0)
    pos = sum(1 for v in vals if v > 0.0)
    return {
        "n": n,
        "mean": round(mean_val, 4),
        "sd": round(sd_val, 4),
        "ci95": [round(boots[lo_idx], 4), round(boots[hi_idx], 4)],
        "neg_wins": f"{neg}/{n}",
        "pos_wins": f"{pos}/{n}",
    }


def run_self_test() -> int:
    import tempfile

    # 1) check_clock_leak
    feats = {
        "j": [1, 2, 3, 4, 5, 6, 7, 8],
        "cum_inserts": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        "q_recent_ratio": [0.4, 0.2, 0.7, 0.3, 0.6, 0.1, 0.5, 0.2],
    }
    leak_res = check_clock_leak(feats, step_col="j", threshold=0.70)
    assert not leak_res["passed"], f"expected clock leak detection: {leak_res}"
    assert "cum_inserts" in leak_res["flagged_features"]
    assert "q_recent_ratio" not in leak_res["flagged_features"]

    clean_res = check_clock_leak(
        {"j": feats["j"], "q_recent_ratio": feats["q_recent_ratio"]},
        step_col="j",
        threshold=0.70,
    )
    assert clean_res["passed"], f"expected clean features to pass: {clean_res}"

    # 2) check_cadence_bracket
    br_ok = check_cadence_bracket(2850.0, {"c5": 2090.0, "c3": 4430.0, "c8": 1100.0})
    assert br_ok["bracketed"] and br_ok["lower_cadence"] == "c5" and br_ok["upper_cadence"] == "c3"
    br_bad = check_cadence_bracket(5500.0, {"c5": 2090.0, "c3": 4430.0})
    assert not br_bad["bracketed"] and br_bad["extrapolation_risk"]

    # 3) check_contamination
    with tempfile.TemporaryDirectory() as td:
        clean_p = Path(td) / "clean.jsonl"
        dirty_p = Path(td) / "dirty.jsonl"
        clean_p.write_text(
            json.dumps({"type": "window", "window": 0, "query_cpu_ms": 950.0, "update_cpu_ms": 50.0, "maint_cpu_ms": 0.0})
            + "\n"
            + json.dumps({"type": "run", "wall_s": 1.05, "load_end": 1.02})
            + "\n",
            encoding="utf-8",
        )
        dirty_p.write_text(
            json.dumps({"type": "window", "window": 0, "query_cpu_ms": 950.0, "update_cpu_ms": 50.0, "maint_cpu_ms": 0.0})
            + "\n"
            + json.dumps({"type": "run", "wall_s": 1.45, "load_end": 6.5})
            + "\n",
            encoding="utf-8",
        )
        smt_p = Path(td) / "silent_smt.jsonl"
        smt_p.write_text(
            json.dumps({"type": "window", "window": 0, "query_cpu_ms": 950.0, "update_cpu_ms": 50.0, "maint_cpu_ms": 0.0})
            + "\n"
            + json.dumps({"type": "run", "wall_s": 1.001, "load_end": 1.0, "probe_slowdown": 1.74})
            + "\n",
            encoding="utf-8",
        )
        c_ok = check_contamination(clean_p, max_wall_cpu_ratio=1.15, max_load=2.0)
        c_bad = check_contamination(dirty_p, max_wall_cpu_ratio=1.15, max_load=2.0)
        c_smt = check_contamination(smt_p, max_wall_cpu_ratio=1.18, max_load=2.0)
        assert c_ok["passed"], f"expected clean run to pass: {c_ok}"
        assert not c_bad["passed"] and len(c_bad["reasons"]) == 2, f"expected 2 flags: {c_bad}"
        assert not c_smt["passed"] and c_smt["reasons"][0].startswith("probe_slowdown"), f"expected probe flag: {c_smt}"

    # 4) paired_bootstrap_ci
    ci = paired_bootstrap_ci([-12.0, -15.0, -10.0, -14.0, -11.0], n_boot=2000, seed=42)
    assert ci["n"] == 5 and ci["neg_wins"] == "5/5" and ci["ci95"][1] < 0.0, ci

    print("audit_hygiene.py --self-test PASS (4/4 gates verified)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DynVecAudit hygiene checker")
    ap.add_argument("--self-test", action="store_true", help="Run built-in unit tests")
    ap.add_argument("--audit-jsonl", type=str, help="Audit a single JSONL replay log")
    ap.add_argument("--max-ratio", type=float, default=1.18)
    ap.add_argument("--max-load", type=float, default=None)
    ap.add_argument("--max-probe", type=float, default=1.25)
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if args.audit_jsonl:
        res = check_contamination(args.audit_jsonl, args.max_ratio, args.max_load, args.max_probe)
        print(json.dumps(res, indent=2))
        return 0 if res["passed"] else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
