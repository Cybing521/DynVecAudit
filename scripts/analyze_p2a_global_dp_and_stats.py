#!/usr/bin/env python3
"""Paper 2A Analytical Engine:
1. Hardware-invariant Exact Scan Work (`exact_scan_ops = sum_w sum_r q[w,r] * delta_n[w,r]`)
   & Spearman rank correlation vs measured CPU time.
2. Exact Multi-Block Global Optimal Schedule via MILP (`scipy.optimize.milp` / HiGHS)
   compared against 1-Block Myopic Clairvoyant Oracle & realized policies (Competitive Ratio).
3. Paired 10,000-resample Bootstrap 95% CI, Wilcoxon signed-rank test, and Cliff's delta.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import spearmanr, wilcoxon

K_BLOCK = 4


def parse_jsonl_full(path: Path) -> dict:
    blocks, windows = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rt = obj.get("record_type") or obj.get("type")
            if rt == "block":
                blocks.append(obj)
            elif rt == "window":
                windows.append(obj)
    q_mat = np.array([w["shard_q_accesses"] for w in windows], dtype=np.float64)
    d_mat = np.array([w["shard_delta_n"] for w in windows], dtype=np.float64)
    scan_ops = float((q_mat * d_mat).sum())
    merge_vecs = sum(float(b.get("delta_chosen", 0.0)) for b in blocks if str(b.get("action", "noop")) != "noop")
    # Also accumulate gc_vectors if recorded
    gc_vecs = sum(float(w.get("gc_vectors", 0.0)) for w in windows)
    q_cpu = sum(float(w.get("query_cpu_ms", 0.0)) for w in windows)
    u_cpu = sum(float(w.get("update_cpu_ms", 0.0)) for w in windows)
    m_cpu = sum(float(b.get("maintenance_cpu_ms", 0.0)) for b in blocks)
    recalls = [float(w["recall_at_10"]) for w in windows if "recall_at_10" in w]
    n_actions = sum(1 for b in blocks if str(b.get("action", "noop")) != "noop")
    return {
        "q_mat": q_mat,
        "d_mat": d_mat,
        "exact_scan_ops": scan_ops,
        "merge_vectors": merge_vecs,
        "gc_vectors": gc_vecs,
        "query_cpu_ms": q_cpu,
        "update_cpu_ms": u_cpu,
        "maint_cpu_ms": m_cpu,
        "total_cpu_ms": q_cpu + u_cpu + m_cpu,
        "mean_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "min_recall": float(np.min(recalls)) if recalls else float("nan"),
        "n_actions": n_actions,
    }


def solve_global_milp(q_mat: np.ndarray, d_mat: np.ndarray, budget: int, K: int = K_BLOCK) -> tuple[float, list[tuple[int, int]]]:
    """Exact segment-flow MILP minimizing sum_{t,r} Q[t,r] * delta_{schedule}[t,r]
    subject to at most 1 shard cleared at each block j in 1..J-1 and total actions <= budget.
    """
    T, R = q_mat.shape
    J = T // K
    # Build arc list (r, j1, j2) with 0 <= j1 < j2 <= J
    arcs: list[tuple[int, int, int]] = []
    costs: list[float] = []
    arc_idx: dict[tuple[int, int, int], int] = {}

    for r in range(R):
        d_r = d_mat[:, r]
        q_r = q_mat[:, r]
        for j1 in range(J):
            base_d = d_r[K * j1 - 1] if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                t0, t1 = K * j1, min(K * j2, T)
                seg_d = np.maximum(0.0, d_r[t0:t1] - base_d)
                w_cost = float((q_r[t0:t1] * seg_d).sum())
                idx = len(arcs)
                arcs.append((r, j1, j2))
                costs.append(w_cost)
                arc_idx[(r, j1, j2)] = idx

    n_vars = len(arcs)
    rows_sp, cols_sp, vals_sp = [], [], []
    lb_list, ub_list = [], []
    con_row = 0

    # 1. Flow conservation per shard r and node v in 0..J
    for r in range(R):
        # Node 0: outflow == 1
        for j2 in range(1, J + 1):
            rows_sp.append(con_row); cols_sp.append(arc_idx[(r, 0, j2)]); vals_sp.append(1.0)
        lb_list.append(1.0); ub_list.append(1.0); con_row += 1

        # Intermediate nodes v = 1..J-1: inflow - outflow == 0
        for v in range(1, J):
            for j1 in range(v):
                rows_sp.append(con_row); cols_sp.append(arc_idx[(r, j1, v)]); vals_sp.append(1.0)
            for j2 in range(v + 1, J + 1):
                rows_sp.append(con_row); cols_sp.append(arc_idx[(r, v, j2)]); vals_sp.append(-1.0)
            lb_list.append(0.0); ub_list.append(0.0); con_row += 1

    # 2. Slot constraint: at each block j in 1..J-1, at most 1 shard enters block j (gets cleared)
    for v in range(1, J):
        for r in range(R):
            for j1 in range(v):
                rows_sp.append(con_row); cols_sp.append(arc_idx[(r, j1, v)]); vals_sp.append(1.0)
        lb_list.append(-np.inf); ub_list.append(1.0); con_row += 1

    # 3. Total action budget constraint <= budget
    for r in range(R):
        for v in range(1, J):
            for j1 in range(v):
                rows_sp.append(con_row); cols_sp.append(arc_idx[(r, j1, v)]); vals_sp.append(1.0)
    lb_list.append(-np.inf); ub_list.append(float(budget)); con_row += 1

    A = sparse.csc_matrix((vals_sp, (rows_sp, cols_sp)), shape=(con_row, n_vars))
    res = milp(
        c=np.array(costs, dtype=np.float64),
        integrality=np.ones(n_vars, dtype=np.int32),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(A, lb_list, ub_list),
    )
    if not res.success:
        raise RuntimeError(f"MILP failed (budget={budget}): {res.message}")

    sched = []
    for idx, val in enumerate(res.x):
        if val > 0.5:
            r, j1, _ = arcs[idx]
            if j1 > 0:
                sched.append((j1, r))
    sched.sort()
    return float(res.fun), sched


def simulate_myopic_oracle(q_mat: np.ndarray, d_mat: np.ndarray, budget: int, K: int = K_BLOCK) -> float:
    """1-block clairvoyant greedy oracle picking argmax_r q_block[j, r] * current_delta[r] up to budget."""
    T, R = q_mat.shape
    J = T // K
    last_clear = np.zeros(R, dtype=np.int64)
    total_ops = 0.0
    used = 0
    for j in range(J):
        t0, t1 = K * j, min(K * (j + 1), T)
        if j >= 1 and used < budget:
            cur_d = np.array(
                [max(0.0, d_mat[t0 - 1, r] - (d_mat[last_clear[r] - 1, r] if last_clear[r] > 0 else 0.0)) for r in range(R)]
            )
            q_blk = q_mat[t0:t1].mean(axis=0)
            best_r = int(np.argmax(q_blk * cur_d))
            if (q_blk[best_r] * cur_d[best_r]) > 0:
                last_clear[best_r] = t0
                used += 1
        for r in range(R):
            base_d = d_mat[last_clear[r] - 1, r] if last_clear[r] > 0 else 0.0
            seg_d = np.maximum(0.0, d_mat[t0:t1, r] - base_d)
            total_ops += float((q_mat[t0:t1, r] * seg_d).sum())
    return total_ops


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / max(x.size * y.size, 1))


def paired_bootstrap_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_directory(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Group all *.jsonl by trace and arm
    jsonls = sorted(p for p in results_dir.glob("*__*.jsonl") if not p.name.endswith("__cf.jsonl"))
    by_trace: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in jsonls:
        stem = p.stem
        if "__" not in stem:
            continue
        tr, arm = stem.split("__", 1)
        try:
            by_trace[tr][arm] = parse_jsonl_full(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    dp_rows = []
    all_scan_ops, all_q_cpu, all_t_cpu = [], [], []
    within_rhos = []

    for tr, arms_map in sorted(by_trace.items()):
        if "noop" not in arms_map:
            continue
        noop_res = arms_map["noop"]
        q_mat, d_mat = noop_res["q_mat"], noop_res["d_mat"]
        noop_ops = noop_res["exact_scan_ops"]

        # Cache global MILP & myopic oracle for observed budgets (+ canonical 3, 5, 9)
        budgets = sorted({v["n_actions"] for v in arms_map.values() if v["n_actions"] > 0} | {3, 5, 9})
        oracle_cache = {}
        for b in budgets:
            g_ops, _ = solve_global_milp(q_mat, d_mat, b)
            m_ops = simulate_myopic_oracle(q_mat, d_mat, b)
            oracle_cache[b] = (g_ops, m_ops)

        tr_ops, tr_qcpu, tr_tcpu = [], [], []
        for arm, v in sorted(arms_map.items()):
            tr_ops.append(v["exact_scan_ops"])
            tr_qcpu.append(v["query_cpu_ms"])
            tr_tcpu.append(v["total_cpu_ms"])
            all_scan_ops.append(v["exact_scan_ops"])
            all_q_cpu.append(v["query_cpu_ms"])
            all_t_cpu.append(v["total_cpu_ms"])

            b = v["n_actions"]
            if b in oracle_cache:
                g_ops, m_ops = oracle_cache[b]
            else:
                g_ops, m_ops = noop_ops, noop_ops

            denom_g = noop_ops - g_ops
            cap_g = (noop_ops - v["exact_scan_ops"]) / denom_g if denom_g > 1e-6 else 1.0
            comp_ratio = v["exact_scan_ops"] / max(g_ops, 1e-6)
            myopic_gap = (m_ops - g_ops) / max(noop_ops, 1e-6)
            dp_rows.append({
                "trace": tr,
                "arm": arm,
                "n_actions": b,
                "exact_scan_ops": f"{v['exact_scan_ops']:.1f}",
                "noop_scan_ops": f"{noop_ops:.1f}",
                "global_opt_ops": f"{g_ops:.1f}",
                "myopic_opt_ops": f"{m_ops:.1f}",
                "global_capture_ratio": f"{cap_g:.4f}",
                "competitive_ratio": f"{comp_ratio:.4f}",
                "myopic_vs_global_gap_pct": f"{100.0 * myopic_gap:.3f}",
                "query_cpu_ms": f"{v['query_cpu_ms']:.2f}",
                "maint_cpu_ms": f"{v['maint_cpu_ms']:.2f}",
                "total_cpu_ms": f"{v['total_cpu_ms']:.2f}",
                "mean_recall": f"{v['mean_recall']:.5f}",
            })
        if len(tr_ops) >= 4:
            r_q, _ = spearmanr(tr_ops, tr_qcpu)
            r_t, _ = spearmanr(tr_ops, tr_tcpu)
            within_rhos.append((float(r_q), float(r_t)))

    dp_csv = out_dir / "exact_ops_and_dp.csv"
    with open(dp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dp_rows[0].keys()))
        w.writeheader()
        w.writerows(dp_rows)

    # Pairwise non-parametric tests across traces
    comparisons = [
        ("past_qmass_sp1", "noop"),
        ("past_qmass_refresh", "noop"),
        ("past_qmass_refresh", "past_qmass_sp1"),
        ("past_qmass_sp1", "fixed_c2_merge"),
        ("past_qmass_sp2", "fixed_c2_merge"),
        ("past_qmass_b4", "fixed_c3_merge"),
        ("past_qmass_b4", "fixed_c2_merge"),
        ("past_qmass_k5", "noop"),
        ("past_qmass_k2", "noop"),
        ("past_qmass_k5", "fixed_c2_merge"),
        ("forecast_ewma_a03", "noop"),
        ("forecast_gbrt_causal", "forecast_ewma_a03"),
        ("forecast_oracle", "forecast_ewma_a03"),
        ("forecast_gbrt_causal_k5", "past_qmass_k5"),
        ("forecast_gbrt_leaky_k5", "forecast_gbrt_causal_k5"),
        ("forecast_oracle_clairvoyant_k5", "past_qmass_k5"),
        ("fixed_c2_refresh", "fixed_c2_merge"),
    ]
    stat_rows = []
    for arm_a, arm_b in comparisons:
        pairs_cpu, pairs_ops = [], []
        cpu_a_all, cpu_b_all, ops_a_all, ops_b_all = [], [], [], []
        for tr, m in sorted(by_trace.items()):
            if arm_a in m and arm_b in m:
                ca, cb = m[arm_a]["total_cpu_ms"], m[arm_b]["total_cpu_ms"]
                oa, ob = m[arm_a]["exact_scan_ops"], m[arm_b]["exact_scan_ops"]
                pairs_cpu.append(cb - ca)  # positive => arm_a saves CPU vs arm_b
                pairs_ops.append(ob - oa)  # positive => arm_a saves ops vs arm_b
                cpu_a_all.append(ca); cpu_b_all.append(cb)
                ops_a_all.append(oa); ops_b_all.append(ob)
        if len(pairs_cpu) < 2:
            continue
        dc = np.array(pairs_cpu, dtype=np.float64)
        do = np.array(pairs_ops, dtype=np.float64)
        mc, lwc, hic = paired_bootstrap_ci(dc)
        mo, lwo, hio = paired_bootstrap_ci(do)
        try:
            wc_p = float(wilcoxon(dc).pvalue) if np.any(dc != 0) else 1.0
        except Exception:
            wc_p = float("nan")
        try:
            wo_p = float(wilcoxon(do).pvalue) if np.any(do != 0) else 1.0
        except Exception:
            wo_p = float("nan")
        stat_rows.append({
            "arm_a": arm_a,
            "arm_b": arm_b,
            "n_traces": len(dc),
            "wins_cpu_a_lt_b": int(np.sum(dc > 0)),
            "cpu_saved_mean_ms": f"{mc:.1f}",
            "cpu_saved_ci95_lo": f"{lwc:.1f}",
            "cpu_saved_ci95_hi": f"{hic:.1f}",
            "cpu_saved_pct": f"{100.0 * mc / np.mean(cpu_b_all):.2f}",
            "wilcoxon_p_cpu": f"{wc_p:.4g}",
            "cliffs_delta_cpu": f"{cliffs_delta(np.array(cpu_b_all), np.array(cpu_a_all)):.3f}",
            "ops_saved_pct": f"{100.0 * mo / np.mean(ops_b_all):.2f}",
            "ops_saved_ci95_lo_pct": f"{100.0 * lwo / np.mean(ops_b_all):.2f}",
            "ops_saved_ci95_hi_pct": f"{100.0 * hio / np.mean(ops_b_all):.2f}",
            "wilcoxon_p_ops": f"{wo_p:.4g}",
        })

    stat_csv = out_dir / "pairwise_stats.csv"
    if stat_rows:
        with open(stat_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            w.writeheader()
            w.writerows(stat_rows)

    glob_rq, _ = spearmanr(all_scan_ops, all_q_cpu)
    glob_rt, _ = spearmanr(all_scan_ops, all_t_cpu)
    print(f"=== ANALYZED {results_dir} ({len(by_trace)} traces, {len(dp_rows)} runs) ===")
    print(f"Exact Scan Ops vs Query CPU: pooled Spearman rho={glob_rq:.4f}, within-trace mean rho={np.mean([x[0] for x in within_rhos]):.4f}")
    print(f"Exact Scan Ops vs Total CPU: pooled Spearman rho={glob_rt:.4f}, within-trace mean rho={np.mean([x[1] for x in within_rhos]):.4f}")
    for row in stat_rows:
        print(
            f"  {row['arm_a']:20s} vs {row['arm_b']:16s} | n={row['n_traces']} wins={row['wins_cpu_a_lt_b']} "
            f"| CPU saved={row['cpu_saved_pct']}% [{row['cpu_saved_ci95_lo']},{row['cpu_saved_ci95_hi']}]ms p={row['wilcoxon_p_cpu']} delta={row['cliffs_delta_cpu']} "
            f"| Ops saved={row['ops_saved_pct']}% p={row['wilcoxon_p_ops']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    odir = Path(args.out_dir) if args.out_dir else rdir
    analyze_directory(rdir, odir)


if __name__ == "__main__":
    main()
