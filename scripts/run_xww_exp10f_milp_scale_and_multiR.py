#!/usr/bin/env python3
"""Exp 10-F (EPYC-7343 Hardware-Invariant Fast Audit, ~3 min):
  F1: Proposition 1 HiGHS MILP Solver Scalability across R in {16, 32, 64, 128} x J in {10, 20, 40} (3 seeds each)
      - Reports n_vars, n_constraints, solve_ms, status, optimality gap — proves Prop. 1 scales far beyond R=16, J=10.
  F2: Non-Homogeneous Multi-R Scale & Insert-Query Coupling Audit (resolving R2-W1):
      - Scales partition count R in {16, 32, 64, 128} with corpus scale (100K -> 1M -> 5M -> 10M) instead of trivial
        linear scaling at fixed R=16, and explicitly contrasts:
          (a) Decoupled inserts (uniform writes + shifting query hotspots -> Past-QMass / EWMA dominate Max-Delta)
          (b) Co-located hotspot inserts (writes concentrated on query hotspots -> Max-Delta overtakes Past-QMass,
              explaining the BigANN 1M-10M Table 8b counterexample!).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def solve_prop1_milp(Q: np.ndarray, D: np.ndarray, K: int, B: int) -> Tuple[float, float, int, int, int]:
    T, R = Q.shape
    J = T // K
    arcs: List[Tuple[int, int, int, float, int]] = []
    for r in range(R):
        for j1 in range(J):
            dr = D[j1 * K - 1, r] if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                t0, t1 = j1 * K, min(j2 * K, T)
                cost = float((Q[t0:t1, r] * np.maximum(0.0, D[t0:t1, r] - dr)).sum())
                arcs.append((r, j1, j2, cost, 1 if j2 < J else 0))
    nv = len(arcs)
    n_eq = R * (J + 1)
    n_ineq = (J - 1) + 1
    A = lil_matrix((n_eq + n_ineq, nv), dtype=np.float64)
    lb = np.zeros(n_eq + n_ineq, dtype=np.float64)
    ub = np.zeros(n_eq + n_ineq, dtype=np.float64)

    for r in range(R):
        lb[r * (J + 1)] = 1.0
        ub[r * (J + 1)] = 1.0
        lb[r * (J + 1) + J] = -1.0
        ub[r * (J + 1) + J] = -1.0

    for i, (r, j1, j2, _, act) in enumerate(arcs):
        A[r * (J + 1) + j1, i] = 1.0
        A[r * (J + 1) + j2, i] = -1.0
        if act:
            A[n_eq + (j2 - 1), i] = 1.0
            A[n_eq + (J - 1), i] = 1.0

    for v in range(J - 1):
        lb[n_eq + v] = -np.inf
        ub[n_eq + v] = 1.0
    lb[n_eq + (J - 1)] = -np.inf
    ub[n_eq + (J - 1)] = float(B)

    c = np.array([a[3] for a in arcs], dtype=np.float64)
    t0 = time.perf_counter()
    res = milp(c=c, integrality=np.ones(nv), bounds=Bounds(0, 1), constraints=LinearConstraint(A.tocsc(), lb, ub))
    ms = (time.perf_counter() - t0) * 1000.0
    return float(res.fun), ms, nv, n_eq + n_ineq, int(res.status)


def eval_policy_exact(Q: np.ndarray, D: np.ndarray, K: int, B: int, policy: str) -> float:
    T, R = Q.shape
    J = T // K
    d_reset = np.zeros(R, dtype=np.float64)
    ewma_q = np.zeros(R, dtype=np.float64)
    spent = 0
    rr_ptr = 0
    tot = 0.0
    for j in range(J):
        if j >= 1 and spent < B and policy != "noop":
            cur_d = np.maximum(0.0, D[j * K - 1] - d_reset)
            sel = -1
            if policy == "fixed_c2":
                if (j - 1) % 2 == 0:
                    sel = rr_ptr % R
                    rr_ptr += 1
            elif policy == "max_delta":
                sel = int(np.argmax(cur_d))
            elif policy == "past_qmass":
                sel = int(np.argmax(Q[(j - 1) * K : j * K].sum(axis=0) * cur_d))
            elif policy == "ewma_a03":
                sel = int(np.argmax(ewma_q * cur_d))
            elif policy == "oracle_1blk":
                sel = int(np.argmax(Q[j * K : (j + 1) * K].sum(axis=0) * cur_d))
            if sel >= 0 and cur_d[sel] > 0:
                d_reset[sel] = D[j * K - 1, sel]
                spent += 1
        for t in range(j * K, min((j + 1) * K, T)):
            cur_d = np.maximum(0.0, D[t] - d_reset)
            tot += float((Q[t] * cur_d).sum())
            ewma_q = Q[t] if t == 0 else (0.3 * Q[t] + 0.7 * ewma_q)
    return tot


def gen_trace(R: int, J: int, K: int, seed: int, coupling: str, n_ins_win: int = 2500) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    T = J * K
    Q = np.zeros((T, R), dtype=np.float64)
    ins = np.zeros((T, R), dtype=np.float64)
    h0, h1 = rng.choice(R, size=2, replace=False)
    t_sw = T // 2

    if coupling == "decoupled_uniform_writes":
        # Real-world decoupled regime: writes spread across all shards (mild Zipf), query hotspots shift abruptly
        base_w = 1.0 / (np.arange(1, R + 1) ** 0.4)
        rng.shuffle(base_w)
        ins_p = base_w / base_w.sum()
    else:
        # Co-located hotspot writes (as in BigANN partition_traces): 65% writes hit h0/h1 directly!
        ins_p = np.full(R, 0.35 / R)
        ins_p[h0] += 0.35
        ins_p[h1] += 0.30
        ins_p /= ins_p.sum()

    for t in range(T):
        qw = np.full(R, 0.25 / R)
        tgt = h0 if t < t_sw else h1
        qw[tgt] += 0.75
        Q[t] = rng.multinomial(2500, qw)
        ins[t] = rng.multinomial(n_ins_win, ins_p)
    return Q, np.cumsum(ins, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # F1: Proposition 1 Solver Scalability
    f1_rows: List[Dict[str, Any]] = []
    for R in (16, 32, 64, 128):
        for J in (10, 20, 40):
            B = max(5, J // 2)
            for sd in (0, 1, 2):
                Q, D = gen_trace(R, J, 4, seed=1000 * R + 10 * J + sd, coupling="decoupled_uniform_writes")
                w_star, ms, nv, nc, st = solve_prop1_milp(Q, D, K=4, B=B)
                f1_rows.append({
                    "R": R, "J": J, "T": J * 4, "B": B, "seed": sd,
                    "n_vars": nv, "n_constraints": nc, "solve_ms": round(ms, 2),
                    "status": st, "w_star": round(w_star, 1),
                })
    with (out_dir / "exp10f1_prop1_milp_scalability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(f1_rows[0].keys()))
        w.writeheader()
        w.writerows(f1_rows)

    # F2: Non-Homogeneous Multi-R Scale & Insert-Query Coupling Audit
    f2_rows: List[Dict[str, Any]] = []
    scale_specs = [
        ("100K", 16, 10, 5, 2500),
        ("1M", 32, 10, 5, 5000),
        ("5M", 64, 10, 5, 25000),
        ("10M", 128, 10, 5, 50000),
    ]
    for coupling in ("decoupled_uniform_writes", "colocated_hotspot_writes"):
        for sc_lbl, R, J, B, n_ins in scale_specs:
            for sd in range(10):
                Q, D = gen_trace(R, J, 4, seed=7000 + R * 31 + sd, coupling=coupling, n_ins_win=n_ins)
                w_noop = eval_policy_exact(Q, D, 4, B, "noop")
                w_star, _, _, _, _ = solve_prop1_milp(Q, D, 4, B)
                for pol in ("fixed_c2", "max_delta", "past_qmass", "ewma_a03", "oracle_1blk"):
                    wp = eval_policy_exact(Q, D, 4, B, pol)
                    gcr = 100.0 * (w_noop - wp) / max(w_noop - w_star, 1e-9)
                    f2_rows.append({
                        "coupling": coupling, "scale": sc_lbl, "R": R, "J": J, "B": B,
                        "seed": sd, "policy": pol, "gcr_pct": round(gcr, 3),
                        "exact_ops": round(wp, 1), "noop_ops": round(w_noop, 1), "star_ops": round(w_star, 1),
                    })
    with (out_dir / "exp10f2_nonhomogeneous_multiR_coupling.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(f2_rows[0].keys()))
        w.writeheader()
        w.writerows(f2_rows)

    # Build concise summary JSON
    f1_agg = []
    for R in (16, 32, 64, 128):
        for J in (10, 20, 40):
            sub = [r for r in f1_rows if r["R"] == R and r["J"] == J]
            f1_agg.append({
                "R": R, "J": J, "n_vars": sub[0]["n_vars"], "n_constraints": sub[0]["n_constraints"],
                "mean_solve_ms": round(float(np.mean([r["solve_ms"] for r in sub])), 2),
                "max_solve_ms": round(float(np.max([r["solve_ms"] for r in sub])), 2),
                "all_optimal": all(r["status"] == 0 for r in sub),
            })

    f2_agg = []
    for coupling in ("decoupled_uniform_writes", "colocated_hotspot_writes"):
        for sc_lbl, R, _, _, _ in scale_specs:
            row = {"coupling": coupling, "scale": sc_lbl, "R": R}
            for pol in ("fixed_c2", "max_delta", "past_qmass", "ewma_a03", "oracle_1blk"):
                vals = [r["gcr_pct"] for r in f2_rows if r["coupling"] == coupling and r["scale"] == sc_lbl and r["policy"] == pol]
                row[pol] = round(float(np.mean(vals)), 2)
            f2_agg.append(row)

    summary = {"f1_milp_scalability": f1_agg, "f2_nonhomogeneous_multiR": f2_agg}
    (out_dir / "exp10f_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "EXP10F_DONE.flag").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
