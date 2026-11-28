#!/usr/bin/env python3
"""R3-Exp 1: Exact orthogonal 2x2 decomposition (Where, When, Foresight) & calibrated-cost (lambda_0) GCR.

Evaluates the exact 2,400 traces of Exp 11-A across:
  - rr_greedy      : round-robin shard selection, greedy spend (baseline)
  - rr_uniform     : round-robin shard selection, uniform pacing cap floor(B*j/(J-1))
  - fixed_c2       : reference executed cadence (c=2, round-robin)
  - past_qmass     : greedy (none), uniform, spacing
  - ewma_a03       : greedy (none), uniform
  - max_delta      : greedy (none), uniform
  - oracle_1blk    : greedy (none), uniform

Reports both lambda = 0 (pure scan ops) and lambda_0 = 3,700 scan-ops/vector
(exact rebuild cost calibrated on Ryzen Core-8, solved against its own exact lambda_0-aware Prop. 1 MILP).
Outputs:
  results/exp11/r3_exp1_orthogonal_decomp.csv
  results/exp11/r3_exp1_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

import run_exp11a_pacing_generality as e11a
import run_xww_exp10f_milp_scale_and_multiR as f2

K = e11a.K
LAMBDA_0 = 3700.0  # calibrated scan-ops per merged/rebuilt vector (Ryzen Core-8)


def solve_prop1_milp_with_cost(
    Q: np.ndarray, D: np.ndarray, K_blk: int, B: int, lam: float, base_n: float | np.ndarray = 0.0
) -> float:
    """Exact segment-flow MILP (Prop. 1) with per-compaction rebuild penalty lam * (base_n[r] + D[j*K-1, r])."""
    if lam == 0.0:
        ws, _, _, _, _ = f2.solve_prop1_milp(Q, D, K_blk, B)
        return float(ws)

    T, R = Q.shape
    J = T // K_blk
    arcs: List[Tuple[int, int, int, float, int]] = []
    for r in range(R):
        for j1 in range(J):
            d_ref = float(D[j1 * K_blk - 1, r]) if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                w = 0.0
                for t in range(j1 * K_blk, min(j2 * K_blk, T)):
                    w += float(Q[t, r]) * max(0.0, float(D[t, r]) - d_ref)
                is_comp = 1 if j2 < J else 0
                if is_comp:
                    # Compaction at boundary j2 rebuilds live shard of size base_n[r] + D[j2*K-1, r]
                    bn_r = float(base_n[r]) if np.ndim(base_n) > 0 else float(base_n)
                    w += lam * (bn_r + float(D[j2 * K_blk - 1, r]))
                arcs.append((r, j1, j2, w, is_comp))

    n_vars = len(arcs)
    c = np.array([a[3] for a in arcs], dtype=np.float64)

    rows_idx: List[int] = []
    cols_idx: List[int] = []
    vals: List[float] = []
    lbs: List[float] = []
    ubs: List[float] = []

    row = 0
    for r in range(R):
        for v in range(J + 1):
            for idx_a, (ar, j1, j2, _, _) in enumerate(arcs):
                if ar != r:
                    continue
                if j1 == v:
                    rows_idx.append(row)
                    cols_idx.append(idx_a)
                    vals.append(1.0)
                if j2 == v:
                    rows_idx.append(row)
                    cols_idx.append(idx_a)
                    vals.append(-1.0)
            rhs = 1.0 if v == 0 else (-1.0 if v == J else 0.0)
            lbs.append(rhs)
            ubs.append(rhs)
            row += 1

    for v in range(1, J):
        for idx_a, (_, _, j2, _, is_comp) in enumerate(arcs):
            if is_comp and j2 == v:
                rows_idx.append(row)
                cols_idx.append(idx_a)
                vals.append(1.0)
        lbs.append(0.0)
        ubs.append(1.0)
        row += 1

    for idx_a, (_, _, _, _, is_comp) in enumerate(arcs):
        if is_comp:
            rows_idx.append(row)
            cols_idx.append(idx_a)
            vals.append(1.0)
    lbs.append(0.0)
    ubs.append(float(B))
    row += 1

    A = csc_matrix((vals, (rows_idx, cols_idx)), shape=(row, n_vars))
    res = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(A, lbs, ubs),
    )
    if not res.success:
        raise RuntimeError(f"MILP failed (lam={lam}): {res.message}")
    return float(res.fun)


def eval_policy_cost(
    Q: np.ndarray, D: np.ndarray, B: int, policy: str, mode: str, lam: float = 0.0, base_n: float | np.ndarray = 0.0
) -> Tuple[float, float]:
    """Return (scan_ops, total_ops_with_lam) for exact trace replay."""
    T, R = Q.shape
    J = T // K
    gap = math.ceil((J - 1) / max(B, 1))
    d_reset = np.zeros(R, dtype=np.float64)
    ewma_q = np.zeros(R, dtype=np.float64)
    spent, last_j = 0, -10 ** 9
    rr_ptr = 0
    scan_tot = 0.0
    reb_tot = 0.0

    for j in range(J):
        allow = j >= 1 and spent < B and policy != "noop"
        if mode == "uniform":
            allow = allow and spent < math.floor(B * j / (J - 1))
        elif mode == "spacing":
            allow = allow and (j - last_j) >= gap

        if policy == "fixed_c2":
            allow = j >= 1 and spent < B and ((j - 1) % 2 == 0)

        if allow:
            cur_d = np.maximum(0.0, D[j * K - 1] - d_reset)
            sel = -1
            if policy in ("rr", "fixed_c2"):
                # pick next shard in round-robin order with cur_d > 0
                for off in range(R):
                    cand = (rr_ptr + off) % R
                    if cur_d[cand] > 0:
                        sel = cand
                        rr_ptr = (cand + 1) % R
                        break
            elif policy == "max_delta":
                sel = int(np.argmax(cur_d))
            elif policy == "past_qmass":
                sel = int(np.argmax(Q[(j - 1) * K : j * K].sum(axis=0) * cur_d))
            elif policy == "ewma_a03":
                sel = int(np.argmax(ewma_q * cur_d))
            elif policy == "oracle_1blk":
                sel = int(np.argmax(Q[j * K : (j + 1) * K].sum(axis=0) * cur_d))
            else:
                raise ValueError(policy)

            if sel >= 0 and cur_d[sel] > 0:
                d_reset[sel] = D[j * K - 1, sel]
                bn_s = float(base_n[sel]) if np.ndim(base_n) > 0 else float(base_n)
                reb_tot += lam * (bn_s + float(D[j * K - 1, sel]))
                spent += 1
                last_j = j

        for t in range(j * K, min((j + 1) * K, T)):
            scan_tot += float((Q[t] * np.maximum(0.0, D[t] - d_reset)).sum())
            ewma_q = Q[t] if t == 0 else (0.3 * Q[t] + 0.7 * ewma_q)

    return scan_tot, scan_tot + reb_tot


def run_cell(args: Tuple[str, str, int, int, int, int]) -> List[Dict[str, object]]:
    gen, coupling, R, J, B, seed = args
    Q, D = e11a.gen_trace(gen, coupling, R, J, seed=zlib.crc32(f"{gen}|{coupling}|{R}|{J}|{B}|{seed}".encode()))
    base_n = 100_000.0 / float(R)
    w0 = float((Q * D).sum())

    ws_0 = solve_prop1_milp_with_cost(Q, D, K, B, lam=0.0, base_n=base_n)
    ws_lam = solve_prop1_milp_with_cost(Q, D, K, B, lam=LAMBDA_0, base_n=base_n)
    denom_0 = max(w0 - ws_0, 1e-9)
    denom_lam = max(w0 - ws_lam, 1e-9)

    arms = [
        ("rr", "none"),
        ("rr", "uniform"),
        ("fixed_c2", "none"),
        ("max_delta", "none"),
        ("max_delta", "uniform"),
        ("past_qmass", "none"),
        ("past_qmass", "uniform"),
        ("past_qmass", "spacing"),
        ("ewma_a03", "none"),
        ("ewma_a03", "uniform"),
        ("oracle_1blk", "none"),
        ("oracle_1blk", "uniform"),
    ]
    rows: List[Dict[str, object]] = []
    for pol, mode in arms:
        s0, tot_lam = eval_policy_cost(Q, D, B, pol, mode, lam=LAMBDA_0, base_n=base_n)
        rows.append({
            "gen": gen,
            "coupling": coupling,
            "R": R,
            "J": J,
            "B": B,
            "seed": seed,
            "policy": pol,
            "mode": mode,
            "gcr_lam0_0_pct": 100.0 * (w0 - s0) / denom_0,
            "gcr_lam0_3700_pct": 100.0 * (w0 - tot_lam) / denom_lam,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/exp11")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gens = ("stationary", "abrupt1", "abrupt2", "abrupt3", "gradual", "periodic")
    jb = ((10, 3), (10, 5), (10, 7), (20, 5), (20, 10))
    cells = [(g, c, R, J, B, s) for g in gens for c in ("decoupled", "colocated") for R in (16, 64) for (J, B) in jb for s in range(args.seeds)]

    with Pool(args.procs) as pool:
        rows = [r for rs in pool.imap_unordered(run_cell, cells, chunksize=8) for r in rs]

    df = pd.DataFrame(rows)
    df.to_csv(out / "r3_exp1_orthogonal_decomp.csv", index=False)

    summary: Dict[str, object] = {"n_traces": len(cells), "lambda_0": LAMBDA_0}
    for tag, col in (("lam0", "gcr_lam0_0_pct"), ("lam3700", "gcr_lam0_3700_pct")):
        piv = df.pivot_table(index=["gen", "coupling", "R", "J", "B", "seed"], columns=["policy", "mode"], values=col)
        rr_g = piv[("rr", "none")]
        rr_u = piv[("rr", "uniform")]
        pq_g = piv[("past_qmass", "none")]
        pq_u = piv[("past_qmass", "uniform")]
        or_g = piv[("oracle_1blk", "none")]
        or_u = piv[("oracle_1blk", "uniform")]
        c2 = piv[("fixed_c2", "none")]

        d_where = pq_g - rr_g
        d_when = pq_u - pq_g
        d_fore = or_u - pq_u
        # Reverse order: When on RR first, then Where on uniform
        d_when_first = rr_u - rr_g
        d_where_after = pq_u - rr_u

        total_check = rr_g + d_where + d_when + d_fore - or_u
        assert float(total_check.abs().max()) < 1e-6, f"Identity failed: {total_check.abs().max()}"

        summary[tag] = {
            "rr_greedy": round(float(rr_g.mean()), 2),
            "rr_uniform": round(float(rr_u.mean()), 2),
            "fixed_c2": round(float(c2.mean()), 2),
            "pq_greedy": round(float(pq_g.mean()), 2),
            "pq_uniform": round(float(pq_u.mean()), 2),
            "oracle_greedy": round(float(or_g.mean()), 2),
            "oracle_uniform": round(float(or_u.mean()), 2),
            "where_gain": round(float(d_where.mean()), 2),
            "when_gain": round(float(d_when.mean()), 2),
            "foresight_gain": round(float(d_fore.mean()), 2),
            "when_first_on_rr": round(float(d_when_first.mean()), 2),
            "where_after_pacing": round(float(d_where_after.mean()), 2),
            "identity_sum": round(float((rr_g + d_where + d_when + d_fore).mean()), 2),
            "when_win_pct": round(float(100.0 * (d_when > 0).mean()), 1),
            "when_loss_gt1pp_pct": round(float(100.0 * (d_when < -1.0).mean()), 1),
            "pq_uniform_beats_oracle_greedy_pct": round(float(100.0 * (pq_u > or_g).mean()), 1),
        }

    (out / "r3_exp1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
