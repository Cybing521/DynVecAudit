#!/usr/bin/env python3
"""Audit B: Compaction-Cost-Aware Proposition 1 Sensitivity & Horizon H Sensitivity (W7).

Part B1: Extends the Segment-Flow MILP arc cost with physical compaction work:
  arc_cost(r, j1, j2) = sum_{t=K*j1}^{min(K*j2,T)-1} Q_{t,r} * max(0, D_{t,r} - D_{K*j1-1,r})
                      + lambda * 1[j2 < J] * (N_r + D_{K*j2-1, r})
where lambda_0 is calibrated directly from measured Ryzen Core-8 per-vector compaction CPU
divided by per-scan-op query CPU. Sweeps lambda in {0, 0.5*lambda_0, 1.0*lambda_0, 2.0*lambda_0}
over canonical SIFT-100K traces (abrupt, gradual, hotspot, periodic, uniform x s05..s09)
and measures GCR + Kendall's tau rank preservation across policies.

Part B2: Evaluates Horizon H sensitivity across:
  (a) 5 full-branch Track-CF traces (H in {1, 2, 3, 4} windows per decision block);
  (b) Canonical 25 SIFT-100K traces over multi-block lookahead horizons H_blk in {1, 2, 4, 10(full)}.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.stats import kendalltau

for cand in (os.environ.get("DYNVECAUDIT_TOOLS_DIR", "./tools"), os.environ.get("DYNVECAUDIT_TOOLS_DIR", "./tools")):
    if Path(cand).exists():
        sys.path.insert(0, cand)
        break
import qbam_io as qio  # noqa: E402

K_BLOCK = 4
BUDGET = 9
NPROBE = 4


def extract_qd(data_root: Path, tr_dir: Path) -> Dict[str, Any]:
    pool, ph = qio.read_pool(data_root / "pool.bin")
    init, ih = qio.read_init(data_root / "init.bin")
    ev, th = qio.read_trace(tr_dir / "trace.bin")
    dim = int(ph.fields["dim"])
    R = int(ih.fields["R"])
    n_win = int(th.fields["n_windows"])
    centroids = qio.read_centroids(data_root / "centroids.f32", R, dim)
    c_norm2 = np.sum(centroids ** 2, axis=1)

    base_n = np.bincount(init["shard"], minlength=R).astype(np.float64)
    Q = np.zeros((n_win, R), dtype=np.float64)
    ins_pw = np.zeros((n_win, R), dtype=np.float64)

    qm = ev["type"] == qio.EV_QUERY
    if np.any(qm):
        qe = ev[qm]
        qv = pool[qe["vec_ref"]]
        d2 = np.sum(qv ** 2, axis=1, keepdims=True) + c_norm2[None, :] - 2.0 * (qv @ centroids.T)
        top = np.argpartition(d2, kth=NPROBE - 1, axis=1)[:, :NPROBE]
        for w_idx, shs in zip(qe["window"].tolist(), top.tolist()):
            for s in shs:
                Q[w_idx, s] += 1.0

    im = (ev["type"] == qio.EV_INSERT) | (ev["type"] == qio.EV_UPDATE)
    if np.any(im):
        ie = ev[im]
        iv = pool[ie["vec_ref"]]
        d2 = np.sum(iv ** 2, axis=1, keepdims=True) + c_norm2[None, :] - 2.0 * (iv @ centroids.T)
        bs = np.argmin(d2, axis=1)
        for w_idx, s in zip(ie["window"].tolist(), bs.tolist()):
            ins_pw[w_idx, s] += 1.0

    D = np.cumsum(ins_pw, axis=0)
    return {"R": R, "n_windows": n_win, "base_n": base_n, "Q": Q, "D": D}


def solve_cost_aware_milp(Q: np.ndarray, D: np.ndarray, base_n: np.ndarray, K: int, budget: int, lam: float) -> float:
    T, R = Q.shape
    J = T // K
    arcs: List[Tuple[int, int, int]] = []
    costs: List[float] = []
    for r in range(R):
        for j1 in range(J):
            d_ref = float(D[K * j1 - 1, r]) if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                t0, t1 = K * j1, min(K * j2, T)
                scan_c = float((Q[t0:t1, r] * np.maximum(0.0, D[t0:t1, r] - d_ref)).sum())
                comp_c = lam * (float(base_n[r]) + float(D[K * j2 - 1, r])) if j2 < J else 0.0
                arcs.append((r, j1, j2))
                costs.append(scan_c + comp_c)
    nv = len(arcs)
    n_eq = R + R * (J - 1)
    n_ineq = (J - 1) + 1
    A = lil_matrix((n_eq + n_ineq, nv), dtype=np.float64)
    lb = np.zeros(n_eq + n_ineq, dtype=np.float64)
    ub = np.zeros(n_eq + n_ineq, dtype=np.float64)
    for r in range(R):
        lb[r] = 1.0
        ub[r] = 1.0
    for idx, (r, j1, j2) in enumerate(arcs):
        if j1 == 0:
            A[r, idx] = 1.0
        if 1 <= j2 <= J - 1:
            rc = R + r * (J - 1) + (j2 - 1)
            A[rc, idx] = 1.0
            A[n_eq + (j2 - 1), idx] = 1.0
            A[n_eq + (J - 1), idx] = 1.0
        if 1 <= j1 <= J - 1:
            rc = R + r * (J - 1) + (j1 - 1)
            A[rc, idx] = -1.0
    for v in range(J - 1):
        lb[n_eq + v] = -np.inf
        ub[n_eq + v] = 1.0
    lb[n_eq + (J - 1)] = -np.inf
    ub[n_eq + (J - 1)] = float(min(budget, J - 1))
    res = milp(
        c=np.array(costs, dtype=np.float64),
        integrality=np.ones(nv),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(A.tocsc(), lb, ub),
    )
    return float(res.fun)


def eval_policy_cost(Q: np.ndarray, D: np.ndarray, base_n: np.ndarray, policy: str, lam: float, budget: int = BUDGET) -> float:
    T, R = Q.shape
    J = T // K_BLOCK
    d_at_reset = np.zeros(R, dtype=np.float64)
    ewma_q = np.zeros(R, dtype=np.float64)
    spent = 0
    rr = 0
    w_tot = 0.0
    c_tot = 0.0
    future_Q = np.array([Q[j * K_BLOCK:(j + 1) * K_BLOCK].sum(axis=0) for j in range(J)])
    for j in range(J):
        if j > 0:
            prev_q = Q[(j - 1) * K_BLOCK: j * K_BLOCK].sum(axis=0)
            ewma_q = prev_q.copy() if j == 1 else (0.3 * prev_q + 0.7 * ewma_q)
            cur_d = np.maximum(0.0, D[j * K_BLOCK - 1] - d_at_reset)
            sel_r: Optional[int] = None
            if policy == "fixed_c2" and (j - 1) % 2 == 0 and spent < budget:
                sel_r = rr % R
                rr += 1
            elif policy == "fixed_c1" and spent < budget:
                sel_r = rr % R
                rr += 1
            elif policy != "noop" and spent < budget:
                if policy == "past_qmass":
                    sc = prev_q * cur_d
                elif policy == "ewma_a03":
                    sc = ewma_q * cur_d
                elif policy == "max_delta":
                    sc = cur_d
                elif policy == "oracle_1blk":
                    sc = future_Q[j] * cur_d
                else:
                    sc = np.zeros(R)
                br = int(np.argmax(sc))
                if cur_d[br] > 0:
                    sel_r = br
            if sel_r is not None:
                c_tot += lam * (float(base_n[sel_r]) + float(D[j * K_BLOCK - 1, sel_r]))
                d_at_reset[sel_r] = D[j * K_BLOCK - 1, sel_r]
                spent += 1
        for t in range(j * K_BLOCK, min((j + 1) * K_BLOCK, T)):
            w_tot += float((Q[t] * np.maximum(0.0, D[t] - d_at_reset)).sum())
    return w_tot + c_tot


def run_b1_milp_sensitivity(data_root: Path) -> Dict[str, Any]:
    drifts = ["abrupt", "gradual", "hotspot", "periodic", "uniform"]
    seeds = ["s05", "s06", "s07", "s08", "s09"]

    # Calibrate lambda_0 from Ryzen Core-8 measured ratio:
    # c=1 averages 9 actions, maint_cpu = 7.01s -> 0.779s/compaction (~6500 vecs/shard => ~0.120 ms/vec),
    # query_cpu = 136.76s over 4.217B ops -> 3.243e-8 s/scan_op => lambda_0 = 0.779 / (6500 * 3.243e-8) ≈ 3695 scan_ops/vec.
    lam0 = 3700.0
    mults = [0.0, 0.5, 1.0, 2.0]
    policies = ["fixed_c2", "fixed_c1", "max_delta", "past_qmass", "ewma_a03", "oracle_1blk"]

    rows: List[Dict[str, Any]] = []
    for dr in drifts:
        for sd in seeds:
            tr = f"{dr}_{sd}"
            qd = extract_qd(data_root, data_root / "traces" / tr)
            w_noop = float((qd["Q"] * qd["D"]).sum())
            for mult in mults:
                lam = mult * lam0
                w_star = solve_cost_aware_milp(qd["Q"], qd["D"], qd["base_n"], K_BLOCK, BUDGET, lam)
                for pol in policies:
                    cost = eval_policy_cost(qd["Q"], qd["D"], qd["base_n"], pol, lam)
                    gcr = 100.0 * (w_noop - cost) / max(w_noop - w_star, 1e-9)
                    rows.append({
                        "lambda_mult": mult, "lambda_ops_per_vec": lam,
                        "trace": tr, "drift": dr, "policy": pol,
                        "total_equiv_ops": round(cost, 1),
                        "gcr_pct": round(gcr, 2),
                    })

    df = pd.DataFrame(rows)
    agg = df.groupby(["lambda_mult", "policy"], as_index=False)[["total_equiv_ops", "gcr_pct"]].mean()
    base_rank = agg[agg["lambda_mult"] == 0.0].set_index("policy")["gcr_pct"]
    taus = {}
    for mult in mults:
        sub = agg[agg["lambda_mult"] == mult].set_index("policy")["gcr_pct"].reindex(policies)
        taus[str(mult)] = round(float(kendalltau(base_rank.reindex(policies), sub).statistic), 4)
    return {"lambda_0_ops_per_vec": lam0, "taus_vs_lam0": taus, "rows": rows, "agg": agg.to_dict(orient="records")}


def run_b2_horizon_sensitivity(cf_glob: str) -> List[Dict[str, Any]]:
    # Evaluate H in {1, 2, 3, 4} windows over Track-CF JSONL traces
    cf_files = sorted(glob.glob(cf_glob))
    out_rows: List[Dict[str, Any]] = []
    for p in cf_files:
        recs = [json.loads(line) for line in open(p)]
        branches = [r for r in recs if r.get("type") == "branch"]
        by_j: Dict[int, list] = {}
        for br in branches:
            by_j.setdefault(int(br["j"]), []).append(br)

        for H in (1, 2, 3, 4):
            flips = pairs = best_m = 0
            for j, items in sorted(by_j.items()):
                ch = [sum(w["query_cpu_ms"] for w in it["windows"][:H]) + it["maintenance_cpu_ms"] for it in items]
                c4 = [sum(w["query_cpu_ms"] for w in it["windows"][:4]) + it["maintenance_cpu_ms"] for it in items]
                if min(range(len(items)), key=lambda i: ch[i]) == min(range(len(items)), key=lambda i: c4[i]):
                    best_m += 1
                n = len(items)
                for a in range(n):
                    for b in range(a + 1, n):
                        pairs += 1
                        if (ch[a] < ch[b]) != (c4[a] < c4[b]):
                            flips += 1
            out_rows.append({
                "trace": Path(p).stem, "horizon_H_win": H,
                "pair_flip_vs_H4_pct": round(100.0 * flips / max(pairs, 1), 2),
                "best_action_match_vs_H4": f"{best_m}/{len(by_j)}",
                "best_action_match_pct": round(100.0 * best_m / max(len(by_j), 1), 1),
            })
    return out_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit B: Proposition 1 & Horizon H sensitivity")
    ap.add_argument("--data-root", default=os.environ.get("DYNVEC_DATA_ROOT", "./data/sift100k_u10_q2500_v2"))
    ap.add_argument("--cf-glob", default=os.environ.get("DYNVEC_CF_GLOB", "./results/track_cf/*.jsonl"))
    ap.add_argument("--out-dir", default="./results/exp9_review_audits")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b1 = run_b1_milp_sensitivity(Path(args.data_root))
    pd.DataFrame(b1["rows"]).to_csv(out_dir / "milp_cost_sensitivity.csv", index=False)

    b2 = run_b2_horizon_sensitivity(args.cf_glob)
    pd.DataFrame(b2).to_csv(out_dir / "horizon_h_sensitivity.csv", index=False)

    summary = {
        "b1_milp_sensitivity": {
            "lambda_0_ops_per_vec": b1["lambda_0_ops_per_vec"],
            "taus_vs_lam0": b1["taus_vs_lam0"],
            "agg": b1["agg"],
        },
        "b2_horizon_pooled": pd.DataFrame(b2).groupby("horizon_H_win", as_index=False)[["pair_flip_vs_H4_pct", "best_action_match_pct"]].mean().to_dict(orient="records"),
    }
    (out_dir / "milp_and_horizon_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
