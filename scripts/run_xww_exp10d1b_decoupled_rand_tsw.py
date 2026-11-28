#!/usr/bin/env python3
"""Exp 10-D1b (EPYC-7343): Decoupled-Hub Randomized t_sw Trap 2 Split-Leakage Audit.

Decouples query hotspot switching from cumulative shard inserts:
  - Inserts arrive uniformly across all R=16 shards (10,000/win), so D_{t,r} grows smoothly across shards.
  - Queries focus (65%) on hub_A=2 for t < t_sw and abruptly jump to hub_B=11 for t >= t_sw.
Evaluates three split protocols:
  1. shared_fixed: train & test share identical switch window t_sw in {16, 24, 28};
     gbrt_leaky uses {leak_block_j, leak_dcum_total, shard_id} and memorizes t_sw -> high switch-block Top-1 & GCR.
  2. train16_24_test28: train on t_sw in {16, 24}, test on unseen t_sw=28 (W5 protocol).
  3. randomized_tsw_n30: 30 traces with randomized t_sw ~ Uniform{10..30} (leave-one-out),
     plus paired bootstrap 95% CIs showing gbrt_leaky collapses to causal baselines once schedule memorization fails.
"""
from __future__ import annotations

import os

for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_ev] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.ensemble import HistGradientBoostingRegressor

K_BLOCK = 4
BUDGET = 9
T_WIN = 40
R_SHARDS = 16


def solve_milp_exact(Q: np.ndarray, D: np.ndarray, K: int = K_BLOCK, B: int = BUDGET) -> float:
    T, R = Q.shape
    J = T // K
    arcs = []
    for r in range(R):
        for j1 in range(J):
            dr = D[j1 * K - 1, r] if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                w = float(sum(Q[t, r] * max(0.0, D[t, r] - dr) for t in range(j1 * K, min(j2 * K, T))))
                arcs.append((r, j1, j2, w, 1 if j2 < J else 0))
    n = len(arcs)
    c = np.array([a[3] for a in arcs])
    A = lil_matrix((R * (J + 1) + 1, n))
    rhs = []
    rid = 0
    for r in range(R):
        for j in range(J + 1):
            for i, (ar, j1, j2, _, _) in enumerate(arcs):
                if ar == r:
                    if j1 == j:
                        A[rid, i] = 1.0
                    if j2 == j:
                        A[rid, i] = -1.0
            rhs.append(1.0 if j == 0 else (-1.0 if j == J else 0.0))
            rid += 1
    for i, (_, _, _, _, act) in enumerate(arcs):
        if act:
            A[rid, i] = 1.0
    res = milp(c=c, integrality=np.ones(n), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A.tocsc(), np.array(rhs + [0.0]), np.array(rhs + [float(B)])))
    return float(res.fun)


def generate_decoupled_qd(seed: int, t_sw: int, hub_a: int = 2, hub_b: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Q = np.zeros((T_WIN, R_SHARDS), dtype=np.float64)
    D = np.zeros((T_WIN, R_SHARDS), dtype=np.float64)
    cum = np.zeros(R_SHARDS, dtype=np.float64)
    ins_p = np.full(R_SHARDS, 1.0 / R_SHARDS)
    for t in range(T_WIN):
        hot = hub_a if t < t_sw else hub_b
        qp = np.full(R_SHARDS, 0.35 / (R_SHARDS - 1))
        qp[hot] = 0.65
        Q[t] = rng.multinomial(2500, qp).astype(np.float64)
        cum += rng.multinomial(10000, ins_p).astype(np.float64)
        D[t] = cum.copy()
    return Q, D


def build_samples(Q: np.ndarray, D: np.ndarray, trace_id: str, t_sw: int) -> List[Dict[str, Any]]:
    T, R = Q.shape
    J = T // K_BLOCK
    out: List[Dict[str, Any]] = []
    ewma = Q[0].copy()
    for t in range(1, K_BLOCK):
        ewma = 0.3 * Q[t] + 0.7 * ewma
    for j in range(1, J):
        t0, t1 = j * K_BLOCK, min((j + 1) * K_BLOCK, T)
        q_prev = Q[(j - 1) * K_BLOCK : t0].sum(axis=0)
        q_last = Q[t0 - 1]
        d_now = D[t0 - 1]
        d_inc4 = d_now - (D[t0 - 1 - K_BLOCK] if t0 - 1 >= K_BLOCK else 0.0)
        y_next = Q[t0:t1].sum(axis=0)
        for r in range(R):
            out.append({
                "trace_id": trace_id, "t_sw": t_sw, "j": j, "r": r,
                "q_prev": float(q_prev[r]), "q_last": float(q_last[r]),
                "ewma": float(ewma[r]),
                "q_ratio": float(q_last[r] / max(q_prev[r] / K_BLOCK, 1.0)),
                "d_now": float(d_now[r]), "d_inc4": float(d_inc4[r]),
                "q_frac": float(q_prev[r] / max(q_prev.sum(), 1.0)),
                "d_frac": float(d_now[r] / max(d_now.sum(), 1.0)),
                "leak_block_j": float(j),
                "leak_dcum_total": float(d_now.sum()),
                "shard_id": float(r),
                "target": float(y_next[r]),
            })
        for t in range(t0, t1):
            ewma = 0.3 * Q[t] + 0.7 * ewma
    return out


def replay(Q: np.ndarray, D: np.ndarray, pol: str, pmap: Dict[int, np.ndarray] | None = None) -> float:
    T, R = Q.shape
    J = T // K_BLOCK
    d_rst = np.zeros(R)
    spent = 0
    w_tot = 0.0
    ewma = np.zeros(R)
    for t in range(K_BLOCK):
        ewma = Q[t] if t == 0 else 0.3 * Q[t] + 0.7 * ewma
    for j in range(J):
        t0, t1 = j * K_BLOCK, min((j + 1) * K_BLOCK, T)
        if j >= 1 and spent < BUDGET:
            cur_d = np.maximum(0.0, D[t0 - 1] - d_rst)
            sel = -1
            if pol == "fixed_c2":
                if (j - 1) % 2 == 0:
                    sel = int(np.argmax(cur_d))
            elif pol == "past_qmass":
                sel = int(np.argmax(Q[(j - 1) * K_BLOCK : t0].sum(axis=0) * cur_d))
            elif pol == "ewma":
                sel = int(np.argmax(ewma * cur_d))
            elif pol == "oracle":
                sel = int(np.argmax(Q[t0:t1].sum(axis=0) * cur_d))
            elif pol in ("gbrt_leaky", "gbrt_causal") and pmap is not None:
                sel = int(np.argmax(np.maximum(0.0, pmap[j]) * cur_d))
            if sel >= 0 and cur_d[sel] > 0:
                d_rst[sel] = D[t0 - 1, sel]
                spent += 1
        for t in range(t0, t1):
            w_tot += float((Q[t] * np.maximum(0.0, D[t] - d_rst)).sum())
            if j >= 1:
                ewma = 0.3 * Q[t] + 0.7 * ewma
    return w_tot


def bootstrap_diff(a: np.ndarray, b: np.ndarray) -> List[float]:
    rng = np.random.default_rng(7)
    d = a - b
    n = len(d)
    boots = np.array([d[rng.integers(0, n, size=n)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [round(float(d.mean()), 2), round(float(lo), 2), round(float(hi), 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ.get("DYNVECAUDIT_OUT_DIR", "./results/exp10d"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_s = np.random.default_rng(2026)
    rand_specs = [(i, f"rand_{i:02d}", int(rng_s.integers(10, 31)), 2, 11) for i in range(30)]
    rand_both_specs = []
    for i in range(30):
        ha = int(rng_s.integers(0, R_SHARDS))
        hb = int((ha + rng_s.integers(1, R_SHARDS)) % R_SHARDS)
        rand_both_specs.append((400 + i, f"rand_both_{i:02d}", int(rng_s.integers(10, 31)), ha, hb))
    fixed_specs = (
        [(100 + i, f"tsw16_{i}", 16, 2, 11) for i in range(4)]
        + [(200 + i, f"tsw24_{i}", 24, 2, 11) for i in range(4)]
        + [(300 + i, f"tsw28_{i}", 28, 2, 11) for i in range(4)]
    )
    causal_cols = ["q_prev", "q_last", "ewma", "q_ratio", "d_now", "d_inc4", "q_frac", "d_frac"]
    leaky_cols = causal_cols + ["leak_block_j", "leak_dcum_total", "shard_id"]

    def _run(specs: List[Tuple[int, str, int, int, int]], mode: str) -> List[Dict[str, Any]]:
        qds, samps, bounds = {}, {}, {}
        for sd, tid, tsw, ha, hb in specs:
            Q, D = generate_decoupled_qd(5000 + sd, tsw, hub_a=ha, hub_b=hb)
            qds[tid] = (Q, D, tsw)
            samps[tid] = build_samples(Q, D, tid, tsw)
            bounds[tid] = (float((Q * D).sum()), solve_milp_exact(Q, D))
        rows = []
        for _, tid_te, tsw_te, _, _ in specs:
            if mode == "train16_24_test28" and tsw_te != 28:
                continue
            Q, D, _ = qds[tid_te]
            wn, ws = bounds[tid_te]
            for pol in ("fixed_c2", "past_qmass", "ewma", "oracle"):
                ops = replay(Q, D, pol)
                rows.append({"cohort": mode, "trace": tid_te, "tsw": tsw_te, "policy": pol,
                             "exact_scan_ops": ops, "gcr_pct": 100.0 * (wn - ops) / max(wn - ws, 1e-9),
                             "switch_top1": np.nan})
            if mode == "shared_fixed":
                tr_ids = [tid for _, tid, tsw, _, _ in specs if tsw == tsw_te and tid != tid_te]
            elif mode == "train16_24_test28":
                tr_ids = [tid for _, tid, tsw, _, _ in specs if tsw in (16, 24)]
            else:
                tr_ids = [tid for _, tid, _, _, _ in specs if tid != tid_te]
            df_tr = pd.DataFrame([r for tid in tr_ids for r in samps[tid]])
            df_te = pd.DataFrame(samps[tid_te])
            sw_j = tsw_te // K_BLOCK
            for pol, cols in [("gbrt_leaky", leaky_cols), ("gbrt_causal", causal_cols)]:
                mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_leaf_nodes=31, random_state=42)
                mdl.fit(df_tr[cols].to_numpy(), df_tr["target"].to_numpy())
                yhat = np.maximum(0.0, mdl.predict(df_te[cols].to_numpy()))
                pmap = {}
                sw_hit = 0
                for jv in sorted(df_te["j"].unique()):
                    m = (df_te["j"] == jv).to_numpy()
                    pv = yhat[m]
                    pmap[int(jv)] = pv
                    if int(jv) == sw_j:
                        yt = df_te.loc[m, "target"].to_numpy()
                        sw_hit = 1 if int(np.argmax(pv)) == int(np.argmax(yt)) else 0
                ops = replay(Q, D, pol, pmap)
                rows.append({"cohort": mode, "trace": tid_te, "tsw": tsw_te, "policy": pol,
                             "exact_scan_ops": ops, "gcr_pct": 100.0 * (wn - ops) / max(wn - ws, 1e-9),
                             "switch_top1": float(sw_hit)})
        return rows

    df = pd.DataFrame(
        _run(fixed_specs, "shared_fixed")
        + _run(fixed_specs, "train16_24_test28")
        + _run(rand_specs, "randomized_tsw_n30")
        + _run(rand_both_specs, "rand_hub_and_rand_tsw_n30")
    )
    df.to_csv(out_dir / "exp10d1b_decoupled_rand_tsw.csv", index=False)
    r30 = df[df["cohort"] == "randomized_tsw_n30"].pivot(index="trace", columns="policy", values="gcr_pct")
    rb = df[df["cohort"] == "rand_hub_and_rand_tsw_n30"].pivot(index="trace", columns="policy", values="gcr_pct")
    summary = {
        "means": df.groupby(["cohort", "policy"], as_index=False)[["gcr_pct", "switch_top1", "exact_scan_ops"]].mean().round(3).to_dict(orient="records"),
        "rand30_bootstrap_gcr_diff": {
            "leaky_minus_causal": bootstrap_diff(r30["gbrt_leaky"].to_numpy(), r30["gbrt_causal"].to_numpy()),
            "causal_minus_past_qmass": bootstrap_diff(r30["gbrt_causal"].to_numpy(), r30["past_qmass"].to_numpy()),
            "past_qmass_minus_fixed_c2": bootstrap_diff(r30["past_qmass"].to_numpy(), r30["fixed_c2"].to_numpy()),
        },
        "rand_both_bootstrap_gcr_diff": {
            "leaky_minus_causal": bootstrap_diff(rb["gbrt_leaky"].to_numpy(), rb["gbrt_causal"].to_numpy()),
            "causal_minus_past_qmass": bootstrap_diff(rb["gbrt_causal"].to_numpy(), rb["past_qmass"].to_numpy()),
            "past_qmass_minus_fixed_c2": bootstrap_diff(rb["past_qmass"].to_numpy(), rb["fixed_c2"].to_numpy()),
        },
    }
    (out_dir / "exp10d1b_decoupled_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
