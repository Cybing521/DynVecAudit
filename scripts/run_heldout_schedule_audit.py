#!/usr/bin/env python3
"""Audit A: Shared-Schedule vs. Held-Out Schedule Split-Leakage Verification (Trap 2 / W5).

Runs on llm_01 (Socket-1 isolated core) using exact (Q, D) matrices from abrupt-switch
traces across four distinct switch timings t_sw in {16, 20, 24, 28}:
  - tsw16/sift100k_u10_q2500_v2/traces/abrupt_s05, abrupt_s06 (t_sw=16)
  - sift100k_u10_q2500_v2/traces/abrupt_s05..s09 (t_sw=20)
  - tsw24/sift100k_u10_q2500_v2/traces/abrupt_s05, abrupt_s06 (t_sw=24)
  - tsw28/sift100k_u10_q2500_v2/traces/abrupt_s05, abrupt_s06 (t_sw=28)

Compares two cross-validation split protocols using identical LightGBM hyperparameters:
  1. Shared-Schedule Split (LOSO): Train on sibling seeds sharing identical t_sw,
     evaluate on held-out seed with the same t_sw.
  2. Held-Out Schedule Split (Leave-One-t_sw-Out): Train strictly on traces with
     t_sw' != t_sw, evaluate on the held-out switch time t_sw.

Evaluates GBRT-Leaky (11 causal features + leak_block_j, leak_dcum_shard, leak_dcum_total)
against GBRT-Causal (11 causal features), Past-QMass, EWMA (a=0.3), Fixed c=2, and Oracle-1blk
under exact_scan_ops and Theorem/Proposition 1 MILP GCR (B=9, K=4).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

for cand in ("/root/qbamdeploy/qbam/external/tools", "/workspace/qbam/external/tools"):
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


def solve_milp_exact(Q: np.ndarray, D: np.ndarray, K: int, budget: int) -> float:
    T, R = Q.shape
    J = T // K
    arcs: List[Tuple[int, int, int]] = []
    costs: List[float] = []
    for r in range(R):
        for j1 in range(J):
            d_ref = float(D[K * j1 - 1, r]) if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                t0, t1 = K * j1, min(K * j2, T)
                cost = float((Q[t0:t1, r] * np.maximum(0.0, D[t0:t1, r] - d_ref)).sum())
                arcs.append((r, j1, j2))
                costs.append(cost)
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


def build_samples(trace_spec: Tuple[int, str, Path]) -> List[Dict[str, Any]]:
    tsw, tid, tr_dir = trace_spec
    data_root = tr_dir.parents[1]
    s = extract_qd(data_root, tr_dir)
    Q, D, base_n, R, n_win = s["Q"], s["D"], s["base_n"], s["R"], s["n_windows"]
    J = n_win // K_BLOCK
    samples: List[Dict[str, Any]] = []
    for j in range(1, J):
        h0 = j * K_BLOCK
        past = Q[max(0, h0 - 8): h0]
        L = past.shape[0]
        x = np.arange(L, dtype=np.float64)
        xc = x - x.mean()
        denom = float((xc ** 2).sum())
        slope = (past * xc[:, None]).sum(axis=0) / denom if denom > 0 else np.zeros(R)
        last4 = Q[max(0, h0 - 4): h0]
        if last4.shape[0] < 4:
            last4 = np.concatenate([np.repeat(last4[:1], 4 - last4.shape[0], axis=0), last4], axis=0)
        d_prev = D[h0 - 1]
        d_prev4 = D[h0 - 5] if h0 >= 5 else np.zeros(R)
        dinc4 = d_prev - d_prev4
        y_true = Q[h0: h0 + K_BLOCK].mean(axis=0)
        samples.append({
            "tsw": tsw, "tid": tid, "j": j, "R": R,
            "qmean8": past.mean(axis=0), "qstd8": past.std(axis=0),
            "slope": slope, "last4": last4, "dinc4": dinc4, "base_n": base_n,
            "glob_qmean8": float(past.mean(axis=0).sum()), "glob_dinc4": float(dinc4.sum()),
            "dcum_shard": d_prev, "dcum_total": float(d_prev.sum()), "y": y_true,
        })
    return samples


def _feat_mat(sub: List[Dict[str, Any]], leaky: bool) -> Tuple[np.ndarray, np.ndarray]:
    R = sub[0]["R"]
    eye = np.eye(R, dtype=np.float64)
    xs, ys = [], []
    for r in sub:
        blk = np.column_stack([
            r["qmean8"], r["qstd8"], r["slope"], r["last4"].T, r["dinc4"], r["base_n"],
            np.full(R, r["glob_qmean8"]), np.full(R, r["glob_dinc4"]), eye,
        ])
        if leaky:
            blk = np.hstack([blk, np.column_stack([np.full(R, float(r["j"])), r["dcum_shard"], np.full(R, r["dcum_total"])])])
        xs.append(blk)
        ys.append(r["y"])
    return np.vstack(xs), np.concatenate(ys)


def replay_ops(Q: np.ndarray, D: np.ndarray, policy: str, pmap: Dict[int, np.ndarray]) -> float:
    T, R = Q.shape
    J = T // K_BLOCK
    d_at_reset = np.zeros(R, dtype=np.float64)
    ewma_q = np.zeros(R, dtype=np.float64)
    spent = 0
    rr = 0
    w_tot = 0.0
    future_Q = np.array([Q[j * K_BLOCK:(j + 1) * K_BLOCK].sum(axis=0) for j in range(J)])
    for j in range(J):
        if j > 0:
            prev_q = Q[(j - 1) * K_BLOCK: j * K_BLOCK].sum(axis=0)
            ewma_q = prev_q.copy() if j == 1 else (0.3 * prev_q + 0.7 * ewma_q)
            cur_d = np.maximum(0.0, D[j * K_BLOCK - 1] - d_at_reset)
            sel_r: Optional[int] = None
            if policy == "fixed_c2" and (j - 1) % 2 == 0 and spent < BUDGET:
                sel_r = rr % R
                rr += 1
            elif policy != "noop" and spent < BUDGET:
                if policy == "past_qmass":
                    sc = prev_q * cur_d
                elif policy == "ewma_a03":
                    sc = ewma_q * cur_d
                elif policy in ("gbrt_causal", "gbrt_leaky"):
                    sc = pmap[j] * cur_d
                elif policy == "oracle_1blk":
                    sc = future_Q[j] * cur_d
                else:
                    sc = np.zeros(R)
                br = int(np.argmax(sc))
                if cur_d[br] > 0:
                    sel_r = br
            if sel_r is not None:
                d_at_reset[sel_r] = D[j * K_BLOCK - 1, sel_r]
                spent += 1
        for t in range(j * K_BLOCK, min((j + 1) * K_BLOCK, T)):
            w_tot += float((Q[t] * np.maximum(0.0, D[t] - d_at_reset)).sum())
    return w_tot


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit A: Held-out schedule split-leakage check")
    ap.add_argument("--data-base", default=os.environ.get("DYNVEC_DATA_BASE", "./data"))
    ap.add_argument("--out-dir", default="./results/exp9_review_audits")
    args = ap.parse_args()
    base = Path(args.data_base)
    specs: List[Tuple[int, str, Path]] = [
        (16, "tsw16_abrupt_s05", base / "tsw16/sift100k_u10_q2500_v2/traces/abrupt_s05"),
        (16, "tsw16_abrupt_s06", base / "tsw16/sift100k_u10_q2500_v2/traces/abrupt_s06"),
        (20, "tsw20_abrupt_s05", base / "sift100k_u10_q2500_v2/traces/abrupt_s05"),
        (20, "tsw20_abrupt_s06", base / "sift100k_u10_q2500_v2/traces/abrupt_s06"),
        (20, "tsw20_abrupt_s07", base / "sift100k_u10_q2500_v2/traces/abrupt_s07"),
        (20, "tsw20_abrupt_s08", base / "sift100k_u10_q2500_v2/traces/abrupt_s08"),
        (20, "tsw20_abrupt_s09", base / "sift100k_u10_q2500_v2/traces/abrupt_s09"),
        (24, "tsw24_abrupt_s05", base / "tsw24/sift100k_u10_q2500_v2/traces/abrupt_s05"),
        (24, "tsw24_abrupt_s06", base / "tsw24/sift100k_u10_q2500_v2/traces/abrupt_s06"),
        (28, "tsw28_abrupt_s05", base / "tsw28/sift100k_u10_q2500_v2/traces/abrupt_s05"),
        (28, "tsw28_abrupt_s06", base / "tsw28/sift100k_u10_q2500_v2/traces/abrupt_s06"),
    ]

    qd_map: Dict[str, Dict[str, Any]] = {}
    samp_map: Dict[str, List[Dict[str, Any]]] = {}
    milp_bounds: Dict[str, Tuple[float, float]] = {}
    for tsw, tid, tr_dir in specs:
        qd = extract_qd(tr_dir.parents[1], tr_dir)
        qd_map[tid] = qd
        samp_map[tid] = build_samples((tsw, tid, tr_dir))
        w_noop = float((qd["Q"] * qd["D"]).sum())
        w_star = solve_milp_exact(qd["Q"], qd["D"], K_BLOCK, BUDGET)
        milp_bounds[tid] = (w_noop, w_star)

    def _fit_predict(train_ids: List[str], test_id: str, leaky: bool) -> Tuple[Dict[int, np.ndarray], float]:
        tr_s = [r for id_ in train_ids for r in samp_map[id_]]
        te_s = samp_map[test_id]
        Xtr, ytr = _feat_mat(tr_s, leaky)
        Xte, _ = _feat_mat(te_s, leaky)
        model = lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.06, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1,
        )
        model.fit(Xtr, ytr)
        yhat = np.maximum(0.0, model.predict(Xte))
        R = te_s[0]["R"]
        pmap: Dict[int, np.ndarray] = {}
        switch_top1_hits = 0
        switch_cnt = 0
        tsw_test = te_s[0]["tsw"]
        sw_j = tsw_test // K_BLOCK
        for i, r in enumerate(te_s):
            pred_v = yhat[i * R:(i + 1) * R]
            pmap[int(r["j"])] = pred_v
            if int(r["j"]) == sw_j:
                switch_cnt += 1
                if int(np.argmax(pred_v)) == int(np.argmax(r["y"])):
                    switch_top1_hits += 1
        return pmap, (switch_top1_hits / max(switch_cnt, 1))

    rows: List[Dict[str, Any]] = []
    for tsw_te, tid_te, _ in specs:
        w_noop, w_star = milp_bounds[tid_te]
        # 1. Shared-Schedule Split: train strictly on sibling seeds with same t_sw
        same_tsw_train = [tid for tsw, tid, _ in specs if tsw == tsw_te and tid != tid_te]
        # 2. Held-Out Schedule Split: train strictly on traces with t_sw != t_sw_te
        diff_tsw_train = [tid for tsw, tid, _ in specs if tsw != tsw_te]

        # Non-ML baselines
        for pol in ("fixed_c2", "past_qmass", "ewma_a03", "oracle_1blk"):
            ops = replay_ops(qd_map[tid_te]["Q"], qd_map[tid_te]["D"], pol, {})
            gcr = 100.0 * (w_noop - ops) / max(w_noop - w_star, 1e-9)
            rows.append({
                "split_protocol": "schedule_invariant",
                "tsw": tsw_te, "trace": tid_te, "policy": pol,
                "exact_scan_ops": round(ops, 1), "gcr_pct": round(gcr, 2),
                "switch_block_top1": np.nan,
            })

        for split_name, tr_ids in [("shared_schedule", same_tsw_train), ("heldout_schedule", diff_tsw_train)]:
            for pol, is_leaky in [("gbrt_leaky", True), ("gbrt_causal", False)]:
                pmap, sw_t1 = _fit_predict(tr_ids, tid_te, is_leaky)
                ops = replay_ops(qd_map[tid_te]["Q"], qd_map[tid_te]["D"], pol, pmap)
                gcr = 100.0 * (w_noop - ops) / max(w_noop - w_star, 1e-9)
                rows.append({
                    "split_protocol": split_name,
                    "tsw": tsw_te, "trace": tid_te, "policy": pol,
                    "exact_scan_ops": round(ops, 1), "gcr_pct": round(gcr, 2),
                    "switch_block_top1": round(sw_t1, 3),
                })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / "heldout_schedule_trap2.csv"
    keys = list(rows[0].keys())
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Aggregate summary
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby(["split_protocol", "policy"], as_index=False)[["exact_scan_ops", "gcr_pct", "switch_block_top1"]].mean()
    summary = {
        "n_traces": len(specs),
        "tsw_values": [16, 20, 24, 28],
        "summary": agg.to_dict(orient="records"),
    }
    (out_dir / "heldout_schedule_trap2.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
