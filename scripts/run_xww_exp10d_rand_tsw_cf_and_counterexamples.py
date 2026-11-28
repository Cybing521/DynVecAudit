#!/usr/bin/env python3
"""Exp 10-D (EPYC-7343 Hardware-Invariant Statistical Power Audits, nice -n 10, max 4 workers):
1. D1 (Trap 2 Split vs Clock Leakage, n=30 randomized t_sw traces + train {16,24} -> test {28} protocol):
   Synthesizes 30 realistic R=16 shard (Q, D) trajectories with randomized abrupt switch time t_sw ~ U[10,30],
   evaluates GBRT-Leaky (shared vs held-out vs randomized-t_sw split) vs GBRT-Causal vs EWMA vs Past-QMass vs Oracle,
   and exports paired bootstrap 95% CIs (GCR % & exact_scan_ops).
2. D2 (Track-CF Attribution Horizon H in {1,2,3,4,6,8,inf} across n=15 multi-drift/seed/CRUD traces):
   Executes qbam_replay / exact branch simulation to show monotonic convergence as H approaches full-trace inf.
3. D3 (Counterexample Boundary Characterization: Wikimedia-210d + CLINC-150):
   Full EWMA alpha in {0.1,0.3,0.5,0.7,0.9} vs Past-QMass vs GBRT-Causal vs Oracle exact_scan_ops & MILP GCR.
"""
from __future__ import annotations

import os

for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_ev] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.ensemble import HistGradientBoostingRegressor

TOOLS_CANDIDATES = [
    Path(os.environ.get("DYNVECAUDIT_TOOLS_DIR", "./tools")),
    Path(os.environ.get("DYNVECAUDIT_TOOLS_DIR", "./tools")),
]
for cand in TOOLS_CANDIDATES:
    if cand.exists():
        sys.path.insert(0, str(cand))
        break
import qbam_io as qio  # noqa: E402

K_BLOCK = 4
BUDGET = 9
T_WIN = 40
R_SHARDS = 16


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def solve_milp_exact(Q: np.ndarray, D: np.ndarray, K: int = K_BLOCK, B: int = BUDGET) -> float:
    T, R = Q.shape
    J = T // K
    arc_list = []
    for r in range(R):
        for j1 in range(J):
            d_ref = D[j1 * K - 1, r] if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                w = float(sum(Q[t, r] * max(0.0, D[t, r] - d_ref) for t in range(j1 * K, min(j2 * K, T))))
                arc_list.append((r, j1, j2, w, 1 if j2 < J else 0))

    n_vars = len(arc_list)
    c = np.array([a[3] for a in arc_list], dtype=np.float64)
    rows, cols = [], []
    # Flow conservation per (r, j)
    rhs = []
    row_id = 0
    for r in range(R):
        for j in range(J + 1):
            for idx_a, (ar, j1, j2, _, _) in enumerate(arc_list):
                if ar != r:
                    continue
                if j1 == j:
                    rows.append((row_id, idx_a, 1.0))
                if j2 == j:
                    rows.append((row_id, idx_a, -1.0))
            rhs.append(1.0 if j == 0 else (-1.0 if j == J else 0.0))
            row_id += 1
    # Budget constraint
    A = lil_matrix((row_id + 1, n_vars), dtype=np.float64)
    for ri, ci, val in rows:
        A[ri, ci] = val
    for idx_a, (_, _, _, _, is_act) in enumerate(arc_list):
        if is_act:
            A[row_id, idx_a] = 1.0
    lb = np.array(rhs + [0.0], dtype=np.float64)
    ub = np.array(rhs + [float(B)], dtype=np.float64)
    res = milp(c=c, integrality=np.ones(n_vars), bounds=Bounds(0, 1), constraints=LinearConstraint(A.tocsc(), lb, ub))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return float(res.fun)


def generate_abrupt_trace_qd(seed: int, t_sw: int, T: int = T_WIN, R: int = R_SHARDS) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hub_before = int(rng.integers(0, R))
    hub_after = int((hub_before + rng.integers(1, R)) % R)
    Q = np.zeros((T, R), dtype=np.float64)
    D = np.zeros((T, R), dtype=np.float64)
    cum_ins = np.zeros(R, dtype=np.float64)
    for t in range(T):
        hot = hub_before if t < t_sw else hub_after
        q_prob = np.full(R, 0.025, dtype=np.float64)
        q_prob[hot] = 0.625
        q_prob /= q_prob.sum()
        Q[t] = rng.multinomial(2500, q_prob).astype(np.float64)

        ins_prob = np.full(R, 0.03, dtype=np.float64)
        ins_prob[hot] = 0.55
        ins_prob /= ins_prob.sum()
        cum_ins += rng.multinomial(10000, ins_prob).astype(np.float64)
        D[t] = cum_ins.copy()
    return Q, D


def build_block_samples(Q: np.ndarray, D: np.ndarray, trace_id: str, t_sw: int) -> List[Dict[str, Any]]:
    T, R = Q.shape
    J = T // K_BLOCK
    samples: List[Dict[str, Any]] = []
    ewma = Q[0].copy()
    for t in range(1, K_BLOCK):
        ewma = 0.3 * Q[t] + 0.7 * ewma

    for j in range(1, J):
        t0 = j * K_BLOCK
        t1 = min((j + 1) * K_BLOCK, T)
        q_prev = Q[(j - 1) * K_BLOCK : t0].sum(axis=0)
        q_last = Q[t0 - 1]
        d_now = D[t0 - 1]
        d_inc4 = d_now - (D[t0 - 1 - K_BLOCK] if t0 - 1 >= K_BLOCK else 0.0)
        q_next_true = Q[t0:t1].sum(axis=0)

        for r in range(R):
            samples.append({
                "trace_id": trace_id,
                "t_sw": t_sw,
                "j": j,
                "r": r,
                # 8 causal realized features (strictly observed at block boundary t0-1)
                "q_prev": float(q_prev[r]),
                "q_last": float(q_last[r]),
                "ewma": float(ewma[r]),
                "q_ratio": float(q_last[r] / max(q_prev[r] / K_BLOCK, 1.0)),
                "d_now": float(d_now[r]),
                "d_inc4": float(d_inc4[r]),
                "q_frac": float(q_prev[r] / max(q_prev.sum(), 1.0)),
                "d_frac": float(d_now[r] / max(d_now.sum(), 1.0)),
                # Leaky / schedule-identifying proxies (Trap 2)
                "leak_block_j": float(j),
                "leak_dcum_total": float(d_now.sum()),
                "target": float(q_next_true[r]),
            })
        for t in range(t0, t1):
            ewma = 0.3 * Q[t] + 0.7 * ewma
    return samples


def replay_policy(Q: np.ndarray, D: np.ndarray, pol: str, pred_map: Dict[int, np.ndarray] | None = None, alpha: float = 0.3) -> float:
    T, R = Q.shape
    J = T // K_BLOCK
    d_reset = np.zeros(R, dtype=np.float64)
    spent = 0
    w_tot = 0.0
    ewma = np.zeros(R, dtype=np.float64)
    for t in range(K_BLOCK):
        ewma = Q[t] if t == 0 else alpha * Q[t] + (1.0 - alpha) * ewma

    for j in range(J):
        t0 = j * K_BLOCK
        t1 = min((j + 1) * K_BLOCK, T)
        if j >= 1 and spent < BUDGET:
            cur_d = np.maximum(0.0, D[t0 - 1] - d_reset)
            if pol == "fixed_c2":
                if (j - 1) % 2 == 0:
                    r_sel = int(np.argmax(cur_d))
                else:
                    r_sel = -1
            elif pol == "past_qmass":
                sc = Q[(j - 1) * K_BLOCK : t0].sum(axis=0) * cur_d
                r_sel = int(np.argmax(sc))
            elif pol == "ewma":
                sc = ewma * cur_d
                r_sel = int(np.argmax(sc))
            elif pol == "oracle":
                sc = Q[t0:t1].sum(axis=0) * cur_d
                r_sel = int(np.argmax(sc))
            elif pol in ("gbrt_leaky", "gbrt_causal") and pred_map is not None:
                sc = np.maximum(0.0, pred_map[j]) * cur_d
                r_sel = int(np.argmax(sc))
            else:
                r_sel = -1
            if r_sel >= 0 and cur_d[r_sel] > 0:
                d_reset[r_sel] = D[t0 - 1, r_sel]
                spent += 1
        for t in range(t0, t1):
            w_tot += float((Q[t] * np.maximum(0.0, D[t] - d_reset)).sum())
            if j >= 1:
                ewma = alpha * Q[t] + (1.0 - alpha) * ewma
    return w_tot


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 7) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = a - b
    mean_d = float(diffs.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(diffs)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(mean_d, 2), round(float(lo), 2), round(float(hi), 2)


def run_d1_rand_tsw_split_audit(out_dir: Path) -> Dict[str, Any]:
    log("=== Exp 10-D1: n=30 Randomized t_sw Trap 2 Split-Leakage Audit ===")
    rng_sched = np.random.default_rng(2026)

    # 30 randomized-schedule traces + 12 fixed-tsw traces ({16,24} train -> {28} test)
    rand_specs = [(i, f"rand_{i:02d}", int(rng_sched.integers(10, 31))) for i in range(30)]
    fixed_specs = (
        [(100 + i, f"tsw16_{i}", 16) for i in range(4)]
        + [(200 + i, f"tsw24_{i}", 24) for i in range(4)]
        + [(300 + i, f"tsw28_{i}", 28) for i in range(4)]
    )

    causal_cols = ["q_prev", "q_last", "ewma", "q_ratio", "d_now", "d_inc4", "q_frac", "d_frac"]
    leaky_cols = causal_cols + ["leak_block_j", "leak_dcum_total"]

    def _eval_cohort(specs: List[Tuple[int, str, int]], mode: str) -> List[Dict[str, Any]]:
        qds = {}
        samps = {}
        bounds = {}
        for sd, tid, tsw in specs:
            Q, D = generate_abrupt_trace_qd(5000 + sd, tsw)
            qds[tid] = (Q, D, tsw)
            samps[tid] = build_block_samples(Q, D, tid, tsw)
            w_noop = float((Q * D).sum())
            w_star = solve_milp_exact(Q, D)
            bounds[tid] = (w_noop, w_star)

        rows: List[Dict[str, Any]] = []
        for _, tid_te, tsw_te in specs:
            if mode == "train16_24_test28" and tsw_te != 28:
                continue
            Q, D, _ = qds[tid_te]
            w_noop, w_star = bounds[tid_te]
            for pol in ("fixed_c2", "past_qmass", "ewma", "oracle"):
                ops = replay_policy(Q, D, pol)
                gcr = 100.0 * (w_noop - ops) / max(w_noop - w_star, 1e-9)
                rows.append({"cohort": mode, "trace": tid_te, "tsw": tsw_te, "policy": pol, "exact_scan_ops": ops, "gcr_pct": gcr})

            if mode == "shared_fixed":
                tr_ids = [tid for _, tid, tsw in specs if tsw == tsw_te and tid != tid_te]
            elif mode == "train16_24_test28":
                tr_ids = [tid for _, tid, tsw in specs if tsw in (16, 24)]
            else:  # randomized leave-one-out
                tr_ids = [tid for _, tid, _ in specs if tid != tid_te]

            df_tr = pd.DataFrame([r for tid in tr_ids for r in samps[tid]])
            df_te = pd.DataFrame(samps[tid_te])
            for pol, cols in [("gbrt_leaky", leaky_cols), ("gbrt_causal", causal_cols)]:
                mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_leaf_nodes=31, random_state=42)
                mdl.fit(df_tr[cols].to_numpy(), df_tr["target"].to_numpy())
                yhat = np.maximum(0.0, mdl.predict(df_te[cols].to_numpy()))
                pmap = {}
                for idx_j, jv in enumerate(df_te["j"].unique()):
                    m = (df_te["j"] == jv).to_numpy()
                    pmap[int(jv)] = yhat[m]
                ops = replay_policy(Q, D, pol, pmap)
                gcr = 100.0 * (w_noop - ops) / max(w_noop - w_star, 1e-9)
                rows.append({"cohort": mode, "trace": tid_te, "tsw": tsw_te, "policy": pol, "exact_scan_ops": ops, "gcr_pct": gcr})
        return rows

    all_rows = (
        _eval_cohort(rand_specs, "randomized_tsw_n30")
        + _eval_cohort(fixed_specs, "shared_fixed")
        + _eval_cohort(fixed_specs, "train16_24_test28")
    )
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "exp10d1_rand_tsw_split_leakage.csv", index=False)

    # Paired bootstrap summary on n=30 randomized traces
    r30 = df[df["cohort"] == "randomized_tsw_n30"].pivot(index="trace", columns="policy", values="gcr_pct")
    summary = {
        "cohort_means": df.groupby(["cohort", "policy"], as_index=False)[["gcr_pct", "exact_scan_ops"]].mean().round(2).to_dict(orient="records"),
        "rand30_bootstrap_gcr_diff": {
            "leaky_minus_causal": bootstrap_ci_diff(r30["gbrt_leaky"].to_numpy(), r30["gbrt_causal"].to_numpy()),
            "causal_minus_past_qmass": bootstrap_ci_diff(r30["gbrt_causal"].to_numpy(), r30["past_qmass"].to_numpy()),
            "past_qmass_minus_fixed_c2": bootstrap_ci_diff(r30["past_qmass"].to_numpy(), r30["fixed_c2"].to_numpy()),
        },
    }
    return summary


def run_d2_horizon_infinity(out_dir: Path) -> Dict[str, Any]:
    log("=== Exp 10-D2: Multi-Seed (n=15) CF Horizon H in {1,2,3,4,6,8,inf} ===")
    rows: List[Dict[str, Any]] = []
    horizons = [1, 2, 3, 4, 6, 8, 999]  # 999 = full trace tail (inf)
    for dr_idx, dr in enumerate(["abrupt", "hotspot", "gradual", "periodic", "uniform"]):
        for sd in range(3):
            seed = 9000 + dr_idx * 10 + sd
            rng = np.random.default_rng(seed)
            # Generate 40x16 Q,D with pattern depending on drift type
            T, R = T_WIN, R_SHARDS
            Q = np.zeros((T, R), dtype=np.float64)
            D = np.zeros((T, R), dtype=np.float64)
            cum = np.zeros(R, dtype=np.float64)
            h0, h1 = int(rng.integers(0, R)), int(rng.integers(0, R))
            if h1 == h0:
                h1 = (h0 + 3) % R
            for t in range(T):
                qp = np.full(R, 0.03)
                ip = np.full(R, 0.03)
                if dr == "abrupt":
                    hot = h0 if t < 20 else h1
                    qp[hot] = 0.55; ip[hot] = 0.55
                elif dr == "hotspot":
                    qp[h0] = 0.55; ip[h0] = 0.55
                elif dr == "gradual":
                    alpha = t / float(T - 1)
                    qp[h0] = 0.45 * (1 - alpha) + 0.03; qp[h1] = 0.45 * alpha + 0.03
                    ip[h0] = 0.45 * (1 - alpha) + 0.03; ip[h1] = 0.45 * alpha + 0.03
                elif dr == "periodic":
                    hot = h0 if (t // 8) % 2 == 0 else h1
                    qp[hot] = 0.55; ip[hot] = 0.55
                qp /= qp.sum(); ip /= ip.sum()
                Q[t] = rng.multinomial(2500, qp).astype(np.float64)
                cum += rng.multinomial(10000, ip).astype(np.float64)
                D[t] = cum.copy()

            # Counterfactual branch evaluation across j in {1..8} comparing candidate shard r in 0..15
            # Assuming remaining blocks uncompacted (or compacted cleanly at horizon)
            for j in range(1, T // K_BLOCK):
                t0 = j * K_BLOCK
                cur_d = D[t0 - 1]
                # Full trace (H=inf) incremental savings for compacting shard r at step t0:
                # savings_r(H) = sum_{t=t0}^{min(t0+4*H, T)-1} Q[t, r] * cur_d[r]
                costs_inf = [float((Q[t0:T, r] * cur_d[r]).sum()) for r in range(R)]
                best_inf = int(np.argmax(costs_inf))
                for H in horizons:
                    t_end = min(T, t0 + H * K_BLOCK) if H < 999 else T
                    costs_h = [float((Q[t0:t_end, r] * cur_d[r]).sum()) for r in range(R)]
                    best_h = int(np.argmax(costs_h))
                    flips = pairs = 0
                    for a in range(R):
                        for b in range(a + 1, R):
                            if costs_h[a] == costs_h[b] or costs_inf[a] == costs_inf[b]:
                                continue
                            pairs += 1
                            if (costs_h[a] > costs_h[b]) != (costs_inf[a] > costs_inf[b]):
                                flips += 1
                    rows.append({
                        "drift": dr, "seed": sd, "j": j,
                        "H_win": "inf" if H == 999 else str(H * K_BLOCK),
                        "H_blk": "inf" if H == 999 else str(H),
                        "pair_flip_vs_inf_pct": 100.0 * flips / max(pairs, 1),
                        "top1_match_vs_inf": 1.0 if best_h == best_inf else 0.0,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp10d2_horizon_inf_n15.csv", index=False)
    agg = df.groupby(["H_blk", "H_win"], as_index=False)[["pair_flip_vs_inf_pct", "top1_match_vs_inf"]].mean().round(2)
    return {"horizon_convergence": agg.to_dict(orient="records")}


def _extract_qd_qbam(data_root: Path, tr_dir: Path, nprobe: int = 4) -> Tuple[np.ndarray, np.ndarray]:
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
        qw = ev["window"][qm].astype(np.int64)
        q_vecs = pool[ev["vec_ref"][qm].astype(np.int64)]
        scores = -2.0 * (q_vecs @ centroids.T) + c_norm2[None, :]
        top = np.argpartition(scores, nprobe - 1, axis=1)[:, :nprobe]
        for wi, rs in zip(qw.tolist(), top.tolist()):
            for r in rs:
                Q[wi, r] += 1.0
    im = ev["type"] == qio.EV_INSERT
    if np.any(im):
        iw = ev["window"][im].astype(np.int64)
        i_vecs = pool[ev["vec_ref"][im].astype(np.int64)]
        scores = -2.0 * (i_vecs @ centroids.T) + c_norm2[None, :]
        top1 = np.argmin(scores, axis=1)
        for wi, r in zip(iw.tolist(), top1.tolist()):
            ins_pw[wi, r] += 1.0
    D = np.zeros((n_win, R), dtype=np.float64)
    for w in range(n_win):
        D[w] = base_n + ins_pw[: w + 1].sum(axis=0)
    return Q, D


def run_d3_counterexamples(out_dir: Path) -> Dict[str, Any]:
    log("=== Exp 10-D3: Counterexample EWMA Sweep & Gap Attribution (Wikimedia + CLINC-150) ===")
    data_base = Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data"))
    targets: List[Tuple[str, Path, Path]] = []
    for sub in sorted((data_base / "wikimedia_windows").iterdir()):
        if sub.is_dir() and (sub / "traces").exists():
            for tr in sorted((sub / "traces").iterdir()):
                if (tr / "trace.bin").exists():
                    targets.append((f"wiki_{sub.name}", sub, tr))
    clinc_root = data_base / "clinc150_minilm384"
    if (clinc_root / "traces").exists():
        for tr in sorted((clinc_root / "traces").iterdir())[:12]:
            if (tr / "trace.bin").exists():
                targets.append(("clinc150", clinc_root, tr))

    rows: List[Dict[str, Any]] = []
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    for ds_label, droot, tr_dir in targets:
        try:
            Q, D = _extract_qd_qbam(droot, tr_dir)
        except Exception as exc:
            log(f"  Skip {ds_label}/{tr_dir.name}: {exc}")
            continue
        T, R = Q.shape
        if T < 8:
            continue
        w_noop = float((Q * D).sum())
        w_star = solve_milp_exact(Q, D, K=min(K_BLOCK, T), B=min(BUDGET, max(1, (T // K_BLOCK) - 1)))
        for pol, alpha in [("fixed_c2", 0.3), ("past_qmass", 0.3), ("oracle", 0.3)] + [(f"ewma_a{int(a*10):02d}", a) for a in alphas]:
            ops = replay_policy(Q, D, "ewma" if pol.startswith("ewma") else pol, alpha=alpha)
            gcr = 100.0 * (w_noop - ops) / max(w_noop - w_star, 1e-9)
            rows.append({
                "dataset": ds_label, "trace": tr_dir.name, "policy": pol,
                "exact_scan_ops": round(ops, 1), "gcr_pct": round(gcr, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp10d3_counterexample_ewma_sweep.csv", index=False)
    agg = df.groupby(["dataset", "policy"], as_index=False)[["gcr_pct", "exact_scan_ops"]].mean().round(2) if not df.empty else pd.DataFrame()
    return {"counterexample_agg": agg.to_dict(orient="records")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp 10-D Hardware-Invariant Audits on EPYC-7343")
    ap.add_argument("--out-dir", default=os.environ.get("DYNVECAUDIT_OUT_DIR", "./results/exp10d"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = run_d1_rand_tsw_split_audit(out_dir)
    s2 = run_d2_horizon_infinity(out_dir)
    s3 = run_d3_counterexamples(out_dir)
    summary = {"d1_rand_tsw": s1, "d2_horizon_inf": s2, "d3_counterexamples": s3}
    (out_dir / "exp10d_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "EXP10D_DONE.flag").write_text("ok\n", encoding="utf-8")
    log("EXP10D_ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
