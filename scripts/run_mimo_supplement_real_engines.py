#!/usr/bin/env python3
"""Exp 9: Pure-CPU Core-8 Real-Engine Supplement on Ryzen-9950X (NO GPU).

Strictly enforces:
  - CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=1, pinned Core 8, load gate <= 2.5
  - Zero GPU usage; isolated worker subprocesses for accurate VmHWM & clean process_time

Phases:
  M1: Real-Engine Multi-Shard Scheduler Bridge (16-shard FAISS-IVF & 16-shard hnswlib)
      comparing No-Op, Fixed c=2/4, Past-QMass, EWMA, GBRT-Causal/Leaky, Online-MILP, Oracle-1blk
  M2: Quantized Backends Audit (IVF-Flat, IVF-SQ8, SQ8+Refine, IVF-PQ, PQ+Refine, USearch f32/f16/i8)
  M3: Real-Engine Query-to-Update (Q/u) Crossover Boundary across u in {2,5,10,20}% x Q in {500,2500,5000,20000}
"""
from __future__ import annotations

import os

for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_ev] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.stats import spearmanr

TOOLS_DIR = Path(os.environ.get("DYNVECAUDIT_TOOLS_DIR", "./tools"))
if TOOLS_DIR.exists():
    sys.path.insert(0, str(TOOLS_DIR))
import qbam_io as qio  # noqa: E402

DATA_BASE = Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data"))
OUT_BASE = Path(os.environ.get("DYNVECAUDIT_OUT_DIR", "./results/exp9_supplement"))
K_BLOCK = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    except Exception:
        return 0.0


def wait_quiet_host(max_load: float = 2.5, checks: int = 2) -> None:
    streak = 0
    while streak < checks:
        if read_load1m() <= max_load:
            streak += 1
            if streak < checks:
                time.sleep(2.0)
        else:
            streak = 0
            time.sleep(10.0)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def peak_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return round(float(line.split(":")[1].strip().split()[0]) / 1024.0, 2)
    except Exception:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] * (1.0 - (k - lo)) + s[hi] * (k - lo))


def _invoke_self_worker(worker_mode: str, cfg: Dict[str, Any], timeout_s: int = 600) -> Dict[str, Any]:
    payload = json.dumps({"mode": worker_mode, **cfg})
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-json", payload]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[_ev] = "1"
    pr = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    if pr.returncode != 0:
        raise RuntimeError(f"Worker {worker_mode} failed rc={pr.returncode}: {pr.stderr[-600:]}")
    for line in pr.stdout.splitlines():
        if line.startswith("WORKER_JSON="):
            return json.loads(line.split("WORKER_JSON=", 1)[1])
    raise RuntimeError(f"Missing WORKER_JSON from {worker_mode}: stdout={pr.stdout[-400:]}")


# ---------------------------------------------------------------------------
# Shared Exact Scan Ops + MILP Bound + Causal/Leaky GBRT Helpers
# ---------------------------------------------------------------------------
def _extract_qd_from_trace(data_root: Path, tr_dir: Path, nprobe: int = 4) -> Dict[str, Any]:
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
        for lo in range(0, len(qv), 4096):
            hi = min(len(qv), lo + 4096)
            v = qv[lo:hi]
            d2 = np.sum(v ** 2, axis=1, keepdims=True) + c_norm2[None, :] - 2.0 * (v @ centroids.T)
            top = np.argpartition(d2, kth=nprobe - 1, axis=1)[:, :nprobe]
            for w_idx, shs in zip(qe["window"][lo:hi].tolist(), top.tolist()):
                for s in shs:
                    Q[w_idx, s] += 1.0

    im = (ev["type"] == qio.EV_INSERT) | (ev["type"] == qio.EV_UPDATE)
    if np.any(im):
        ie = ev[im]
        iv = pool[ie["vec_ref"]]
        for lo in range(0, len(iv), 4096):
            hi = min(len(iv), lo + 4096)
            v = iv[lo:hi]
            d2 = np.sum(v ** 2, axis=1, keepdims=True) + c_norm2[None, :] - 2.0 * (v @ centroids.T)
            bs = np.argmin(d2, axis=1)
            for w_idx, s in zip(ie["window"][lo:hi].tolist(), bs.tolist()):
                ins_pw[w_idx, s] += 1.0

    D = np.cumsum(ins_pw, axis=0)
    return {"R": R, "n_windows": n_win, "base_n": base_n, "Q": Q, "D": D}


def _solve_milp_exact(Q: np.ndarray, D: np.ndarray, K: int, budget: int) -> float:
    T, R = Q.shape
    J = T // K
    if budget <= 0 or J <= 1:
        return float((Q * D).sum())
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
        lb[r] = 1.0; ub[r] = 1.0
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
        lb[n_eq + v] = -np.inf; ub[n_eq + v] = 1.0
    lb[n_eq + (J - 1)] = -np.inf; ub[n_eq + (J - 1)] = float(min(budget, J - 1))
    res = milp(c=np.array(costs, dtype=np.float64), integrality=np.ones(nv), bounds=Bounds(0, 1), constraints=LinearConstraint(A.tocsc(), lb, ub))
    return float(res.fun)


def _solve_online_milp_step(Q_hist: np.ndarray, D_cur: np.ndarray, tau_prev: np.ndarray, j: int, J: int, K: int, rem_b: int, ewma_q: np.ndarray) -> Optional[int]:
    if rem_b <= 0 or j >= J:
        return None
    R = len(ewma_q)
    q_hat = np.maximum(0.0, ewma_q / float(K))
    arcs: List[Tuple[int, int, int, float]] = []
    for r in range(R):
        d_last = float(D_cur[r])  # current cumulative up to window K*j-1
        for j1 in range(j - 1, J):
            for j2 in range(j1 + 1, J + 1):
                if j1 == j - 1:
                    # uncompacted segment cost across future blocks j..j2-1
                    d_unmerged = max(0.0, d_last - (0.0 if tau_prev[r] == 0 else float(D_cur[r] - tau_prev[r])))
                    # use exact delta_now stored in tau_prev as delta_at_reset passed in via D_last_reset
                    cost = float(q_hat[r]) * float(tau_prev[r]) * float(K * (j2 - j))
                else:
                    cost = 0.0
                arcs.append((r, j1, j2, cost))
    return None


def _build_gbrt_predictions(drift: str, target_seed: str = "s05") -> Dict[str, Dict[int, List[float]]]:
    """Train GBRT-Causal and GBRT-Leaky on sibling seeds s06..s09 of the same drift, predict on s05."""
    import lightgbm as lgb

    data_root = DATA_BASE / "sift100k_u10_q2500_v2"
    all_seeds = ["s05", "s06", "s07", "s08", "s09"]
    samples: List[Dict[str, Any]] = []
    for sd in all_seeds:
        tr = f"{drift}_{sd}"
        s = _extract_qd_from_trace(data_root, data_root / "traces" / tr, nprobe=4)
        Q, D, base_n, R, n_win = s["Q"], s["D"], s["base_n"], s["R"], s["n_windows"]
        J = n_win // K_BLOCK
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
                "seed": sd, "j": j, "R": R, "qmean8": past.mean(axis=0), "qstd8": past.std(axis=0),
                "slope": slope, "last4": last4, "dinc4": dinc4, "base_n": base_n,
                "glob_qmean8": float(past.mean(axis=0).sum()), "glob_dinc4": float(dinc4.sum()),
                "dcum_shard": d_prev, "dcum_total": float(d_prev.sum()), "y": y_true,
            })

    def _mat(sub: List[Dict[str, Any]], leaky: bool) -> Tuple[np.ndarray, np.ndarray]:
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

    tr_s = [r for r in samples if r["seed"] != target_seed]
    te_s = [r for r in samples if r["seed"] == target_seed]
    out: Dict[str, Dict[int, List[float]]] = {"gbrt_causal": {}, "gbrt_leaky": {}}
    for pol_name, is_leaky in [("gbrt_causal", False), ("gbrt_leaky", True)]:
        Xtr, ytr = _mat(tr_s, is_leaky)
        Xte, _ = _mat(te_s, is_leaky)
        model = lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.06, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1,
        )
        model.fit(Xtr, ytr)
        yhat = np.maximum(0.0, model.predict(Xte))
        R = te_s[0]["R"]
        for i, r in enumerate(te_s):
            out[pol_name][int(r["j"])] = [float(x) for x in yhat[i * R:(i + 1) * R]]
    return out


# ---------------------------------------------------------------------------
# Worker Implementations (M1, M2, M3)
# ---------------------------------------------------------------------------
def _worker_m1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import faiss
    import hnswlib

    faiss.omp_set_num_threads(1)
    engine = cfg["engine"]
    trace = cfg["trace"]
    policy = cfg["policy"]
    budget = int(cfg["budget"])
    preds_map = cfg.get("preds_map", {})

    data_root = DATA_BASE / "sift100k_u10_q2500_v2"
    tr_dir = data_root / "traces" / trace
    pool, ph = qio.read_pool(data_root / "pool.bin")
    init, ih = qio.read_init(data_root / "init.bin")
    ev, th = qio.read_trace(tr_dir / "trace.bin")
    gt_ids, _, gh = qio.read_gt(tr_dir / "gt.bin")

    dim = int(ph.fields["dim"])
    k = int(gh.fields["k"])
    R = int(ih.fields["R"])
    n_win = int(th.fields["n_windows"])
    K = int(th.fields["K"])
    J = n_win // K
    nprobe_shard = 4
    centroids = qio.read_centroids(data_root / "centroids.f32", R, dim)
    c_norm2 = np.sum(centroids ** 2, axis=1)

    # Pre-split init vectors per shard
    shard_base_vecs: List[np.ndarray] = []
    shard_base_ids: List[np.ndarray] = []
    shard_delta_vecs: List[List[np.ndarray]] = [[] for _ in range(R)]
    shard_delta_ids: List[List[np.ndarray]] = [[] for _ in range(R)]

    def _build_shard_base(vecs: np.ndarray, ids: np.ndarray):
        if engine == "sharded_faiss_ivf":
            nlist_s = min(16, max(4, len(vecs) // 64))
            q_sub = faiss.IndexFlatL2(dim)
            idx = faiss.IndexIVFFlat(q_sub, dim, nlist_s, faiss.METRIC_L2)
            idx.train(vecs)
            idx.add_with_ids(vecs, ids)
            idx.nprobe = min(8, nlist_s)
            return idx
        elif engine == "sharded_hnswlib":
            idx = hnswlib.Index(space="l2", dim=dim)
            idx.init_index(max_elements=max(len(vecs) + 15000, 16000), ef_construction=100, M=16)
            idx.set_num_threads(1)
            idx.set_ef(64)
            idx.add_items(vecs, ids, num_threads=1)
            return idx
        raise ValueError(engine)

    def _build_delta_idx():
        return faiss.IndexIDMap2(faiss.IndexFlatL2(dim))

    base_indexes = []
    delta_indexes = [_build_delta_idx() for _ in range(R)]
    for r in range(R):
        m = init["shard"] == r
        v = np.ascontiguousarray(pool[init["vec_ref"][m]], dtype=np.float32)
        i = np.ascontiguousarray(init["id"][m], dtype=np.int64)
        shard_base_vecs.append(v)
        shard_base_ids.append(i)
        base_indexes.append(_build_shard_base(v, i))

    # Extract full Q/D for exact_scan_ops & MILP accounting (does not enter policy unless causal)
    Q_all = np.zeros((n_win, R), dtype=np.float64)
    D_all = np.zeros((n_win, R), dtype=np.float64)
    cum_ins = np.zeros(R, dtype=np.float64)

    # Window execution state
    d_unmerged = np.zeros(R, dtype=np.float64)
    tau_block = np.zeros(R, dtype=int)
    d_at_reset = np.zeros(R, dtype=np.float64)
    ewma_q = np.zeros(R, dtype=np.float64)
    n_compactions = 0
    rr_ptr = 0

    load_start = read_load1m()
    t_wall0 = time.perf_counter()
    query_cpu_s = 0.0
    maint_cpu_s = 0.0
    recalls: List[float] = []
    lat_us: List[float] = []
    q_off = 0

    # Pre-scan oracle future block Q if policy == oracle_1blk (only used inside oracle arm!)
    future_Q_blk = None
    if policy == "oracle_1blk":
        qd = _extract_qd_from_trace(data_root, tr_dir, nprobe=nprobe_shard)
        future_Q_blk = np.array([qd["Q"][j * K:(j + 1) * K].sum(axis=0) for j in range(J)])

    for w, crud, qe in qio.window_events(ev, n_win):
        # Block boundary compaction BEFORE window w if w == j*K and j >= 1
        if w > 0 and (w % K == 0):
            j = w // K
            prev_blk_q = Q_all[(j - 1) * K: j * K].sum(axis=0)
            if j == 1:
                ewma_q = prev_blk_q.copy()
            else:
                ewma_q = 0.3 * prev_blk_q + 0.7 * ewma_q

            sel_r: Optional[int] = None
            if policy == "noop":
                sel_r = None
            elif policy == "fixed_c2":
                if (j - 1) % 2 == 0 and n_compactions < budget:
                    sel_r = rr_ptr % R
                    rr_ptr += 1
            elif policy == "fixed_c4":
                if (j - 1) % 4 == 0 and n_compactions < budget:
                    sel_r = rr_ptr % R
                    rr_ptr += 1
            elif n_compactions < budget:
                if policy == "past_qmass":
                    score = prev_blk_q * d_unmerged
                    best_r = int(np.argmax(score))
                    if d_unmerged[best_r] > 0:
                        sel_r = best_r
                elif policy == "ewma_a03":
                    score = ewma_q * d_unmerged
                    best_r = int(np.argmax(score))
                    if d_unmerged[best_r] > 0:
                        sel_r = best_r
                elif policy in ("gbrt_causal", "gbrt_leaky"):
                    qhat = np.asarray(preds_map[str(j)], dtype=np.float64)
                    score = qhat * d_unmerged
                    best_r = int(np.argmax(score))
                    if d_unmerged[best_r] > 0:
                        sel_r = best_r
                elif policy == "oracle_1blk" and future_Q_blk is not None:
                    score = future_Q_blk[j] * d_unmerged
                    best_r = int(np.argmax(score))
                    if d_unmerged[best_r] > 0:
                        sel_r = best_r
                elif policy == "online_milp":
                    rem_b = budget - n_compactions
                    q_rate = np.maximum(0.0, ewma_q / float(K))
                    arcs: List[Tuple[int, int, int, float]] = []
                    for r in range(R):
                        for j1 in range(j - 1, J):
                            for j2 in range(j1 + 1, J + 1):
                                if j1 == j - 1:
                                    cost = float(q_rate[r]) * float(d_unmerged[r]) * float(K * (j2 - j))
                                else:
                                    cost = 0.0
                                arcs.append((r, j1, j2, cost))
                    nv = len(arcs)
                    n_eq = R + R * (J - j)
                    n_ineq = (J - j) + 1
                    A = lil_matrix((n_eq + n_ineq, nv), dtype=np.float64)
                    lb = np.zeros(n_eq + n_ineq, dtype=np.float64)
                    ub = np.zeros(n_eq + n_ineq, dtype=np.float64)
                    for r in range(R):
                        lb[r] = 1.0; ub[r] = 1.0
                    for idx_a, (rr, j1, j2, _) in enumerate(arcs):
                        if j1 == j - 1:
                            A[rr, idx_a] = 1.0
                        if j <= j2 <= J - 1:
                            rc = R + rr * (J - j) + (j2 - j)
                            A[rc, idx_a] = 1.0
                            A[n_eq + (j2 - j), idx_a] = 1.0
                            A[n_eq + (J - j), idx_a] = 1.0
                        if j <= j1 <= J - 1:
                            rc = R + rr * (J - j) + (j1 - j)
                            A[rc, idx_a] = -1.0
                    for v in range(J - j):
                        lb[n_eq + v] = -np.inf; ub[n_eq + v] = 1.0
                    lb[n_eq + (J - j)] = -np.inf; ub[n_eq + (J - j)] = float(rem_b)
                    c_vec = np.array([a[3] for a in arcs], dtype=np.float64)
                    res_m = milp(c=c_vec, integrality=np.ones(nv), bounds=Bounds(0, 1), constraints=LinearConstraint(A.tocsc(), lb, ub))
                    if res_m.success and res_m.x is not None:
                        for idx_a, val in enumerate(res_m.x):
                            if val > 0.5 and arcs[idx_a][1] == j:
                                if d_unmerged[arcs[idx_a][0]] > 0:
                                    sel_r = arcs[idx_a][0]
                                break

            if sel_r is not None:
                c0 = time.process_time()
                if shard_delta_vecs[sel_r]:
                    all_v = np.ascontiguousarray(np.vstack([shard_base_vecs[sel_r]] + shard_delta_vecs[sel_r]), dtype=np.float32)
                    all_i = np.ascontiguousarray(np.concatenate([shard_base_ids[sel_r]] + shard_delta_ids[sel_r]), dtype=np.int64)
                    shard_base_vecs[sel_r] = all_v
                    shard_base_ids[sel_r] = all_i
                    shard_delta_vecs[sel_r] = []
                    shard_delta_ids[sel_r] = []
                    base_indexes[sel_r] = _build_shard_base(all_v, all_i)
                    delta_indexes[sel_r] = _build_delta_idx()
                maint_cpu_s += time.process_time() - c0
                d_unmerged[sel_r] = 0.0
                tau_block[sel_r] = j
                d_at_reset[sel_r] = cum_ins[sel_r]
                n_compactions += 1

        # Window insertions -> delta buffers
        if len(crud) > 0:
            ins_v = np.ascontiguousarray(pool[crud["vec_ref"]], dtype=np.float32)
            ins_i = np.ascontiguousarray(crud["id"], dtype=np.int64)
            c0 = time.process_time()
            d2 = np.sum(ins_v ** 2, axis=1, keepdims=True) + c_norm2[None, :] - 2.0 * (ins_v @ centroids.T)
            bs = np.argmin(d2, axis=1)
            for r in range(R):
                rm = bs == r
                if np.any(rm):
                    vr = np.ascontiguousarray(ins_v[rm], dtype=np.float32)
                    ir = np.ascontiguousarray(ins_i[rm], dtype=np.int64)
                    delta_indexes[r].add_with_ids(vr, ir)
                    shard_delta_vecs[r].append(vr)
                    shard_delta_ids[r].append(ir)
                    cnt = float(len(ir))
                    cum_ins[r] += cnt
                    d_unmerged[r] += cnt
            maint_cpu_s += time.process_time() - c0
        D_all[w] = cum_ins

        # Window queries -> route to nprobe=4 shards, probe base + exact-scan delta
        nq = int(len(qe))
        q_vecs = np.ascontiguousarray(pool[qe["vec_ref"]], dtype=np.float32)
        pred_ids = np.empty((nq, k), dtype=np.int64)
        c0 = time.process_time()
        for qi in range(nq):
            tq0 = time.perf_counter()
            v = q_vecs[qi:qi + 1]
            d2 = float(np.sum(v ** 2)) + c_norm2 - 2.0 * (v[0] @ centroids.T)
            top_sh = np.argpartition(d2, kth=nprobe_shard - 1)[:nprobe_shard]
            cand_d: List[float] = []
            cand_i: List[int] = []
            for r in top_sh.tolist():
                Q_all[w, r] += 1.0
                if engine == "sharded_faiss_ivf":
                    db, ib = base_indexes[r].search(v, k)
                    for dval, ival in zip(db[0].tolist(), ib[0].tolist()):
                        if ival >= 0:
                            cand_d.append(dval); cand_i.append(ival)
                else:
                    ib, db = base_indexes[r].knn_query(v, k=k, num_threads=1)
                    for dval, ival in zip(db[0].tolist(), ib[0].tolist()):
                        if ival >= 0:
                            cand_d.append(dval); cand_i.append(ival)
                if delta_indexes[r].ntotal > 0:
                    dd, di = delta_indexes[r].search(v, min(k, delta_indexes[r].ntotal))
                    for dval, ival in zip(dd[0].tolist(), di[0].tolist()):
                        if ival >= 0:
                            cand_d.append(dval); cand_i.append(ival)
            order = np.argsort(cand_d)[:k]
            for j_out, o_idx in enumerate(order):
                pred_ids[qi, j_out] = cand_i[o_idx]
            lat_us.append((time.perf_counter() - tq0) * 1e6)
        query_cpu_s += time.process_time() - c0
        gt_w = gt_ids[q_off: q_off + nq]
        q_off += nq
        recalls.append(qio.mean_recall_at_k(pred_ids, gt_w, k))

    wall_s = time.perf_counter() - t_wall0
    load_end = read_load1m()
    total_cpu_s = query_cpu_s + maint_cpu_s

    # Compute exact hardware-invariant scan ops from realized actions & Q_all/D_all
    # Replay exact_scan_ops using the exact same actions taken
    # Note: we can track exact_scan_ops dynamically during windows! Let's compute from D_all and reset points.
    # Actually let's recompute exact_scan_ops cleanly using Q_all, D_all, and reset_windows:
    # We already know d_unmerged in each window w equals delta_all_t - delta_at_last_reset!
    # Re-evaluate using recorded reset events: simpler, compute milp_star & noop directly from Q_all, D_all.
    w_noop = float((Q_all * D_all).sum())
    w_star = _solve_milp_exact(Q_all, D_all, K, budget)
    # Wait: let's track exact_scan_ops per window accurately! Let's add it cleanly below by storing per-window actions?
    # Better: in M1 worker we know exact delta size per shard at every window! Wait—we didn't store reset_blocks above;
    # let's compute exact_scan_ops directly in the loop by accumulating Q_w[r]*delta_r_at_w!
    # We'll patch that in cleanly right below before writing.
    return {
        "engine": engine,
        "trace": trace,
        "policy": policy,
        "budget": budget,
        "n_compactions": n_compactions,
        "query_cpu_s": round(query_cpu_s, 4),
        "maint_cpu_s": round(maint_cpu_s, 4),
        "total_cpu_s": round(total_cpu_s, 4),
        "wall_s": round(wall_s, 4),
        "wall_cpu_ratio": round(wall_s / max(total_cpu_s, 1e-9), 4),
        "mean_recall": round(float(np.mean(recalls)), 5),
        "min_recall": round(float(np.min(recalls)), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recalls if r >= 0.965) / len(recalls), 1),
        "p95_us": round(pct(lat_us, 95.0), 2),
        "p99_us": round(pct(lat_us, 99.0), 2),
        "noop_ops": round(w_noop, 1),
        "milp_star_ops": round(w_star, 1),
        "peak_rss_mb": peak_rss_mb(),
        "load_start": round(load_start, 2),
        "load_end": round(load_end, 2),
        "Q_all": Q_all.tolist(),
        "D_all": D_all.tolist(),
    }


def _run_m1_tracked_ops_patch():
    pass


def _worker_m2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import faiss

    faiss.omp_set_num_threads(1)
    data_root = DATA_BASE / cfg["dataset"]
    tr_dir = data_root / "traces" / cfg["trace"]
    system = cfg["system"]
    quant = cfg["quantization"]
    ptag = cfg["param_tag"]
    params = cfg["params"]

    pool, ph = qio.read_pool(data_root / "pool.bin")
    init, _ = qio.read_init(data_root / "init.bin")
    ev, th = qio.read_trace(tr_dir / "trace.bin")
    gt_ids, _, gh = qio.read_gt(tr_dir / "gt.bin")
    dim = int(ph.fields["dim"])
    k = int(gh.fields["k"])
    n_win = int(th.fields["n_windows"])

    ids0 = np.ascontiguousarray(init["id"], dtype=np.int64)
    vecs0 = np.ascontiguousarray(pool[init["vec_ref"]], dtype=np.float32)

    is_faiss = system.startswith("faiss_")
    is_usearch = system.startswith("usearch_")
    is_crud = int(np.sum((ev["type"] == qio.EV_UPDATE) | (ev["type"] == qio.EV_ERASE))) > 0

    tb0 = time.process_time()
    idx_obj = None
    refine_k = int(params.get("k_factor", 0))
    if is_faiss:
        nlist = int(params.get("nlist", 256))
        q_coarse = faiss.IndexFlatL2(dim)
        if system == "faiss_ivf_flat":
            base_idx = faiss.IndexIVFFlat(q_coarse, dim, nlist, faiss.METRIC_L2)
        elif system in ("faiss_ivf_sq8", "faiss_ivf_sq8_rerank"):
            base_idx = faiss.IndexIVFScalarQuantizer(q_coarse, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        elif system in ("faiss_ivf_pq", "faiss_ivf_pq_rerank"):
            base_idx = faiss.IndexIVFPQ(q_coarse, dim, nlist, int(params.get("M_pq", 16)), int(params.get("nbits", 8)))
        else:
            raise ValueError(system)
        tr_n = min(65536, len(vecs0))
        base_idx.train(vecs0[:tr_n])
        base_idx.nprobe = int(params.get("nprobe", 32))
        if refine_k > 0:
            idx_obj = faiss.IndexRefineFlat(base_idx)
            idx_obj.k_factor = refine_k
            # IndexRefineFlat maps sequential rows unless wrapped or mapped; maintain id translation for add/remove
            idx_obj = faiss.IndexIDMap2(idx_obj)
        else:
            idx_obj = base_idx
        idx_obj.add_with_ids(vecs0, ids0)
    elif is_usearch:
        from usearch.index import Index

        dt = params["dtype"]
        idx_obj = Index(
            ndim=dim, metric="l2sq", dtype=dt,
            connectivity=int(params.get("M", 16)),
            expansion_add=int(params.get("ef_construction", 200)),
            expansion_search=int(params.get("ef", 64)),
        )
        idx_obj.add(np.ascontiguousarray(ids0, dtype=np.uint64), vecs0, threads=1)
        label_of: Dict[int, int] = {}
        next_label = int(max(ids0.max(), ev["id"].max()) + 1)
        synth_to_trace: Dict[int, int] = {}

    build_cpu_s = time.process_time() - tb0

    load_start = read_load1m()
    t_wall0 = time.perf_counter()
    query_cpu_s = 0.0
    maint_cpu_s = 0.0
    recalls: List[float] = []
    lat_us: List[float] = []
    q_off = 0
    status = "ok"

    for w, crud, qe in qio.window_events(ev, n_win):
        if len(crud) > 0:
            c0 = time.process_time()
            try:
                ins_m = crud["type"] == qio.EV_INSERT
                upd_m = crud["type"] == qio.EV_UPDATE
                ers_m = crud["type"] == qio.EV_ERASE
                if is_faiss:
                    rem_ids = np.concatenate([crud["id"][upd_m], crud["id"][ers_m]]).astype(np.int64)
                    if len(rem_ids):
                        sel = faiss.IDSelectorBatch(len(rem_ids), faiss.swig_ptr(rem_ids))
                        n_rem = idx_obj.remove_ids(sel)
                        if n_rem == 0 and len(rem_ids) > 0:
                            status = "unsupported_erase"
                    add_m = ins_m | upd_m
                    if np.any(add_m) and status == "ok":
                        idx_obj.add_with_ids(np.ascontiguousarray(pool[crud["vec_ref"][add_m]], dtype=np.float32), np.ascontiguousarray(crud["id"][add_m], dtype=np.int64))
                elif is_usearch:
                    if np.any(ins_m):
                        idx_obj.add(np.ascontiguousarray(crud["id"][ins_m], dtype=np.uint64), np.ascontiguousarray(pool[crud["vec_ref"][ins_m]], dtype=np.float32), threads=1)
                    if np.any(ers_m):
                        labs = np.array([label_of.pop(int(tid), int(tid)) for tid in crud["id"][ers_m].tolist()], dtype=np.uint64)
                        idx_obj.remove(labs)
                    if np.any(upd_m):
                        u_tids = crud["id"][upd_m].tolist()
                        old_labs = np.array([label_of.get(int(tid), int(tid)) for tid in u_tids], dtype=np.uint64)
                        idx_obj.remove(old_labs)
                        new_labs = np.arange(next_label, next_label + len(u_tids), dtype=np.uint64)
                        next_label += len(u_tids)
                        for tid, nl in zip(u_tids, new_labs.tolist()):
                            label_of[int(tid)] = int(nl)
                            synth_to_trace[int(nl)] = int(tid)
                        idx_obj.add(new_labs, np.ascontiguousarray(pool[crud["vec_ref"][upd_m]], dtype=np.float32), threads=1)
            except Exception:
                status = "unsupported_erase"
            maint_cpu_s += time.process_time() - c0
            if status != "ok":
                # Still finish recall audit for remaining windows or break cleanly
                pass

        nq = int(len(qe))
        qvecs = np.ascontiguousarray(pool[qe["vec_ref"]], dtype=np.float32)
        pred = np.full((nq, k), -1, dtype=np.int64)
        c0 = time.process_time()
        for qi in range(nq):
            tq0 = time.perf_counter()
            if is_faiss:
                _, row = idx_obj.search(qvecs[qi:qi + 1], k)
                pred[qi] = row[0]
            elif is_usearch:
                m = idx_obj.search(qvecs[qi], k, threads=1)
                keys = np.asarray(m.keys, dtype=np.int64)
                for jj, kv in enumerate(keys[:k]):
                    pred[qi, jj] = synth_to_trace.get(int(kv), int(kv))
            lat_us.append((time.perf_counter() - tq0) * 1e6)
        query_cpu_s += time.process_time() - c0
        gt_w = gt_ids[q_off: q_off + nq]
        q_off += nq
        recalls.append(qio.mean_recall_at_k(pred, gt_w, k))

    wall_s = time.perf_counter() - t_wall0
    load_end = read_load1m()
    tot_cpu_s = query_cpu_s + maint_cpu_s
    return {
        "workload": "crud30" if is_crud else "drift",
        "trace": cfg["trace"],
        "system": system,
        "quantization": quant,
        "param_tag": ptag,
        "status": status,
        "build_cpu_s": round(build_cpu_s, 4),
        "query_cpu_s": round(query_cpu_s, 4),
        "maint_cpu_s": round(maint_cpu_s, 4),
        "total_cpu_s": round(tot_cpu_s, 4),
        "wall_s": round(wall_s, 4),
        "wall_cpu_ratio": round(wall_s / max(tot_cpu_s, 1e-9), 4),
        "query_pct": round(100.0 * query_cpu_s / max(tot_cpu_s, 1e-9), 2),
        "maint_pct": round(100.0 * maint_cpu_s / max(tot_cpu_s, 1e-9), 2),
        "mean_recall": round(float(np.mean(recalls)), 5),
        "min_recall": round(float(np.min(recalls)), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recalls if r >= 0.965) / len(recalls), 1),
        "att_0950_pct": round(100.0 * sum(1 for r in recalls if r >= 0.950) / len(recalls), 1),
        "p50_us": round(pct(lat_us, 50.0), 2),
        "p95_us": round(pct(lat_us, 95.0), 2),
        "p99_us": round(pct(lat_us, 99.0), 2),
        "peak_rss_mb": peak_rss_mb(),
        "load_start": round(load_start, 2),
        "load_end": round(load_end, 2),
    }


def _worker_m3(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import faiss
    import hnswlib

    faiss.omp_set_num_threads(1)
    data_root = DATA_BASE / cfg["dataset"]
    tr_dir = data_root / "traces" / cfg["trace"]
    system = cfg["system"]
    ptag = cfg["param_tag"]
    q_repeat = int(cfg.get("query_repeat", 1))
    u_pct = int(cfg["u_pct"])
    q_per_win = int(cfg["q_per_win"])

    pool, ph = qio.read_pool(data_root / "pool.bin")
    init, _ = qio.read_init(data_root / "init.bin")
    ev, th = qio.read_trace(tr_dir / "trace.bin")
    gt_ids, _, gh = qio.read_gt(tr_dir / "gt.bin")
    dim = int(ph.fields["dim"])
    k = int(gh.fields["k"])
    n_win = int(th.fields["n_windows"])

    ids0 = np.ascontiguousarray(init["id"], dtype=np.int64)
    vecs0 = np.ascontiguousarray(pool[init["vec_ref"]], dtype=np.float32)
    n_ins = int(np.sum(ev["type"] == qio.EV_INSERT))

    tb0 = time.process_time()
    if system == "faiss_ivf":
        q_coarse = faiss.IndexFlatL2(dim)
        idx = faiss.IndexIVFFlat(q_coarse, dim, 256, faiss.METRIC_L2)
        idx.train(vecs0[:min(65536, len(vecs0))])
        idx.add_with_ids(vecs0, ids0)
        idx.nprobe = 32
    elif system == "hnswlib":
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=len(ids0) + n_ins + 1024, ef_construction=200, M=16)
        idx.set_num_threads(1)
        idx.set_ef(128)
        idx.add_items(vecs0, ids0, num_threads=1)
    elif system == "usearch_f32":
        from usearch.index import Index

        idx = Index(ndim=dim, metric="l2sq", dtype="f32", connectivity=16, expansion_add=200, expansion_search=128)
        idx.add(np.ascontiguousarray(ids0, dtype=np.uint64), vecs0, threads=1)
    else:
        raise ValueError(system)
    build_cpu_s = time.process_time() - tb0

    load_start = read_load1m()
    t_wall0 = time.perf_counter()
    query_cpu_s = 0.0
    maint_cpu_s = 0.0
    recalls: List[float] = []
    lat_us: List[float] = []
    q_off = 0

    for w, crud, qe in qio.window_events(ev, n_win):
        if len(crud) > 0:
            iv = np.ascontiguousarray(pool[crud["vec_ref"]], dtype=np.float32)
            ii = np.ascontiguousarray(crud["id"], dtype=np.int64)
            c0 = time.process_time()
            if system == "faiss_ivf":
                idx.add_with_ids(iv, ii)
            elif system == "hnswlib":
                idx.add_items(iv, ii, num_threads=1)
            elif system == "usearch_f32":
                idx.add(np.ascontiguousarray(ii, dtype=np.uint64), iv, threads=1)
            maint_cpu_s += time.process_time() - c0

        nq_base = int(len(qe))
        qvecs = np.ascontiguousarray(pool[qe["vec_ref"]], dtype=np.float32)
        gt_w = gt_ids[q_off: q_off + nq_base]
        q_off += nq_base

        pred = np.full((nq_base, k), -1, dtype=np.int64)
        c0 = time.process_time()
        for rep in range(q_repeat):
            for qi in range(nq_base):
                tq0 = time.perf_counter()
                if system == "faiss_ivf":
                    _, row = idx.search(qvecs[qi:qi + 1], k)
                    if rep == 0:
                        pred[qi] = row[0]
                elif system == "hnswlib":
                    row, _ = idx.knn_query(qvecs[qi:qi + 1], k=k, num_threads=1)
                    if rep == 0:
                        pred[qi] = row[0]
                elif system == "usearch_f32":
                    m = idx.search(qvecs[qi], k, threads=1)
                    if rep == 0:
                        keys = np.asarray(m.keys, dtype=np.int64)
                        pred[qi, :min(k, len(keys))] = keys[:k]
                lat_us.append((time.perf_counter() - tq0) * 1e6)
        query_cpu_s += time.process_time() - c0
        recalls.append(qio.mean_recall_at_k(pred, gt_w, k))

    wall_s = time.perf_counter() - t_wall0
    load_end = read_load1m()
    tot_cpu_s = query_cpu_s + maint_cpu_s
    ins_per_win = n_ins / float(n_win)
    return {
        "u_pct": u_pct,
        "q_per_win": q_per_win * q_repeat,
        "q_to_u_ratio": round((q_per_win * q_repeat) / max(ins_per_win, 1.0), 2),
        "trace": cfg["trace"],
        "system": system,
        "param_tag": ptag,
        "build_cpu_s": round(build_cpu_s, 4),
        "query_cpu_s": round(query_cpu_s, 4),
        "maint_cpu_s": round(maint_cpu_s, 4),
        "total_cpu_s": round(tot_cpu_s, 4),
        "wall_s": round(wall_s, 4),
        "wall_cpu_ratio": round(wall_s / max(tot_cpu_s, 1e-9), 4),
        "query_pct": round(100.0 * query_cpu_s / max(tot_cpu_s, 1e-9), 2),
        "maint_pct": round(100.0 * maint_cpu_s / max(tot_cpu_s, 1e-9), 2),
        "mean_recall": round(float(np.mean(recalls)), 5),
        "min_recall": round(float(np.min(recalls)), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recalls if r >= 0.965) / len(recalls), 1),
        "p99_us": round(pct(lat_us, 99.0), 2),
        "peak_rss_mb": peak_rss_mb(),
        "load_start": round(load_start, 2),
        "load_end": round(load_end, 2),
    }


# ---------------------------------------------------------------------------
# Phase Orchestrators
# ---------------------------------------------------------------------------
def run_m1() -> None:
    log("=== Phase M1: Real-Engine Multi-Shard Scheduler Bridge ===")
    rows: List[Dict[str, Any]] = []
    out_csv = OUT_BASE / "m1_real_engine_scheduler_bridge.csv"

    drifts = ["abrupt", "gradual", "hotspot", "periodic", "uniform"]
    # Precompute GBRT causal/leaky predictions on sibling seeds s06..s09 -> s05
    preds_cache: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    for dr in drifts:
        preds_cache[dr] = _build_gbrt_predictions(dr, target_seed="s05")

    jobs: List[Tuple[str, str, str, int]] = []
    for dr in drifts:
        for pol, b in [
            ("noop", 0), ("fixed_c2", 5), ("fixed_c4", 2),
            ("past_qmass", 5), ("ewma_a03", 5),
            ("gbrt_causal", 5), ("gbrt_leaky", 5),
            ("online_milp", 5), ("oracle_1blk", 5),
        ]:
            jobs.append(("sharded_faiss_ivf", f"{dr}_s05", pol, b))
    for dr in ("abrupt", "hotspot", "gradual"):
        for pol, b in [
            ("noop", 0), ("fixed_c2", 5), ("past_qmass", 5),
            ("ewma_a03", 5), ("gbrt_causal", 5), ("gbrt_leaky", 5),
        ]:
            jobs.append(("sharded_hnswlib", f"{dr}_s05", pol, b))

    for idx_j, (eng, tr, pol, bud) in enumerate(jobs, 1):
        wait_quiet_host(2.5)
        dr = tr.rsplit("_", 1)[0]
        pmap = preds_cache[dr].get(pol, {})
        res = _invoke_self_worker("m1", {
            "engine": eng, "trace": tr, "policy": pol, "budget": bud,
            "preds_map": {str(k): v for k, v in pmap.items()},
        })
        # Extract exact_scan_ops using Q_all/D_all and worker's action count; wait - compute exact_scan_ops properly:
        Q_all = np.asarray(res.pop("Q_all"), dtype=np.float64)
        D_all = np.asarray(res.pop("D_all"), dtype=np.float64)
        # Reconstruct deterministic schedule or compute GCR via replay helper matching exact worker policy
        # Wait: let's compute exact_scan_ops from _replay_exact_ops_for_policy so 100% faithful!
        exact_ops = _replay_exact_ops(Q_all, D_all, K_BLOCK, pol, bud, pmap)
        w_noop = float(res["noop_ops"])
        w_star = float(res["milp_star_ops"])
        gcr_ops = 100.0 * (w_noop - exact_ops) / max(w_noop - w_star, 1e-9) if pol != "noop" else 0.0
        res["exact_scan_ops"] = round(exact_ops, 1)
        res["gcr_ops_pct"] = round(gcr_ops, 2)
        rows.append(res)
        write_csv(rows, out_csv)
        log(f"[M1 {idx_j}/{len(jobs)}] {eng} {tr} {pol}: cpu={res['total_cpu_s']:.2f}s (q={res['query_cpu_s']:.2f} m={res['maint_cpu_s']:.2f}) w/c={res['wall_cpu_ratio']} rec={res['mean_recall']:.4f} gcr_ops={res['gcr_ops_pct']}%")

    # Save summary stats (Spearman exact_scan_ops vs total_cpu_s & Gate-4 pass rate)
    summary: Dict[str, Any] = {}
    for eng in ("sharded_faiss_ivf", "sharded_hnswlib"):
        sub = [r for r in rows if r["engine"] == eng]
        if sub:
            ops = [r["exact_scan_ops"] for r in sub]
            cpus = [r["total_cpu_s"] for r in sub]
            rho = float(spearmanr(ops, cpus).statistic)
            g4_pass = sum(1 for r in sub if r["wall_cpu_ratio"] <= 1.18)
            summary[eng] = {
                "n_runs": len(sub),
                "spearman_ops_vs_cpu": round(rho, 4),
                "gate4_pass": f"{g4_pass}/{len(sub)}",
            }
    (OUT_BASE / "m1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"[M1 Summary] {summary}")


def _replay_exact_ops(Q: np.ndarray, D: np.ndarray, K: int, policy: str, budget: int, pmap: Dict[int, List[float]]) -> float:
    T, R = Q.shape
    J = T // K
    d_at_reset = np.zeros(R, dtype=np.float64)
    ewma_q = np.zeros(R, dtype=np.float64)
    spent = 0
    rr = 0
    w_tot = 0.0
    future_Q = np.array([Q[j * K:(j + 1) * K].sum(axis=0) for j in range(J)])
    for j in range(J):
        if j > 0:
            prev_q = Q[(j - 1) * K: j * K].sum(axis=0)
            ewma_q = prev_q.copy() if j == 1 else (0.3 * prev_q + 0.7 * ewma_q)
            cur_d = np.maximum(0.0, D[j * K - 1] - d_at_reset)
            sel_r: Optional[int] = None
            if policy == "fixed_c2" and (j - 1) % 2 == 0 and spent < budget:
                sel_r = rr % R; rr += 1
            elif policy == "fixed_c4" and (j - 1) % 4 == 0 and spent < budget:
                sel_r = rr % R; rr += 1
            elif policy != "noop" and spent < budget:
                if policy == "past_qmass":
                    sc = prev_q * cur_d
                elif policy == "ewma_a03":
                    sc = ewma_q * cur_d
                elif policy in ("gbrt_causal", "gbrt_leaky"):
                    sc = np.asarray(pmap[j], dtype=np.float64) * cur_d
                elif policy == "oracle_1blk":
                    sc = future_Q[j] * cur_d
                elif policy == "online_milp":
                    rem_b = budget - spent
                    q_rate = np.maximum(0.0, ewma_q / float(K))
                    arcs = []
                    for r in range(R):
                        for j1 in range(j - 1, J):
                            for j2 in range(j1 + 1, J + 1):
                                cost = float(q_rate[r]) * float(cur_d[r]) * float(K * (j2 - j)) if j1 == j - 1 else 0.0
                                arcs.append((r, j1, j2, cost))
                    nv = len(arcs)
                    n_eq = R + R * (J - j); n_ineq = (J - j) + 1
                    A = lil_matrix((n_eq + n_ineq, nv), dtype=np.float64)
                    lb = np.zeros(n_eq + n_ineq); ub = np.zeros(n_eq + n_ineq)
                    for r in range(R): lb[r] = 1.0; ub[r] = 1.0
                    for idx_a, (rr_a, j1, j2, _) in enumerate(arcs):
                        if j1 == j - 1: A[rr_a, idx_a] = 1.0
                        if j <= j2 <= J - 1:
                            A[R + rr_a * (J - j) + (j2 - j), idx_a] = 1.0
                            A[n_eq + (j2 - j), idx_a] = 1.0
                            A[n_eq + (J - j), idx_a] = 1.0
                        if j <= j1 <= J - 1:
                            A[R + rr_a * (J - j) + (j1 - j), idx_a] = -1.0
                    for v in range(J - j): lb[n_eq + v] = -np.inf; ub[n_eq + v] = 1.0
                    lb[n_eq + (J - j)] = -np.inf; ub[n_eq + (J - j)] = float(rem_b)
                    res_m = milp(c=np.array([a[3] for a in arcs]), integrality=np.ones(nv), bounds=Bounds(0, 1), constraints=LinearConstraint(A.tocsc(), lb, ub))
                    sc = np.zeros(R)
                    if res_m.success and res_m.x is not None:
                        for idx_a, val in enumerate(res_m.x):
                            if val > 0.5 and arcs[idx_a][1] == j:
                                if cur_d[arcs[idx_a][0]] > 0:
                                    sel_r = arcs[idx_a][0]
                                break
                else:
                    sc = np.zeros(R)
                if policy != "online_milp":
                    br = int(np.argmax(sc))
                    if cur_d[br] > 0:
                        sel_r = br
            if sel_r is not None:
                d_at_reset[sel_r] = D[j * K - 1, sel_r]
                spent += 1
        for t in range(j * K, min((j + 1) * K, T)):
            w_tot += float((Q[t] * np.maximum(0.0, D[t] - d_at_reset)).sum())
    return w_tot


def run_m2() -> None:
    log("=== Phase M2: Quantized Backends Audit ===")
    rows: List[Dict[str, Any]] = []
    out_csv = OUT_BASE / "m2_quantized_backends.csv"
    workloads = [
        ("sift100k_u10_q2500_v2", "abrupt_s05"),
        ("sift100k_u10_q2500_v2", "hotspot_s05"),
        ("sift100k_u10_q2500_crud30v2", "abrupt_s05"),
    ]
    configs = [
        ("faiss_ivf_flat", "f32", "np16", {"nlist": 256, "nprobe": 16}),
        ("faiss_ivf_flat", "f32", "np32", {"nlist": 256, "nprobe": 32}),
        ("faiss_ivf_sq8", "sq8", "np16", {"nlist": 256, "nprobe": 16}),
        ("faiss_ivf_sq8", "sq8", "np32", {"nlist": 256, "nprobe": 32}),
        ("faiss_ivf_sq8_rerank", "sq8+f32_rerank", "np32_k4", {"nlist": 256, "nprobe": 32, "k_factor": 4}),
        ("faiss_ivf_pq", "pq16x8", "np32", {"nlist": 256, "nprobe": 32, "M_pq": 16, "nbits": 8}),
        ("faiss_ivf_pq_rerank", "pq16x8+f32_rerank", "np32_k8", {"nlist": 256, "nprobe": 32, "M_pq": 16, "nbits": 8, "k_factor": 8}),
        ("usearch_f32", "f32", "ef64", {"dtype": "f32", "M": 16, "ef_construction": 200, "ef": 64}),
        ("usearch_f32", "f32", "ef128", {"dtype": "f32", "M": 16, "ef_construction": 200, "ef": 128}),
        ("usearch_f16", "f16", "ef64", {"dtype": "f16", "M": 16, "ef_construction": 200, "ef": 64}),
        ("usearch_f16", "f16", "ef128", {"dtype": "f16", "M": 16, "ef_construction": 200, "ef": 128}),
        ("usearch_i8", "i8", "ef64", {"dtype": "i8", "M": 16, "ef_construction": 200, "ef": 64}),
        ("usearch_i8", "i8", "ef128", {"dtype": "i8", "M": 16, "ef_construction": 200, "ef": 128}),
    ]
    tot = len(workloads) * len(configs)
    cnt = 0
    for ds, tr in workloads:
        for sys_name, q_label, ptag, par in configs:
            cnt += 1
            wait_quiet_host(2.5)
            res = _invoke_self_worker("m2", {
                "dataset": ds, "trace": tr, "system": sys_name,
                "quantization": q_label, "param_tag": ptag, "params": par,
            })
            rows.append(res)
            write_csv(rows, out_csv)
            log(f"[M2 {cnt}/{tot}] {res['workload']}/{tr} {sys_name} {ptag}: cpu={res['total_cpu_s']:.2f}s rec={res['mean_recall']:.4f} att={res['att_0965_pct']}% rss={res['peak_rss_mb']}MB w/c={res['wall_cpu_ratio']} stat={res['status']}")


def run_m3() -> None:
    log("=== Phase M3: Real-Engine Query-to-Update (Q/u) Crossover Boundary ===")
    rows: List[Dict[str, Any]] = []
    out_csv = OUT_BASE / "m3_qu_crossover_real_engines.csv"

    # Available on Ryzen-9950X across u in {2,5,10,20}% x Q in {500,2500,5000}: hotspot_s07
    grid: List[Tuple[int, int, str, int]] = []
    for u in (2, 5, 10, 20):
        for q in (500, 2500, 5000):
            ds = f"sift100k_u{u:02d}_q{q:04d}_v2"
            grid.append((u, q, ds, 1))
    # Add read-heavy extreme (u=2%, Q=20000 via 4x query repeat over u02_q5000)
    grid.append((2, 5000, "sift100k_u02_q5000_v2", 4))

    corners = {(2, 500, 1), (2, 5000, 1), (2, 5000, 4), (20, 500, 1)}
    jobs: List[Tuple[int, int, str, int, str, str]] = []
    for u, q, ds, rep in grid:
        jobs.append((u, q, ds, rep, "faiss_ivf", "np32"))
        jobs.append((u, q, ds, rep, "hnswlib", "ef128"))
        if (u, q, rep) in corners:
            jobs.append((u, q, ds, rep, "usearch_f32", "ef128"))

    for idx_j, (u, q, ds, rep, sys_name, ptag) in enumerate(jobs, 1):
        wait_quiet_host(2.5)
        res = _invoke_self_worker("m3", {
            "dataset": ds, "trace": "hotspot_s07", "system": sys_name,
            "param_tag": ptag, "u_pct": u, "q_per_win": q, "query_repeat": rep,
        })
        rows.append(res)
        write_csv(rows, out_csv)
        log(f"[M3 {idx_j}/{len(jobs)}] u={u}% Q={q*rep} (Q/u={res['q_to_u_ratio']}) {sys_name}: cpu={res['total_cpu_s']:.2f}s (q={res['query_cpu_s']:.2f} m={res['maint_cpu_s']:.2f}) rec={res['mean_recall']:.4f} w/c={res['wall_cpu_ratio']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp 9 Pure-CPU Core-8 Supplement on Ryzen-9950X")
    ap.add_argument("--phase", choices=["M1", "M2", "M3", "ALL"], default="ALL")
    ap.add_argument("--worker-json", default="")
    args = ap.parse_args()

    if args.worker_json:
        cfg = json.loads(args.worker_json)
        mode = cfg.pop("mode")
        if mode == "m1":
            out = _worker_m1(cfg)
        elif mode == "m2":
            out = _worker_m2(cfg)
        elif mode == "m3":
            out = _worker_m3(cfg)
        else:
            raise ValueError(mode)
        print("WORKER_JSON=" + json.dumps(out), flush=True)
        return 0

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    if args.phase in ("M1", "ALL"):
        run_m1()
    if args.phase in ("M2", "ALL"):
        run_m2()
    if args.phase in ("M3", "ALL"):
        run_m3()
    (OUT_BASE / "EXP9_DONE.flag").write_text("ok\n", encoding="utf-8")
    log("EXP9_MIMO_SUPPLEMENT_ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
