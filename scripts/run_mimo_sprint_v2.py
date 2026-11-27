#!/usr/bin/env python3
"""PVLDB Sprint v2 Offload Runner for Ryzen 9 9950X (16 physical cores).

Executes the 4 self-contained suites (`p0_2`, `p1_1b`, `p1_1c`, `p1_2`) on spaced physical
cores (`0..15`, zero HT siblings `16..31`) via `qbam_replay`
pinned with `taskset -c <core>` and single-thread BLAS env vars.

Zero cross-host batch pooling: each suite is self-contained on a single host and records
`wall_ms` and `wall_cpu_ratio` for every run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.ensemble import HistGradientBoostingRegressor

REPLAY_BIN = Path("/workspace/qbam/build_rev4/qbam_replay")
QBAM_DATA = Path("/workspace/qbam/data")
WORKLOAD_DIR = Path("/workspace/qbam/python/workloads")
WORKLOAD_YAML = Path("/workspace/qbam/configs/workload_shift.yaml")
SPRINT_DATA = Path("/workspace/pvldb/data_sprint_v2")
SPRINT_RES = Path("/workspace/pvldb/results/sprint_v2")

K_BLOCK = 4
BUILD_W = 12
POOL_6 = [0, 2, 4, 6, 8, 10]
POOL_8 = [0, 2, 4, 6, 8, 10, 12, 14]
SKLEARN_LOCK = threading.Lock()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Exact Theorem 1 Segment-Flow MILP Solver (arbitrary R, J, B)
# -----------------------------------------------------------------------------
def solve_segment_flow_milp(
    q_mat: np.ndarray,
    d_mat: np.ndarray,
    B: int,
    K: int = K_BLOCK,
    init_delta: np.ndarray | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Solve exact segment-flow MILP via HiGHS; returns (opt_ops, [(j, shard), ...])."""
    T, R = q_mat.shape
    J = T // K
    if B <= 0 or J <= 1:
        if init_delta is None:
            return float((q_mat * d_mat).sum()), []
        return float((q_mat * (d_mat + init_delta[None, :])).sum()), []

    B_eff = min(int(B), J - 1)
    arcs: list[tuple[int, int, int]] = []
    costs: list[float] = []
    for r in range(R):
        d_base_init = float(init_delta[r]) if init_delta is not None else 0.0
        for j1 in range(J):
            t_start = K * j1
            d_reset = float(d_mat[t_start - 1, r]) if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                t_end = min(K * j2, T)
                if j1 == 0:
                    eff_d = d_base_init + d_mat[t_start:t_end, r]
                else:
                    eff_d = np.maximum(0.0, d_mat[t_start:t_end, r] - d_reset)
                c = float((q_mat[t_start:t_end, r] * eff_d).sum())
                arcs.append((r, j1, j2))
                costs.append(c)

    n_vars = len(arcs)
    c_vec = np.array(costs, dtype=np.float64)

    n_eq = R + R * (J - 1)
    n_ineq = (J - 1) + 1
    A = lil_matrix((n_eq + n_ineq, n_vars), dtype=np.float64)
    lb = np.zeros(n_eq + n_ineq, dtype=np.float64)
    ub = np.zeros(n_eq + n_ineq, dtype=np.float64)

    for r in range(R):
        lb[r] = 1.0
        ub[r] = 1.0
    for idx, (r, j1, j2) in enumerate(arcs):
        if j1 == 0:
            A[r, idx] = 1.0
        if 1 <= j2 <= J - 1:
            row_c = R + r * (J - 1) + (j2 - 1)
            A[row_c, idx] = 1.0
            A[n_eq + (j2 - 1), idx] = 1.0
            A[n_eq + (J - 1), idx] = 1.0
        if 1 <= j1 <= J - 1:
            row_c = R + r * (J - 1) + (j1 - 1)
            A[row_c, idx] = -1.0

    for v in range(J - 1):
        lb[n_eq + v] = -np.inf
        ub[n_eq + v] = 1.0
    lb[n_eq + (J - 1)] = -np.inf
    ub[n_eq + (J - 1)] = float(B_eff)

    res = milp(
        c=c_vec,
        integrality=np.ones(n_vars, dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(A.tocsc(), lb, ub),
    )
    if not res.success:
        raise RuntimeError(f"HiGHS MILP failed: {res.message}")

    sched: list[tuple[int, int]] = []
    for idx, val in enumerate(res.x):
        if val > 0.5:
            r, j1, j2 = arcs[idx]
            if 1 <= j1 <= J - 1:
                sched.append((j1, r))
    sched.sort()
    return float(res.fun), sched


def solve_myopic_1blk(q_mat: np.ndarray, d_mat: np.ndarray, B: int, K: int = K_BLOCK) -> float:
    """1-block myopic clairvoyant oracle under action budget B."""
    T, R = q_mat.shape
    J = T // K
    rem_B = int(B)
    last_reset = np.full(R, -1, dtype=np.int64)
    total_ops = 0.0
    for j in range(J):
        t0, t1 = K * j, min(K * (j + 1), T)
        if j >= 1 and rem_B > 0:
            best_gain, best_r = 0.0, -1
            for r in range(R):
                d_prev = float(d_mat[last_reset[r], r]) if last_reset[r] >= 0 else 0.0
                d_now_base = float(d_mat[t0 - 1, r])
                cur_delta = np.maximum(0.0, d_mat[t0:t1, r] - d_prev)
                reset_delta = np.maximum(0.0, d_mat[t0:t1, r] - d_now_base)
                gain = float((q_mat[t0:t1, r] * (cur_delta - reset_delta)).sum())
                if gain > best_gain + 1e-12:
                    best_gain, best_r = gain, r
            if best_r >= 0:
                last_reset[best_r] = t0 - 1
                rem_B -= 1
        for r in range(R):
            d_prev = float(d_mat[last_reset[r], r]) if last_reset[r] >= 0 else 0.0
            total_ops += float((q_mat[t0:t1, r] * np.maximum(0.0, d_mat[t0:t1, r] - d_prev)).sum())
    return total_ops


def seasonal_acf_pred(q: np.ndarray, w0: int, ew03: np.ndarray, lags=(8, 12, 16, 20, 24), thresh=0.35) -> np.ndarray:
    """Strictly causal single-stream lag-autocorrelation predictor."""
    R = q.shape[1]
    if w0 < min(lags) + K_BLOCK:
        return ew03.copy()
    best_lag, best_r = -1, -1.0
    hist = q[:w0]
    for lag in lags:
        if w0 < lag + K_BLOCK + 2:
            continue
        x = hist[:-lag]
        y = hist[lag:]
        corrs = []
        for r in range(R):
            xr = x[:, r] - x[:, r].mean()
            yr = y[:, r] - y[:, r].mean()
            denom = float(np.sqrt((xr ** 2).sum() * (yr ** 2).sum()))
            if denom > 1e-9:
                corrs.append(float((xr * yr).sum() / denom))
        mean_r = float(np.mean(corrs)) if corrs else 0.0
        if mean_r > best_r:
            best_r = mean_r
            best_lag = lag
    if best_lag > 0 and best_r >= thresh:
        return hist[w0 - best_lag : w0 - best_lag + K_BLOCK].mean(axis=0)
    return ew03.copy()


def build_single_stream_online_preds(
    q: np.ndarray,
    d: np.ndarray,
    use_seasonal_lags: bool = False,
) -> dict[str, np.ndarray]:
    """Strictly single-stream online walk-forward predictions (zero sibling-seed training)."""
    T, R = q.shape
    J = T // K_BLOCK
    ew_arr = np.zeros((J, R), dtype=np.float64)
    sacf_arr = np.zeros((J, R), dtype=np.float64)
    gbrt_arr = np.zeros((J, R), dtype=np.float64)
    oracle_arr = np.zeros((J, R), dtype=np.float64)

    X_by_j: dict[int, np.ndarray] = {}
    y_by_j: dict[int, np.ndarray] = {}

    for j in range(J):
        w0, w1 = j * K_BLOCK, min((j + 1) * K_BLOCK, T)
        yb = q[w0:w1].mean(axis=0)
        oracle_arr[j] = yb
        y_by_j[j] = yb
        if j == 0:
            continue
        h4 = q[max(0, w0 - 4) : w0]
        h8 = q[max(0, w0 - 8) : w0]
        qp4 = h4.mean(0)
        qp8 = h8.mean(0)
        ql1 = q[w0 - 1]
        qsd = h8.std(0)
        ew = q[0].astype(np.float64).copy()
        for wt in range(1, w0):
            ew = 0.3 * q[wt] + 0.7 * ew
        ew_arr[j] = ew
        sacf = seasonal_acf_pred(q, w0, ew)
        sacf_arr[j] = sacf
        dc = d[w0 - 1]
        dp = d[w0 - 5] if w0 >= 5 else np.zeros(R)
        di = dc - dp
        tp4 = float(qp4.sum())

        lag12 = q[w0 - 12 : w0 - 8].mean(0) if w0 >= 12 else ew
        lag16 = q[w0 - 16 : w0 - 12].mean(0) if w0 >= 16 else ew
        lag20 = q[w0 - 20 : w0 - 16].mean(0) if w0 >= 20 else ew

        feats = []
        for r in range(R):
            base = [
                float(ql1[r]), float(qp4[r]), float(qp8[r]), float(qsd[r]),
                float(ew[r]), float(sacf[r]), float(qp4[r] / max(tp4, 1e-9)), float(di[r]),
            ]
            if use_seasonal_lags:
                base += [float(lag12[r]), float(lag16[r]), float(lag20[r])]
            feats.append(base)
        X_by_j[j] = np.array(feats, dtype=np.float64)

        if j < 3:
            gbrt_arr[j] = ew
        else:
            X_tr = np.vstack([X_by_j[jj] for jj in range(1, j)])
            y_tr = np.concatenate([y_by_j[jj] for jj in range(1, j)])
            with SKLEARN_LOCK:
                model = HistGradientBoostingRegressor(
                    max_iter=80, max_depth=4, min_samples_leaf=5, learning_rate=0.08, random_state=42
                )
                model.fit(X_tr, y_tr)
                gbrt_arr[j] = np.maximum(0.0, model.predict(X_by_j[j]))

    return {
        "ewma_a03": ew_arr,
        "seasonal_acf": sacf_arr,
        "gbrt_online": gbrt_arr,
        "oracle_clairvoyant": oracle_arr,
    }


# -----------------------------------------------------------------------------
# C++ Engine Runner & Exact DP Table Writer
# -----------------------------------------------------------------------------
def _read_wall_ms_from_log(log_p: Path) -> float | None:
    if not log_p.exists():
        return None
    for line in log_p.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]:
        if line.startswith("WALL_MS="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def parse_jsonl(path: Path, wall_ms: float | None = None) -> dict:
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
    qc = sum(float(w.get("query_cpu_ms", 0.0)) for w in windows)
    uc = sum(float(w.get("update_cpu_ms", 0.0)) for w in windows)
    mc = sum(float(b.get("maintenance_cpu_ms", 0.0)) for b in blocks)
    total_cpu = qc + uc + mc
    rec = [float(w["recall_at_10"]) for w in windows if "recall_at_10" in w]
    p95 = [float(b["p95_wall_us"]) for b in blocks if "p95_wall_us" in b]
    p99 = [float(b["p99_wall_us"]) for b in blocks if "p99_wall_us" in b]
    q_mat = np.array([w["shard_q_accesses"] for w in windows], dtype=np.float64)
    d_mat = np.array([w["shard_delta_n"] for w in windows], dtype=np.float64)
    n_act = sum(1 for b in blocks if str(b.get("action", "noop")) != "noop")
    if wall_ms is None:
        wall_ms = _read_wall_ms_from_log(path.with_suffix(".log"))
    wall_ratio = (float(wall_ms) / max(total_cpu, 1.0)) if wall_ms is not None else float("nan")
    return {
        "exact_scan_ops": float((q_mat * d_mat).sum()),
        "query_cpu_ms": qc,
        "update_cpu_ms": uc,
        "maint_cpu_ms": mc,
        "total_cpu_ms": total_cpu,
        "wall_ms": float(wall_ms) if wall_ms is not None else float("nan"),
        "wall_cpu_ratio": round(wall_ratio, 4) if wall_ms is not None else float("nan"),
        "mean_recall": float(np.mean(rec)) if rec else float("nan"),
        "min_recall": float(np.min(rec)) if rec else float("nan"),
        "mean_p95_us": float(np.mean(p95)) if p95 else float("nan"),
        "mean_p99_us": float(np.mean(p99)) if p99 else float("nan"),
        "n_actions": n_act,
        "q_accesses": q_mat,
        "delta_n": d_mat,
    }


def run_replay(cfg: dict, out_jsonl: Path, force: bool = False) -> dict:
    cfg_p = out_jsonl.with_suffix(".config.json")
    log_p = out_jsonl.with_suffix(".log")
    wall_ms: float | None = None
    if force or not (out_jsonl.exists() and out_jsonl.stat().st_size > 200):
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_jsonl.with_suffix(".jsonl.tmp")
        if tmp.exists():
            tmp.unlink()
        cfg_p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        pin_cpu = int(cfg["pin_cpu"])
        env = dict(os.environ)
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        cmd = ["taskset", "-c", str(pin_cpu), str(REPLAY_BIN), "--config", str(cfg_p), "--out", str(tmp)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        log_p.write_text(f"WALL_MS={wall_ms:.2f}\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"qbam_replay failed ({cfg['run_id']}): {proc.stderr[-300:]}")
        tmp.replace(out_jsonl)
    return parse_jsonl(out_jsonl, wall_ms=wall_ms)


def make_cfg(
    ds_dir: Path,
    trace_name: str,
    policy: str,
    pp: dict,
    run_id: str,
    pin_cpu: int,
    *,
    R: int = 16,
    dim: int = 128,
    nprobe: int = 4,
    ef_search: int = 32,
    overfetch_start: int = 40,
    use_gt: bool = True,
) -> dict:
    gt_path = ds_dir / "traces" / trace_name / "gt.bin"
    return {
        "pool": str(ds_dir / "pool.bin"),
        "init": str(ds_dir / "init.bin"),
        "centroids": str(ds_dir / "centroids.f32"),
        "trace": str(ds_dir / "traces" / trace_name / "trace.bin"),
        "gt": str(gt_path) if (use_gt and gt_path.exists()) else None,
        "R": R,
        "dim": dim,
        "k": 10,
        "M": 16,
        "efConstruction": 100,
        "efSearch": ef_search,
        "nprobe": nprobe,
        "overfetch_start": overfetch_start,
        "mode": "replay",
        "seed": 0,
        "pin_cpu": pin_cpu,
        "policy": policy,
        "policy_params": pp,
        "budget": {"kappa_RE": 0.0, "c_ref_cpu_ms": 0.0},
        "run_id": run_id,
        "delta_mode": "scan",
    }


def write_dp_and_stats(rows: list[dict], noop_by_trace: dict[str, dict], out_dir: Path) -> None:
    milp_cache: dict[tuple[str, int], tuple[float, float]] = {}
    out_rows = []
    for r in rows:
        tr = r["trace"]
        B = int(r["n_actions"])
        q_noop = noop_by_trace[tr]["q_accesses"]
        d_noop = noop_by_trace[tr]["delta_n"]
        noop_ops = float((q_noop * d_noop).sum())
        key = (tr, B)
        if key not in milp_cache:
            if B == 0:
                milp_cache[key] = (noop_ops, noop_ops)
            else:
                opt_ops, _ = solve_segment_flow_milp(q_noop, d_noop, B=B, K=K_BLOCK)
                myo_ops = solve_myopic_1blk(q_noop, d_noop, B=B, K=K_BLOCK)
                milp_cache[key] = (opt_ops, myo_ops)
        opt_ops, myo_ops = milp_cache[key]
        ops = float(r["exact_scan_ops"])
        denom = noop_ops - opt_ops
        gcr = 1.0 if B == 0 else ((noop_ops - ops) / denom if denom > 1e-9 else 1.0)
        cr = ops / max(opt_ops, 1e-9)
        myo_gap = (myo_ops - opt_ops) / max(opt_ops, 1e-9) * 100.0
        rec = {k: v for k, v in r.items() if k not in ("q_accesses", "delta_n")}
        rec.update({
            "noop_scan_ops": noop_ops,
            "global_opt_ops": opt_ops,
            "myopic_opt_ops": myo_ops,
            "global_capture_ratio": gcr,
            "competitive_ratio": cr,
            "myopic_vs_global_gap_pct": myo_gap,
        })
        out_rows.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(out_dir / "exact_ops_and_dp.csv", index=False)
    max_ratio = float(df["wall_cpu_ratio"].max()) if "wall_cpu_ratio" in df.columns and len(df) else float("nan")
    log(f"  -> Wrote {out_dir / 'exact_ops_and_dp.csv'} ({len(df)} rows, max wall/cpu={max_ratio:.4f})")


def write_pred_csv(path: Path, pred_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trace", "j", "shard", "model", "q_pred"])
        w.writeheader()
        w.writerows(pred_rows)


# -----------------------------------------------------------------------------
# Helper: Prepare Variant Directory (Hardlinks + Custom K-Means for R != 16)
# -----------------------------------------------------------------------------
def _link_safe(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def ensure_variant_dir(src_vdir: Path, dst_vdir: Path, R: int = 16, Q: int = 2500) -> None:
    dst_vdir.mkdir(parents=True, exist_ok=True)
    (dst_vdir / "traces").mkdir(parents=True, exist_ok=True)
    src_pool = src_vdir / "pool.bin"
    dst_pool = dst_vdir / "pool.bin"
    _link_safe(src_pool, dst_pool)

    if R == 16 and (src_vdir / "init.bin").exists() and (src_vdir / "centroids.f32").exists():
        for fn in ("init.bin", "centroids.f32", "profile.json"):
            sp, dp = src_vdir / fn, dst_vdir / fn
            if sp.exists() and not dp.exists():
                if fn == "profile.json":
                    prof = json.loads(sp.read_text(encoding="utf-8"))
                    prof["queries_per_window"] = Q
                    dp.write_text(json.dumps(prof, indent=2), encoding="utf-8")
                else:
                    _link_safe(sp, dp)
        return

    if str(WORKLOAD_DIR) not in sys.path:
        sys.path.insert(0, str(WORKLOAD_DIR))
    import common as C

    cen_path = dst_vdir / "centroids.f32"
    init_path = dst_vdir / "init.bin"
    prof_path = dst_vdir / "profile.json"
    if cen_path.exists() and init_path.exists() and prof_path.exists():
        return

    src_prof = json.loads((src_vdir / "profile.json").read_text(encoding="utf-8"))
    n_init = int(src_prof["n_init"])
    dim = int(src_prof["dim"])
    mm = C.load_pool(dst_pool)
    X_init = np.asarray(mm[:n_init], dtype=np.float32)

    rng = np.random.default_rng(42 + R)
    cent = X_init[rng.choice(n_init, R, replace=False)].copy()
    for _ in range(25):
        cn2 = (cent ** 2).sum(1)
        assign = np.empty(n_init, dtype=np.int32)
        for i in range(0, n_init, 20000):
            xb = X_init[i : i + 20000]
            d2 = (xb ** 2).sum(1, keepdims=True) + cn2[None, :] - 2.0 * (xb @ cent.T)
            assign[i : i + 20000] = d2.argmin(1)
        new_c = np.zeros_like(cent)
        for r_i in range(R):
            m = assign == r_i
            new_c[r_i] = X_init[m].mean(0) if m.any() else X_init[rng.integers(0, n_init)]
        cent = new_c.astype(np.float32)

    cn2 = (cent ** 2).sum(1)
    assign = np.empty(n_init, dtype=np.uint32)
    for i in range(0, n_init, 20000):
        xb = X_init[i : i + 20000]
        d2 = (xb ** 2).sum(1, keepdims=True) + cn2[None, :] - 2.0 * (xb @ cent.T)
        assign[i : i + 20000] = d2.argmin(1).astype(np.uint32)

    C.write_centroids(cen_path, cent)
    C.write_init(init_path, np.arange(1, n_init + 1, dtype=np.uint64), assign,
                 np.arange(n_init, dtype=np.uint32), dim)

    prof = dict(src_prof)
    prof["R"] = R
    prof["queries_per_window"] = Q
    prof["files"] = {
        "pool.bin": {"bytes": dst_pool.stat().st_size, "sha256": C.sha256_file(dst_pool)},
        "init.bin": {"bytes": init_path.stat().st_size, "sha256": C.sha256_file(init_path)},
        "centroids.f32": {"bytes": cen_path.stat().st_size, "sha256": C.sha256_file(cen_path)},
    }
    prof_path.write_text(json.dumps(prof, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# MODULE: p0_2 — Long-Horizon Multi-Cycle Seasonal Positive Control
# -----------------------------------------------------------------------------
def run_p0_2() -> None:
    log("=== [P0-2] Long-Horizon Multi-Cycle Seasonal Positive Control (T=40,80,120,160) ===")
    base_wl = yaml.safe_load(WORKLOAD_YAML.read_text(encoding="utf-8"))
    src_vdir = QBAM_DATA / "sift100k_u10_q2500_v2"
    horizons = [40, 80, 120, 160]
    seeds = list(range(5, 11))

    build_jobs = []
    for T_win in horizons:
        wl = json.loads(json.dumps(base_wl))
        wl["stream"]["n_windows"] = T_win
        wl["stream"]["queries_per_window"] = 1000
        wl["drift"]["periodic"]["period_windows"] = 20
        droot = SPRINT_DATA / f"seasonal_T{T_win}"
        droot.mkdir(parents=True, exist_ok=True)
        ypath = droot / f"workload_T{T_win}.yaml"
        ypath.write_text(yaml.safe_dump(wl, sort_keys=False), encoding="utf-8")
        suff = f"q1000_T{T_win}"
        vdir = droot / f"sift100k_u05_{suff}"
        ensure_variant_dir(src_vdir, vdir, R=16, Q=1000)
        for s in seeds:
            build_jobs.append((T_win, droot, ypath, suff, vdir, s))

    def _build_one(job):
        T_win, droot, ypath, suff, vdir, s = job
        tname = f"periodic_s{s:02d}"
        tdir = vdir / "traces" / tname
        if not (tdir / "trace.bin").exists():
            subprocess.run([
                sys.executable, str(WORKLOAD_DIR / "generate_trace.py"),
                "--dataset", "sift100k", "--rate", "0.05", "--drift", "periodic", "--seed", str(s),
                "--variant-suffix", suff, "--queries-per-window", "1000",
                "--randomize-hubs-per-seed", "--hub-salt", "101",
                "--workload-config", str(ypath), "--data-root", str(droot), "--force",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"T{T_win}/{tname}"

    log(f"[P0-2] Building {len(build_jobs)} seasonal traces (BUILD_W={BUILD_W})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=BUILD_W) as ex:
        done = list(ex.map(_build_one, build_jobs))
    log(f"[P0-2] Built/verified {len(done)} seasonal traces.")

    for T_win in horizons:
        suff = f"q1000_T{T_win}"
        vdir = SPRINT_DATA / f"seasonal_T{T_win}" / f"sift100k_u05_{suff}"
        out_dir = SPRINT_RES / "p0_2_long_seasonal" / f"T{T_win}"
        out_dir.mkdir(parents=True, exist_ok=True)
        traces = [f"periodic_s{s:02d}" for s in seeds]

        def _noop(idx_tr):
            idx, tr = idx_tr
            pin = POOL_6[idx % len(POOL_6)]
            return tr, run_replay(make_cfg(vdir, tr, "noop", {}, f"p02_T{T_win}_{tr}__noop", pin, use_gt=False),
                                  out_dir / f"{tr}__noop.jsonl")

        noop_by_tr, rows = {}, []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(POOL_6)) as ex:
            for tr, res in ex.map(_noop, enumerate(traces)):
                noop_by_tr[tr] = res
                rows.append({"horizon_T": T_win, "trace": tr, "arm": "noop", **res})

        pred_rows = []
        for tr in traces:
            q_m, d_m = noop_by_tr[tr]["q_accesses"], noop_by_tr[tr]["delta_n"]
            preds = build_single_stream_online_preds(q_m, d_m, use_seasonal_lags=True)
            J, R = preds["ewma_a03"].shape
            for j in range(J):
                for r in range(R):
                    for mn, mat in preds.items():
                        pred_rows.append({"trace": tr, "j": j, "shard": r, "model": mn, "q_pred": f"{mat[j, r]:.6f}"})
        pcsv = out_dir / "predictions.csv"
        write_pred_csv(pcsv, pred_rows)

        def _replay_tr(idx_tr):
            idx, tr = idx_tr
            pin = POOL_6[idx % len(POOL_6)]
            local = []
            for aname, pol, pp in [
                ("forecast_ewma_a03", "forecast_qmass", {"pred_path": str(pcsv), "pred_model": "ewma_a03", "trace_name": tr, "qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
                ("forecast_seasonal_acf", "forecast_qmass", {"pred_path": str(pcsv), "pred_model": "seasonal_acf", "trace_name": tr, "qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
                ("forecast_gbrt_online", "forecast_qmass", {"pred_path": str(pcsv), "pred_model": "gbrt_online", "trace_name": tr, "qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
                ("forecast_oracle", "forecast_qmass", {"pred_path": str(pcsv), "pred_model": "oracle_clairvoyant", "trace_name": tr, "qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
            ]:
                res = run_replay(make_cfg(vdir, tr, pol, pp, f"p02_T{T_win}_{tr}__{aname}", pin, use_gt=False),
                                 out_dir / f"{tr}__{aname}.jsonl")
                local.append({"horizon_T": T_win, "trace": tr, "arm": aname, **res})
            return local

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(POOL_6)) as ex:
            for ch in ex.map(_replay_tr, enumerate(traces)):
                rows.extend(ch)
        write_dp_and_stats(rows, noop_by_tr, out_dir)
    log("=== [P0-2] COMPLETE ===")


# -----------------------------------------------------------------------------
# MODULE: p1_1b — Partition Granularity Sweep R in {8, 16, 32, 64}
# -----------------------------------------------------------------------------
def run_p1_1b() -> None:
    log("=== [P1-1b] Partition Granularity Sweep R in {8, 16, 32, 64} ===")
    base_wl = yaml.safe_load(WORKLOAD_YAML.read_text(encoding="utf-8"))
    src_100k = QBAM_DATA / "sift100k_u10_q2500_v2"
    r_traces = [f"{d}_s{s:02d}" for d in ("abrupt", "gradual", "hotspot", "uniform") for s in (5, 6)]

    r_build_jobs = []
    for R_val in (8, 16, 32, 64):
        nprobe = max(2, R_val // 4)
        wl = json.loads(json.dumps(base_wl))
        wl["stream"]["queries_per_window"] = 1000
        wl["data_shard_hotspots"]["derive_groups_from_queries"] = True
        wl["data_shard_hotspots"]["shard_groups"] = {
            "A": list(range(0, R_val, 2)),
            "B": list(range(1, R_val, 2)),
        }
        wl["data_shard_hotspots"]["hubs"] = {"A": [0, 2], "B": [1, 3]}
        wl["diagnostics"]["nprobe"] = nprobe
        droot = SPRINT_DATA / f"R_{R_val}"
        droot.mkdir(parents=True, exist_ok=True)
        ypath = droot / f"workload_R{R_val}.yaml"
        ypath.write_text(yaml.safe_dump(wl, sort_keys=False), encoding="utf-8")
        suff = f"q1000_R{R_val}"
        vdir = droot / f"sift100k_u10_{suff}"
        ensure_variant_dir(src_100k, vdir, R=R_val, Q=1000)
        for tr in r_traces:
            drift, ss = tr.rsplit("_s", 1)
            r_build_jobs.append((R_val, droot, ypath, suff, vdir, drift, int(ss)))

    def _build_r(job):
        R_val, droot, ypath, suff, vdir, drift, seed = job
        tname = f"{drift}_s{seed:02d}"
        tdir = vdir / "traces" / tname
        if not (tdir / "trace.bin").exists():
            subprocess.run([
                sys.executable, str(WORKLOAD_DIR / "generate_trace.py"),
                "--dataset", "sift100k", "--rate", "0.10", "--drift", drift, "--seed", str(seed),
                "--variant-suffix", suff, "--queries-per-window", "1000",
                "--randomize-hubs-per-seed", "--hub-salt", "77",
                "--workload-config", str(ypath), "--data-root", str(droot), "--force",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"R{R_val}/{tname}"

    log(f"[P1-1b] Building {len(r_build_jobs)} R-sweep traces (BUILD_W={BUILD_W})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=BUILD_W) as ex:
        list(ex.map(_build_r, r_build_jobs))

    for R_val in (8, 16, 32, 64):
        nprobe = max(2, R_val // 4)
        suff = f"q1000_R{R_val}"
        vdir = SPRINT_DATA / f"R_{R_val}" / f"sift100k_u10_{suff}"
        out_r = SPRINT_RES / "p1_1_scale_and_R" / f"R_{R_val}"
        out_r.mkdir(parents=True, exist_ok=True)

        def _r_noop(idx_tr):
            idx, tr = idx_tr
            pin = POOL_8[idx % len(POOL_8)]
            return tr, run_replay(make_cfg(vdir, tr, "noop", {}, f"r{R_val}_{tr}__noop", pin, R=R_val, nprobe=nprobe, use_gt=False),
                                  out_r / f"{tr}__noop.jsonl")

        noop_r, rows_r = {}, []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(POOL_8)) as ex:
            for tr, res in ex.map(_r_noop, enumerate(r_traces)):
                noop_r[tr] = res
                rows_r.append({"R": R_val, "nprobe": nprobe, "trace": tr, "arm": "noop", **res})

        pred_r = []
        for tr in r_traces:
            q_m = noop_r[tr]["q_accesses"]
            T, Rv = q_m.shape
            J = T // K_BLOCK
            for j in range(J):
                w0, w1 = j * K_BLOCK, min((j + 1) * K_BLOCK, T)
                yb = q_m[w0:w1].mean(axis=0)
                for r in range(Rv):
                    pred_r.append({"trace": tr, "j": j, "shard": r, "model": "max_delta_sp1", "q_pred": "1.0"})
                    pred_r.append({"trace": tr, "j": j, "shard": r, "model": "oracle_clairvoyant", "q_pred": f"{yb[r]:.6f}"})
        pcsv_r = out_r / "predictions.csv"
        write_pred_csv(pcsv_r, pred_r)

        def _r_arms(idx_tr):
            idx, tr = idx_tr
            pin = POOL_8[idx % len(POOL_8)]
            local = []
            for aname, pol, pp in [
                ("fixed_c1_merge", "fixed", {"cadence_blocks": 1, "phase": 1, "shard_order": "round_robin", "action_mode": "merge"}),
                ("max_delta_sp1", "forecast_qmass", {"pred_path": str(pcsv_r), "pred_model": "max_delta_sp1", "trace_name": tr, "qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
                ("past_qmass_sp1", "past_qmass", {"qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
                ("forecast_oracle", "forecast_qmass", {"pred_path": str(pcsv_r), "pred_model": "oracle_clairvoyant", "trace_name": tr, "qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
            ]:
                res = run_replay(make_cfg(vdir, tr, pol, pp, f"r{R_val}_{tr}__{aname}", pin, R=R_val, nprobe=nprobe, use_gt=False),
                                 out_r / f"{tr}__{aname}.jsonl")
                local.append({"R": R_val, "nprobe": nprobe, "trace": tr, "arm": aname, **res})
            return local

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(POOL_8)) as ex:
            for ch in ex.map(_r_arms, enumerate(r_traces)):
                rows_r.extend(ch)
        write_dp_and_stats(rows_r, noop_r, out_r)
    log("=== [P1-1b] COMPLETE ===")


# -----------------------------------------------------------------------------
# MODULE: p1_1c — efSearch Sensitivity {16, 32, 64, 128}
# -----------------------------------------------------------------------------
def run_p1_1c(overfetch_start: int = 40, subdir: str = "efsearch_sweep") -> None:
    log(f"=== [P1-1c] efSearch Sweep {{16, 32, 64, 128}} (overfetch_start={overfetch_start}) ===")
    canon_dir = QBAM_DATA / "sift100k_u10_q2500_v2"
    ef_traces = [f"{d}_s{s:02d}" for d in ("abrupt", "gradual", "hotspot") for s in (5, 6)]
    out_ef = SPRINT_RES / "p1_1_scale_and_R" / subdir
    out_ef.mkdir(parents=True, exist_ok=True)
    log(f"[P1-1c/{subdir}] Running efSearch sweep {{16,32,64,128}} on {len(ef_traces)} traces...")

    def _ef_job(idx_tr):
        idx, tr = idx_tr
        pin = POOL_6[idx % len(POOL_6)]
        local = []
        for ef in (16, 32, 64, 128):
            for aname, pol, pp in [
                ("noop", "noop", {}),
                ("fixed_c1_merge", "fixed", {"cadence_blocks": 1, "phase": 1, "shard_order": "round_robin", "action_mode": "merge"}),
                ("past_qmass_sp1", "past_qmass", {"qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"}),
            ]:
                res = run_replay(
                    make_cfg(canon_dir, tr, pol, pp, f"ef{ef}_of{overfetch_start}_{tr}__{aname}",
                             pin, ef_search=ef, overfetch_start=overfetch_start, use_gt=True),
                    out_ef / f"ef{ef}__{tr}__{aname}.jsonl",
                )
                local.append({"efSearch": ef, "overfetch_start": overfetch_start, "trace": tr, "arm": aname,
                              **{k: v for k, v in res.items() if k not in ("q_accesses", "delta_n")}})
        return local

    ef_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(POOL_6)) as ex:
        for ch in ex.map(_ef_job, enumerate(ef_traces)):
            ef_rows.extend(ch)
    df = pd.DataFrame(ef_rows)
    df.to_csv(out_ef / "efsearch_summary.csv", index=False)
    log(f"  -> Wrote {out_ef / 'efsearch_summary.csv'} ({len(df)} rows, max wall/cpu={df['wall_cpu_ratio'].max():.4f})")
    log("=== [P1-1c] COMPLETE ===")


# -----------------------------------------------------------------------------
# MODULE: p1_2 — Clean Fine-Grid Tombstone Phase Transition (0..20% CRUD)
# -----------------------------------------------------------------------------
def run_p1_2() -> None:
    log("=== [P1-2] Fine-Grid Tombstone Phase Transition (0,2,4,6,8,10,20%) ===")
    base_wl = yaml.safe_load(WORKLOAD_YAML.read_text(encoding="utf-8"))
    src_100k = QBAM_DATA / "sift100k_u10_q2500_v2"
    traces = [f"{d}_s{s:02d}" for d in ("abrupt", "hotspot") for s in (5, 6, 7)]

    crud_specs = [
        (0, {"mode": "insert_only"}),
        (2, {"mode": "mixed", "insert_fraction": 0.98, "update_fraction": 0.013, "erase_fraction": 0.007}),
        (4, {"mode": "mixed", "insert_fraction": 0.96, "update_fraction": 0.027, "erase_fraction": 0.013}),
        (6, {"mode": "mixed", "insert_fraction": 0.94, "update_fraction": 0.040, "erase_fraction": 0.020}),
        (8, {"mode": "mixed", "insert_fraction": 0.92, "update_fraction": 0.053, "erase_fraction": 0.027}),
        (10, {"mode": "mixed", "insert_fraction": 0.90, "update_fraction": 0.060, "erase_fraction": 0.040}),
        (20, {"mode": "mixed", "insert_fraction": 0.80, "update_fraction": 0.130, "erase_fraction": 0.070}),
    ]

    build_jobs = []
    vdirs = {}
    for pct, cdict in crud_specs:
        wl = json.loads(json.dumps(base_wl))
        wl["crud"] = cdict
        droot = SPRINT_DATA / f"fine_crud_{pct:02d}"
        droot.mkdir(parents=True, exist_ok=True)
        ypath = droot / f"workload_crud{pct:02d}.yaml"
        ypath.write_text(yaml.safe_dump(wl, sort_keys=False), encoding="utf-8")
        suff = f"q2500_finecrud{pct:02d}"
        vdir = droot / f"sift100k_u10_{suff}"
        ensure_variant_dir(src_100k, vdir, R=16, Q=2500)
        vdirs[pct] = vdir
        for tr in traces:
            drift, ss = tr.rsplit("_s", 1)
            build_jobs.append((pct, droot, ypath, suff, vdir, drift, int(ss)))

    def _build_crud(job):
        pct, droot, ypath, suff, vdir, drift, seed = job
        tname = f"{drift}_s{seed:02d}"
        tdir = vdir / "traces" / tname
        if not (tdir / "trace.bin").exists():
            subprocess.run([
                sys.executable, str(WORKLOAD_DIR / "generate_trace.py"),
                "--dataset", "sift100k", "--rate", "0.10", "--drift", drift, "--seed", str(seed),
                "--variant-suffix", suff, "--queries-per-window", "2500",
                "--randomize-hubs-per-seed", "--hub-salt", "0",
                "--workload-config", str(ypath), "--data-root", str(droot), "--force",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not (tdir / "gt.bin").exists():
            subprocess.run([
                sys.executable, str(WORKLOAD_DIR / "compute_gt.py"),
                "--dataset", "sift100k", "--rate", "0.10", "--drift", drift, "--seed", str(seed),
                "--variant-suffix", suff, "--data-root", str(droot),
                "--device", "cpu", "--verify-n", "10", "--force",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"crud{pct:02d}/{tname}"

    log(f"[P1-2] Building {len(build_jobs)} fine-grid CRUD traces + CPU GT (BUILD_W={BUILD_W})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=BUILD_W) as ex:
        list(ex.map(_build_crud, build_jobs))

    out_root = SPRINT_RES / "p1_2_fine_tombstone"
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for pct, _ in crud_specs:
        vdir = vdirs[pct]
        out_p = out_root / f"crud{pct:02d}"
        out_p.mkdir(parents=True, exist_ok=True)

        def _pair_job(idx_tr):
            idx, tr = idx_tr
            pin = POOL_6[idx % len(POOL_6)]
            r_sp1 = run_replay(
                make_cfg(vdir, tr, "past_qmass", {"qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"},
                         f"fcrud{pct:02d}_{tr}__sp1", pin, use_gt=True),
                out_p / f"{tr}__past_qmass_sp1.jsonl",
            )
            r_ref = run_replay(
                make_cfg(vdir, tr, "past_qmass", {"qmass_window": 4, "spacing_blocks": 1, "action_mode": "refresh"},
                         f"fcrud{pct:02d}_{tr}__ref", pin, use_gt=True),
                out_p / f"{tr}__past_qmass_refresh.jsonl",
            )
            return [
                {"crud_pct": pct, "trace": tr, "arm": "past_qmass_sp1", **{k: v for k, v in r_sp1.items() if k not in ("q_accesses", "delta_n")}},
                {"crud_pct": pct, "trace": tr, "arm": "past_qmass_refresh", **{k: v for k, v in r_ref.items() if k not in ("q_accesses", "delta_n")}},
            ]

        pct_rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(POOL_6)) as ex:
            for ch in ex.map(_pair_job, enumerate(traces)):
                pct_rows.extend(ch)
        df_p = pd.DataFrame(pct_rows)
        df_p.to_csv(out_p / "summary.csv", index=False)
        all_rows.extend(pct_rows)
        log(f"[P1-2] CRUD {pct:02d}% DONE ({len(pct_rows)} runs, max wall/cpu={df_p['wall_cpu_ratio'].max():.4f})")

    pd.DataFrame(all_rows).to_csv(out_root / "fine_tombstone_all.csv", index=False)
    log("=== [P1-2] COMPLETE ===")


# -----------------------------------------------------------------------------
# SMOKE TEST
# -----------------------------------------------------------------------------
def run_smoke_test() -> None:
    log("=== [SMOKE TEST] Checking build_rev4 gt:null, kappa_RE=0.0 actions, efSearch, wall/cpu ===")
    base_wl = yaml.safe_load(WORKLOAD_YAML.read_text(encoding="utf-8"))
    src_vdir = QBAM_DATA / "sift100k_u10_q2500_v2"
    smoke_data = SPRINT_DATA / "_smoke_test"
    smoke_res = SPRINT_RES / "_smoke_test"
    shutil.rmtree(smoke_data, ignore_errors=True)
    shutil.rmtree(smoke_res, ignore_errors=True)
    smoke_data.mkdir(parents=True, exist_ok=True)
    smoke_res.mkdir(parents=True, exist_ok=True)

    wl = json.loads(json.dumps(base_wl))
    wl["stream"]["n_windows"] = 40
    wl["stream"]["queries_per_window"] = 1000
    wl["drift"]["periodic"]["period_windows"] = 20
    ypath = smoke_data / "workload_T40.yaml"
    ypath.write_text(yaml.safe_dump(wl, sort_keys=False), encoding="utf-8")
    suff = "q1000_smoke40"
    vdir = smoke_data / f"sift100k_u05_{suff}"
    ensure_variant_dir(src_vdir, vdir, R=16, Q=1000)

    subprocess.run([
        sys.executable, str(WORKLOAD_DIR / "generate_trace.py"),
        "--dataset", "sift100k", "--rate", "0.05", "--drift", "periodic", "--seed", "5",
        "--variant-suffix", suff, "--queries-per-window", "1000",
        "--randomize-hubs-per-seed", "--hub-salt", "101",
        "--workload-config", str(ypath), "--data-root", str(smoke_data), "--force",
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    r_noop = run_replay(
        make_cfg(vdir, "periodic_s05", "noop", {}, "smoke_p02_noop", pin_cpu=0, use_gt=False),
        smoke_res / "periodic_s05__noop.jsonl",
        force=True,
    )
    r_qmass = run_replay(
        make_cfg(vdir, "periodic_s05", "past_qmass", {"qmass_window": 4, "spacing_blocks": 1, "action_mode": "merge"},
                 "smoke_p02_qmass", pin_cpu=2, use_gt=False),
        smoke_res / "periodic_s05__past_qmass_sp1.jsonl",
        force=True,
    )
    r_ef16 = run_replay(
        make_cfg(src_vdir, "abrupt_s05", "fixed", {"cadence_blocks": 1, "phase": 1, "shard_order": "round_robin", "action_mode": "merge"},
                 "smoke_ef16", pin_cpu=4, ef_search=16, use_gt=True),
        smoke_res / "ef16_abrupt_s05__fixed_c1.jsonl",
        force=True,
    )

    for label, r in [("p02_noop", r_noop), ("p02_past_qmass", r_qmass), ("ef16_fixed_c1", r_ef16)]:
        log(f"  [{label}] n_actions={r['n_actions']}, cpu={r['total_cpu_ms']:.1f}ms, wall={r['wall_ms']:.1f}ms, ratio={r['wall_cpu_ratio']:.4f}, recall={r['mean_recall']:.4f}")
        assert r["wall_ms"] <= r["total_cpu_ms"] * 1.15 + 6500.0, f"{label} wall_ms {r['wall_ms']} > 1.15*cpu+6500"
    assert r_ef16["wall_cpu_ratio"] <= 1.15, f"ef16_fixed_c1 wall_cpu_ratio {r_ef16['wall_cpu_ratio']} > 1.15"
    assert r_qmass["n_actions"] == 9, f"Expected n_actions==9 for T=40 spacing=1, got {r_qmass['n_actions']}"
    assert r_ef16["n_actions"] == 9, f"Expected n_actions==9 for ef16 fixed_c1, got {r_ef16['n_actions']}"
    assert not np.isnan(r_ef16["mean_recall"]) and r_ef16["mean_recall"] > 0.5, f"Bad ef16 recall: {r_ef16['mean_recall']}"

    shutil.rmtree(smoke_data, ignore_errors=True)
    shutil.rmtree(smoke_res, ignore_errors=True)
    log("=== [SMOKE TEST] ALL 3 ASSERTIONS PASSED & CLEANED UP ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="p0_2,p1_1b,p1_1c,p1_2", help="Comma-separated modules to run")
    ap.add_argument("--smoke-test", action="store_true", help="Run smoke test and exit")
    args = ap.parse_args()
    if args.smoke_test:
        run_smoke_test()
        return
    mods = [m.strip() for m in args.modules.split(",") if m.strip()]
    t0 = time.perf_counter()
    if "p0_2" in mods:
        run_p0_2()
    if "p1_1b" in mods:
        run_p1_1b()
    if "p1_1c" in mods:
        run_p1_1c()
    if "p1_1c_of10" in mods:
        run_p1_1c(overfetch_start=10, subdir="efsearch_sweep_of10")
    if "p1_2" in mods:
        run_p1_2()
    log(f"ALL MIMO SPRINT V2 MODULES FINISHED in {(time.perf_counter() - t0) / 60.0:.1f} min.")


if __name__ == "__main__":
    main()
