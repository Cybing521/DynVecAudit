#!/usr/bin/env python3
"""Exp 2 & Exp 3 Orchestrator for PVLDB EA&B (DynVecAudit).

Runs Phase 1 (SIFT-128 on Ryzen 9 9950X) or Phase 2 (MiniLM-384 / CLIP-512 on EPYC 7343):
  - Suite `sift25`   : 25 traces (5 drifts x 5 seeds) x {noop, past_qmass, bam_price,
                      8 forecast_qmass predictors, 8 executed fixed cadences c=1..8}
  - Suite `phaselock`: t_sw in {16,20,24,28} x 5 seeds x {noop, spacing=1/2/3, fixed c=1..4}
  - Suite `refresh`  : action_mode="refresh" comparison against "merge"

Pre-replay hygiene & diagnostics:
  - Generates causal and leaky forecasts into `<out_dir>/predictions.csv`
  - Runs `audit_hygiene.check_clock_leak()` on causal vs leaky feature matrices
  - Computes block-level Top-1 shard agreement (overall + abrupt j=5,6) and JSD
  - Pins every qbam_replay execution with `taskset -c <core> nice -n 19` + load gate
  - Audits completed JSONLs with `audit_hygiene.check_contamination()` & `check_cadence_bracket()`
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import audit_hygiene as ah  # noqa: E402

DRIFTS_DEFAULT = ("hotspot", "abrupt", "gradual", "uniform", "periodic")
SEEDS_DEFAULT = ("s05", "s06", "s07", "s08", "s09")
PREDICTOR_MODELS = (
    "past_mean",
    "ewma_a03",
    "ewma_a07",
    "cusum_reset",
    "ridge_ar",
    "gbrt_causal",
    "gbrt_leaky",
    "oracle_clairvoyant",
)
FIXED_CADENCES = (1, 2, 3, 4, 5, 6, 7, 8)
PHASELOCK_TSWS = (16, 20, 24, 28)

DEFAULT_C_SCAN = 3.2546e-5
DEFAULT_C_INSERT = 0.04573
DEFAULT_C_REF = 1094.5
DEFAULT_K = 4


# ---------------------------------------------------------------------------
# Binary readers matching QBAM frozen contract (self-contained, stdlib/numpy only)
# ---------------------------------------------------------------------------
def _read_header(path: Path, magic: bytes, fmt: str) -> Tuple[Tuple[int, ...], int]:
    with path.open("rb") as f:
        got = f.read(8)
        if got != magic:
            raise ValueError(f"{path}: bad magic {got!r} != {magic!r}")
        vals = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        size = f.tell()
    return vals, size


def load_trace_shard_series(
    data_root: Path,
    trace_dir: Path,
    nprobe: int = 4,
) -> Dict[str, Any]:
    """Compute realized per-window shard query accesses, delta inserts, and base sizes."""
    import numpy as np

    pool_vals, pool_off = _read_header(data_root / "pool.bin", b"QBAMPOOL", "<III")
    _, dim, nvec = pool_vals
    pool = np.fromfile(data_root / "pool.bin", dtype="<f4", offset=pool_off).reshape(nvec, dim)

    init_vals, init_off = _read_header(data_root / "init.bin", b"QBAMINIT", "<IIII")
    _, _, n_init, R = init_vals
    init_dtype = np.dtype([("id", "<u8"), ("shard", "<u4"), ("vec_ref", "<u4")])
    init_rec = np.fromfile(data_root / "init.bin", dtype=init_dtype, offset=init_off, count=n_init)

    centroids = np.fromfile(data_root / "centroids.f32", dtype="<f4").reshape(R, dim)

    tr_vals, tr_off = _read_header(trace_dir / "trace.bin", b"QBAMTRAC", "<IIIII")
    _, _, n_events, n_windows, K = tr_vals
    ev_dtype = np.dtype(
        [("window", "<u4"), ("type", "u1"), ("id", "<u8"), ("version", "<u8"), ("vec_ref", "<u4"), ("flags", "u1")]
    )
    ev = np.fromfile(trace_dir / "trace.bin", dtype=ev_dtype, offset=tr_off, count=n_events)

    base_n = np.bincount(init_rec["shard"], minlength=R).astype(np.float64)
    q_accesses = np.zeros((n_windows, R), dtype=np.float64)
    ins_per_win = np.zeros((n_windows, R), dtype=np.float64)

    # Batch nearest-centroid routing matching L2 routing in ReplayEngine
    c_norm2 = np.sum(centroids ** 2, axis=1)

    q_mask = ev["type"] == 3  # EV_QUERY
    if np.any(q_mask):
        q_ev = ev[q_mask]
        q_vecs = pool[q_ev["vec_ref"]]
        # Compute squared L2 in chunks to keep memory tiny even for 1M / high-dim
        chunk = 4096
        for lo in range(0, q_vecs.shape[0], chunk):
            hi = min(q_vecs.shape[0], lo + chunk)
            v = q_vecs[lo:hi]
            d2 = np.sum(v ** 2, axis=1, keepdims=True) + c_norm2[None, :] - 2.0 * (v @ centroids.T)
            top_shards = np.argpartition(d2, kth=nprobe - 1, axis=1)[:, :nprobe]
            wins = q_ev["window"][lo:hi]
            for row_w, shards in zip(wins.tolist(), top_shards.tolist()):
                for s in shards:
                    q_accesses[row_w, s] += 1.0

    ins_mask = (ev["type"] == 0) | (ev["type"] == 1)  # EV_INSERT | EV_UPDATE
    if np.any(ins_mask):
        i_ev = ev[ins_mask]
        i_vecs = pool[i_ev["vec_ref"]]
        chunk = 4096
        for lo in range(0, i_vecs.shape[0], chunk):
            hi = min(i_vecs.shape[0], lo + chunk)
            v = i_vecs[lo:hi]
            d2 = np.sum(v ** 2, axis=1, keepdims=True) + c_norm2[None, :] - 2.0 * (v @ centroids.T)
            best_shards = np.argmin(d2, axis=1)
            wins = i_ev["window"][lo:hi]
            for row_w, s in zip(wins.tolist(), best_shards.tolist()):
                ins_per_win[row_w, s] += 1.0

    cum_delta = np.cumsum(ins_per_win, axis=0)
    return {
        "R": int(R),
        "dim": int(dim),
        "n_windows": int(n_windows),
        "K": int(K),
        "base_n": base_n,
        "q_accesses": q_accesses,
        "ins_per_win": ins_per_win,
        "cum_delta": cum_delta,
    }


# ---------------------------------------------------------------------------
# Causal & Leaky Forecasters + Pre-Replay Diagnostics
# ---------------------------------------------------------------------------
def _ewma_next(q_hist: Any, alpha: float) -> Any:
    import numpy as np

    if q_hist.shape[0] == 0:
        return np.zeros(q_hist.shape[1], dtype=np.float64)
    m = q_hist[0].astype(np.float64).copy()
    for w in range(1, q_hist.shape[0]):
        m = alpha * q_hist[w] + (1.0 - alpha) * m
    return m


def _cusum_next(q_hist: Any, k_sigma: float = 3.0, alpha: float = 0.5) -> Any:
    import numpy as np

    n, R = q_hist.shape
    if n == 0:
        return np.zeros(R, dtype=np.float64)
    m = q_hist[0].astype(np.float64).copy()
    var = np.ones(R, dtype=np.float64)
    s_hi = np.zeros(R, dtype=np.float64)
    s_lo = np.zeros(R, dtype=np.float64)
    for w in range(1, n):
        v = q_hist[w]
        sigma = np.sqrt(np.maximum(var, 1e-6))
        s_hi = np.maximum(0.0, s_hi + (v - m) / sigma - k_sigma / 2.0)
        s_lo = np.maximum(0.0, s_lo - (v - m) / sigma - k_sigma / 2.0)
        reset = (s_hi > k_sigma) | (s_lo > k_sigma)
        m = np.where(reset, v, alpha * v + (1.0 - alpha) * m)
        var = np.where(reset, 1.0, (1.0 - alpha) * var + alpha * (v - m) ** 2)
        s_hi = np.where(reset, 0.0, s_hi)
        s_lo = np.where(reset, 0.0, s_lo)
    return m


def _build_block_samples(series_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    import numpy as np

    rows: List[Dict[str, Any]] = []
    for trace_name in sorted(series_map):
        rec = series_map[trace_name]
        q = rec["q_accesses"]
        cum_d = rec["cum_delta"]
        base_n = rec["base_n"]
        K = rec["K"]
        R = rec["R"]
        n_blocks = rec["n_windows"] // K
        parts = trace_name.rsplit("_s", 1)
        drift = parts[0] if len(parts) == 2 else trace_name
        seed = parts[1] if len(parts) == 2 else "00"

        for j in range(1, n_blocks):
            h0 = j * K
            past = q[max(0, h0 - 8): h0]
            L = past.shape[0]
            x = np.arange(L, dtype=np.float64)
            xc = x - x.mean()
            denom = float((xc ** 2).sum())
            slope = (past * xc[:, None]).sum(axis=0) / denom if denom > 0 else np.zeros(R)
            last4 = q[max(0, h0 - 4): h0]
            if last4.shape[0] < 4:
                pad = np.repeat(last4[:1], 4 - last4.shape[0], axis=0)
                last4 = np.concatenate([pad, last4], axis=0)
            d_prev = cum_d[h0 - 1]
            d_prev4 = cum_d[h0 - 5] if h0 >= 5 else np.zeros(R)
            dinc4 = d_prev - d_prev4
            y_true_per_win = q[h0: h0 + K].mean(axis=0)

            rows.append({
                "trace": trace_name,
                "drift": drift,
                "seed": seed,
                "j": j,
                "h0": h0,
                "R": R,
                "qmean8": past.mean(axis=0),
                "qstd8": past.std(axis=0),
                "slope": slope,
                "last4": last4,
                "dinc4": dinc4,
                "dcum_shard": d_prev,
                "dcum_total": float(d_prev.sum()),
                "base_n": base_n,
                "glob_qmean8": float(past.mean(axis=0).sum()),
                "glob_dinc4": float(dinc4.sum()),
                "y_true": y_true_per_win,
            })
    return rows


def _extract_feature_matrix(
    rows: Sequence[Dict[str, Any]],
    leaky: bool = False,
) -> Tuple[Any, Any, List[str]]:
    import numpy as np

    if not rows:
        return np.zeros((0, 1)), np.zeros((0,)), []
    R = rows[0]["R"]
    feat_names = (
        ["qmean8", "qstd8", "slope", "q_lag4", "q_lag3", "q_lag2", "q_lag1", "dinc4", "base_n", "glob_qmean8", "glob_dinc4"]
        + [f"shard_{r}" for r in range(R)]
    )
    if leaky:
        feat_names = feat_names + ["leak_block_j", "leak_dcum_shard", "leak_dcum_total"]

    xs = []
    ys = []
    eye = np.eye(R, dtype=np.float64)
    for row in rows:
        base_block = np.column_stack([
            row["qmean8"],
            row["qstd8"],
            row["slope"],
            row["last4"].T,
            row["dinc4"],
            row["base_n"],
            np.full(R, row["glob_qmean8"]),
            np.full(R, row["glob_dinc4"]),
            eye,
        ])
        if leaky:
            leak_cols = np.column_stack([
                np.full(R, float(row["j"])),
                row["dcum_shard"],
                np.full(R, row["dcum_total"]),
            ])
            base_block = np.hstack([base_block, leak_cols])
        xs.append(base_block)
        ys.append(row["y_true"])
    return np.vstack(xs), np.concatenate(ys), feat_names


def _fit_predict_ridge(X_tr: Any, y_tr: Any, X_te: Any, lam: float = 10.0) -> Any:
    import numpy as np

    mu = X_tr.mean(axis=0)
    sd = np.maximum(X_tr.std(axis=0), 1e-6)
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd
    y_mu = float(y_tr.mean())
    d = Z_tr.shape[1]
    A = Z_tr.T @ Z_tr + lam * np.eye(d)
    w = np.linalg.solve(A, Z_tr.T @ (y_tr - y_mu))
    return np.maximum(0.0, y_mu + Z_te @ w)


def _fit_predict_gbrt(X_tr: Any, y_tr: Any, X_te: Any, seed: int = 42) -> Tuple[Any, str]:
    import numpy as np

    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        return np.maximum(0.0, model.predict(X_te)), "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        return np.maximum(0.0, model.predict(X_te)), "sklearn_histgbr"


def _jsd(p: Sequence[float], q: Sequence[float]) -> float:
    sp = sum(max(0.0, float(x)) for x in p)
    sq = sum(max(0.0, float(x)) for x in q)
    if sp <= 0 or sq <= 0:
        return 0.0
    pn = [max(0.0, float(x)) / sp for x in p]
    qn = [max(0.0, float(x)) / sq for x in q]
    m = [0.5 * (a + b) for a, b in zip(pn, qn)]
    div = 0.0
    for a, b, mm in zip(pn, qn, m):
        if a > 1e-12 and mm > 1e-12:
            div += 0.5 * a * math.log2(a / mm)
        if b > 1e-12 and mm > 1e-12:
            div += 0.5 * b * math.log2(b / mm)
    return div


def generate_predictions_and_audit(
    data_root: Path,
    trace_names: Sequence[str],
    out_dir: Path,
    nprobe: int = 4,
) -> Tuple[Path, Dict[str, Any]]:
    """Build predictions.csv (all 8 models) using leave-one-seed-out CV for learned models."""
    import numpy as np

    series_map = {}
    for tn in trace_names:
        tdir = data_root / "traces" / tn
        if not (tdir / "trace.bin").exists():
            continue
        series_map[tn] = load_trace_shard_series(data_root, tdir, nprobe=nprobe)

    if not series_map:
        raise FileNotFoundError(f"No valid trace.bin found under {data_root / 'traces'}")

    rows = _build_block_samples(series_map)
    R = rows[0]["R"]

    # Audit causal vs leaky feature matrices with DynVecAudit check_clock_leak()
    Xc_all, _, c_names = _extract_feature_matrix(rows, leaky=False)
    Xl_all, _, l_names = _extract_feature_matrix(rows, leaky=True)
    step_vec = [float(r["j"]) for r in rows for _ in range(R)]

    causal_dict = {"_step_j": step_vec}
    for col_i, cname in enumerate(c_names):
        if not cname.startswith("shard_"):
            causal_dict[cname] = Xc_all[:, col_i].tolist()
    causal_leak_audit = ah.check_clock_leak(causal_dict, step_col="_step_j", threshold=0.85)

    leaky_dict = {"_step_j": step_vec}
    for col_i, lname in enumerate(l_names):
        if not lname.startswith("shard_"):
            leaky_dict[lname] = Xl_all[:, col_i].tolist()
    leaky_leak_audit = ah.check_clock_leak(leaky_dict, step_col="_step_j", threshold=0.85)

    # Leave-one-seed-out predictions for ridge_ar, gbrt_causal, gbrt_leaky
    seeds = sorted({r["seed"] for r in rows})
    preds_by_model: Dict[str, Dict[Tuple[str, int], Any]] = {m: {} for m in PREDICTOR_MODELS}
    gbrt_backend = "none"

    for test_seed in seeds:
        tr_rows = [r for r in rows if r["seed"] != test_seed]
        te_rows = [r for r in rows if r["seed"] == test_seed]
        if not tr_rows:
            # Single-seed fallback (e.g. smoke test): train on all rows except test block
            tr_rows = te_rows
        Xtr_c, ytr_c, _ = _extract_feature_matrix(tr_rows, leaky=False)
        Xte_c, _, _ = _extract_feature_matrix(te_rows, leaky=False)
        Xtr_l, ytr_l, _ = _extract_feature_matrix(tr_rows, leaky=True)
        Xte_l, _, _ = _extract_feature_matrix(te_rows, leaky=True)

        yhat_ridge = _fit_predict_ridge(Xtr_c, ytr_c, Xte_c)
        yhat_gbrt_c, gbrt_backend = _fit_predict_gbrt(Xtr_c, ytr_c, Xte_c, seed=42)
        yhat_gbrt_l, _ = _fit_predict_gbrt(Xtr_l, ytr_l, Xte_l, seed=42)

        for idx, r in enumerate(te_rows):
            sl = slice(idx * R, (idx + 1) * R)
            key = (r["trace"], r["j"])
            preds_by_model["ridge_ar"][key] = yhat_ridge[sl]
            preds_by_model["gbrt_causal"][key] = yhat_gbrt_c[sl]
            preds_by_model["gbrt_leaky"][key] = yhat_gbrt_l[sl]

    for r in rows:
        key = (r["trace"], r["j"])
        q = series_map[r["trace"]]["q_accesses"]
        h0 = r["h0"]
        preds_by_model["past_mean"][key] = r["qmean8"]
        preds_by_model["ewma_a03"][key] = _ewma_next(q[:h0], alpha=0.3)
        preds_by_model["ewma_a07"][key] = _ewma_next(q[:h0], alpha=0.7)
        preds_by_model["cusum_reset"][key] = _cusum_next(q[:h0], k_sigma=3.0)
        preds_by_model["oracle_clairvoyant"][key] = r["y_true"]

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["trace", "drift", "seed", "j", "shard", "model", "q_pred", "q_true", "delta_n", "split", "in_sample"])
        for model in PREDICTOR_MODELS:
            for r in rows:
                key = (r["trace"], r["j"])
                qp = preds_by_model[model][key]
                for s in range(R):
                    wr.writerow([
                        r["trace"], r["drift"], r["seed"], r["j"], s, model,
                        f"{float(qp[s]):.6f}", f"{float(r['y_true'][s]):.6f}",
                        f"{float(r['dcum_shard'][s]):.1f}",
                        "loso_cv" if len(seeds) > 1 else "single_seed",
                        "false",
                    ])

    # Pre-replay predictor diagnostics
    diag: Dict[str, Any] = {
        "gbrt_backend": gbrt_backend,
        "clock_leak_audit": {
            "gbrt_causal": causal_leak_audit,
            "gbrt_leaky": leaky_leak_audit,
        },
        "models": {},
    }
    for model in PREDICTOR_MODELS:
        agree_all, agree_sw5, agree_sw6, jsds = [], [], [], []
        for r in rows:
            key = (r["trace"], r["j"])
            qp = preds_by_model[model][key]
            qt = r["y_true"]
            dn = r["dcum_shard"]
            pred_top = int(np.argmax(qp * dn))
            true_top = int(np.argmax(qt * dn))
            hit = 1.0 if pred_top == true_top else 0.0
            agree_all.append(hit)
            if r["drift"] == "abrupt" and r["j"] == 5:
                agree_sw5.append(hit)
            if r["drift"] == "abrupt" and r["j"] == 6:
                agree_sw6.append(hit)
            jsds.append(_jsd(qp.tolist(), qt.tolist()))
        diag["models"][model] = {
            "n_blocks": len(agree_all),
            "top1_agree_all": round(sum(agree_all) / len(agree_all), 4),
            "abrupt_j5_top1": round(sum(agree_sw5) / len(agree_sw5), 4) if agree_sw5 else None,
            "abrupt_j6_top1": round(sum(agree_sw6) / len(agree_sw6), 4) if agree_sw6 else None,
            "mean_jsd": round(sum(jsds) / len(jsds), 4),
        }

    (out_dir / "predictor_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    return pred_csv, diag


# ---------------------------------------------------------------------------
# Task Planning & Config Builder
# ---------------------------------------------------------------------------
def build_replay_config(
    data_root: Path,
    trace_rel: Path,
    run_id: str,
    policy: str,
    policy_params: Dict[str, Any],
    pin_cpu: int,
    kappa: float = 5.0,
    c_ref: float = DEFAULT_C_REF,
    dim: int = 128,
    nprobe: int = 4,
    ef_search: int = 32,
) -> Dict[str, Any]:
    tr_path = trace_rel / "trace.bin"
    gt_path = trace_rel / "gt.bin"
    return {
        "pool": str(data_root / "pool.bin"),
        "init": str(data_root / "init.bin"),
        "centroids": str(data_root / "centroids.f32"),
        "R": 16,
        "dim": dim,
        "k": 10,
        "M": 16,
        "efConstruction": 100,
        "efSearch": ef_search,
        "nprobe": nprobe,
        "overfetch_start": 40,
        "mode": "replay",
        "delta_mode": "scan",
        "seed": 0,
        "pin_cpu": pin_cpu,
        "budget": {
            "kappa_RE": kappa,
            "c_ref_cpu_ms": c_ref,
        },
        "trace": str(tr_path),
        "gt": str(gt_path),
        "policy": policy,
        "policy_params": policy_params,
        "run_id": run_id,
    }


def plan_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    pred_csv = out_dir / "predictions.csv"
    kappas = [float(x) for x in args.kappas.split(",") if x.strip()]

    sift_traces = [f"{d}_{s}" for d in DRIFTS_DEFAULT for s in SEEDS_DEFAULT]
    tasks: List[Dict[str, Any]] = []

    if args.suite in ("sift25", "all"):
        for tname in sift_traces:
            tdir = data_root / "traces" / tname
            if not args.dry_run and not (tdir / "trace.bin").exists():
                continue
            for kappa in kappas:
                ktag = f"k{int(kappa) if kappa.is_integer() else kappa}"
                # 1. noop (shared across kappas; keep under k5 tag once)
                if kappa == kappas[0]:
                    tasks.append({
                        "suite": "sift25", "trace": tname, "arm": "noop", "kappa": kappa,
                        "data_root": data_root, "trace_dir": tdir,
                        "policy": "noop", "policy_params": {},
                    })
                # 2. past_qmass & bam_price
                tasks.append({
                    "suite": "sift25", "trace": tname, "arm": f"past_qmass_{ktag}", "kappa": kappa,
                    "data_root": data_root, "trace_dir": tdir,
                    "policy": "past_qmass",
                    "policy_params": {"action_mode": "merge", "qmass_window": 8, "spacing_blocks": 1},
                })
                tasks.append({
                    "suite": "sift25", "trace": tname, "arm": f"bam_price_{ktag}", "kappa": kappa,
                    "data_root": data_root, "trace_dir": tdir,
                    "policy": "bam_price",
                    "policy_params": {
                        "action_mode": "merge", "qmass_window": 8, "spacing_blocks": 1,
                        "c_scan": args.c_scan, "c_insert": args.c_insert,
                        "lambda0": 0.0, "eta": 1.0, "lambda_max": 1000.0,
                        "pred_path": str(pred_csv), "pred_model": "ewma_a03", "trace_name": tname,
                    },
                })
                # 3. 8 forecast_qmass predictors
                for model in PREDICTOR_MODELS:
                    tasks.append({
                        "suite": "sift25", "trace": tname, "arm": f"fcst_{model}_{ktag}", "kappa": kappa,
                        "data_root": data_root, "trace_dir": tdir,
                        "policy": "forecast_qmass",
                        "policy_params": {
                            "action_mode": "merge", "qmass_window": 8, "spacing_blocks": 1,
                            "pred_path": str(pred_csv), "pred_model": model, "trace_name": tname,
                        },
                    })
            # 4. 8 executed fixed cadences c=1..8 (budget-independent structural brackets)
            for c in FIXED_CADENCES:
                tasks.append({
                    "suite": "sift25", "trace": tname, "arm": f"fixed_c{c}_merge", "kappa": 5.0,
                    "data_root": data_root, "trace_dir": tdir,
                    "policy": "fixed",
                    "policy_params": {
                        "cadence_blocks": c, "phase": 1,
                        "shard_order": "round_robin", "action_mode": "merge",
                    },
                })

    if args.suite in ("phaselock", "all"):
        pl_root = Path(args.phaselock_root) if args.phaselock_root else data_root
        for tsw in PHASELOCK_TSWS:
            for seed in SEEDS_DEFAULT:
                # Support either `<pl_root>/traces/abrupt_tsw{tsw}_{seed}` or `<pl_root>/abrupt_tsw{tsw}/traces/abrupt_{seed}`
                cand1 = pl_root / "traces" / f"abrupt_tsw{tsw}_{seed}"
                cand2 = data_root / "traces" / f"abrupt_tsw{tsw}_{seed}"
                tdir = cand1 if (cand1 / "trace.bin").exists() else cand2
                tname = f"abrupt_tsw{tsw}_{seed}"
                if not args.dry_run and not (tdir / "trace.bin").exists():
                    continue
                for sp in (1, 2, 3):
                    tasks.append({
                        "suite": "phaselock", "trace": tname, "arm": f"past_sp{sp}_k3", "kappa": 3.0,
                        "data_root": data_root, "trace_dir": tdir,
                        "policy": "past_qmass",
                        "policy_params": {"action_mode": "merge", "qmass_window": 8, "spacing_blocks": sp},
                    })
                for c in (1, 2, 3, 4):
                    tasks.append({
                        "suite": "phaselock", "trace": tname, "arm": f"fixed_c{c}", "kappa": 5.0,
                        "data_root": data_root, "trace_dir": tdir,
                        "policy": "fixed",
                        "policy_params": {
                            "cadence_blocks": c, "phase": 1,
                            "shard_order": "round_robin", "action_mode": "merge",
                        },
                    })

    if args.suite in ("refresh", "all"):
        for tname in sift_traces:
            tdir = data_root / "traces" / tname
            if not args.dry_run and not (tdir / "trace.bin").exists():
                continue
            tasks.append({
                "suite": "refresh", "trace": tname, "arm": "past_qmass_refresh_k5", "kappa": 5.0,
                "data_root": data_root, "trace_dir": tdir,
                "policy": "past_qmass",
                "policy_params": {"action_mode": "refresh", "qmass_window": 8, "spacing_blocks": 1},
            })
            for c in (2, 3, 5):
                tasks.append({
                    "suite": "refresh", "trace": tname, "arm": f"fixed_c{c}_refresh", "kappa": 5.0,
                    "data_root": data_root, "trace_dir": tdir,
                    "policy": "fixed",
                    "policy_params": {
                        "cadence_blocks": c, "phase": 1,
                        "shard_order": "round_robin", "action_mode": "refresh",
                    },
                })
    return tasks


def _read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return 0.0


def _wait_hygiene_gate(max_load: float, timeout_s: int = 86400) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ld = _read_load1m()
        proc = subprocess.run(["pgrep", "-x", "qbam_replay"], capture_output=True, text=True)
        if ld <= max_load and not proc.stdout.strip():
            return
        time.sleep(5)
    raise RuntimeError(f"Hygiene gate timeout: load={_read_load1m():.2f} > {max_load}")


def execute_tasks(args: argparse.Namespace, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    run_dir = out_dir / "runs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f"qbam_replay binary not found: {binary}")

    completed = []
    flagged = []
    for idx, t in enumerate(tasks, 1):
        stem = f"{t['suite']}__{t['trace']}__{t['arm']}"
        cfg_path = cfg_dir / f"{stem}.config.json"
        out_jsonl = run_dir / f"{stem}.jsonl"
        log_path = run_dir / f"{stem}.log"

        cfg = build_replay_config(
            data_root=t["data_root"],
            trace_rel=t["trace_dir"],
            run_id=stem,
            policy=t["policy"],
            policy_params=t["policy_params"],
            pin_cpu=args.core,
            kappa=t["kappa"],
            c_ref=args.c_ref,
            dim=args.dim,
            nprobe=args.nprobe,
            ef_search=args.ef_search,
        )
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

        if args.skip_existing and out_jsonl.exists() and out_jsonl.stat().st_size > 0:
            audit = ah.check_contamination(out_jsonl, max_wall_cpu_ratio=1.18, max_load=args.max_load + 0.5)
            if audit["passed"]:
                completed.append({"stem": stem, "status": "skipped_clean", **audit})
                continue
            out_jsonl.unlink()  # remove dirty run before retry

        _wait_hygiene_gate(args.max_load)
        cmd = ["taskset", "-c", str(args.core), "nice", "-n", "19", str(binary), "--config", str(cfg_path), "--out", str(out_jsonl)]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
        log_path.write_text(res.stdout + "\n--- STDERR ---\n" + res.stderr, encoding="utf-8")
        if res.returncode != 0:
            raise RuntimeError(f"[{idx}/{len(tasks)}] {stem} failed rc={res.returncode}: {res.stderr[-300:]}")

        audit = ah.check_contamination(out_jsonl, max_wall_cpu_ratio=1.18, max_load=args.max_load + 0.5)
        entry = {"stem": stem, "wall_s": round(wall_s, 3), **audit}
        if audit["passed"]:
            completed.append(entry)
        else:
            flagged.append(entry)
        print(f"[{idx}/{len(tasks)}] {stem} cpu={audit['cpu_ms']}ms wall={wall_s:.1f}s passed={audit['passed']}", flush=True)

    summary = {"total_planned": len(tasks), "completed_clean": len(completed), "flagged": flagged}
    (out_dir / "exp2_exp3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="PVLDB Exp 2 & Exp 3 Predictors + Cadence Orchestrator")
    ap.add_argument("--data-root", default="/workspace/qbam/data/sift100k_u10_q2500_v2")
    ap.add_argument("--phaselock-root", default="")
    ap.add_argument("--binary", default="/workspace/qbam/build_rev4/qbam_replay")
    ap.add_argument("--out-dir", default="/workspace/pvldb/results/exp2_predictors_cadence")
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--suite", choices=["sift25", "phaselock", "refresh", "all"], default="sift25")
    ap.add_argument("--kappas", default="5.0", help="Comma-separated budget tiers, e.g. 2.0,3.0,5.0")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nprobe", type=int, default=4)
    ap.add_argument("--ef-search", type=int, default=32)
    ap.add_argument("--c-scan", type=float, default=DEFAULT_C_SCAN)
    ap.add_argument("--c-insert", type=float, default=DEFAULT_C_INSERT)
    ap.add_argument("--c-ref", type=float, default=DEFAULT_C_REF)
    ap.add_argument("--max-load", type=float, default=1.8)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--generate-predictions-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = plan_tasks(args)
    by_suite: Dict[str, int] = {}
    for t in tasks:
        by_suite[t["suite"]] = by_suite.get(t["suite"], 0) + 1

    if args.dry_run:
        sample = tasks[2] if len(tasks) > 2 else tasks[0]
        sample_cfg = build_replay_config(
            data_root=sample["data_root"],
            trace_rel=sample["trace_dir"],
            run_id=f"{sample['suite']}__{sample['trace']}__{sample['arm']}",
            policy=sample["policy"],
            policy_params=sample["policy_params"],
            pin_cpu=args.core,
            kappa=sample["kappa"],
            c_ref=args.c_ref,
            dim=args.dim,
            nprobe=args.nprobe,
            ef_search=args.ef_search,
        )
        print(json.dumps({
            "dry_run": True,
            "suite": args.suite,
            "total_tasks": len(tasks),
            "tasks_by_suite": by_suite,
            "predictors": list(PREDICTOR_MODELS),
            "fixed_cadences": list(FIXED_CADENCES),
            "sample_config": sample_cfg,
        }, indent=2))
        return 0

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    trace_names = sorted({t["trace"] for t in tasks if t["data_root"] == data_root})
    pred_csv, diag = generate_predictions_and_audit(data_root, trace_names, out_dir, nprobe=args.nprobe)
    print(f"Wrote {pred_csv}; causal clock-leak passed={diag['clock_leak_audit']['gbrt_causal']['passed']}, "
          f"leaky flagged={list(diag['clock_leak_audit']['gbrt_leaky']['flagged_features'])}")
    if not diag["clock_leak_audit"]["gbrt_causal"]["passed"]:
        raise RuntimeError(f"Causal predictor failed clock-leak gate: {diag['clock_leak_audit']['gbrt_causal']}")

    if args.generate_predictions_only:
        return 0

    summary = execute_tasks(args, tasks)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
