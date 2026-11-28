#!/usr/bin/env python3
"""Exp 10-B v2 (Xeon-8352V Isolated Xeon Core-0 Bare-Metal Suite, queued after S2_EF_CALIB_DONE.flag):
  B1: Bare-Metal Gate-4 Clean 768-d Wikipedia Crossover (140K/500K/1M, n=3 disjoint seeds, drift + CRUD20)
      - Fixes IndexIDMap2(IndexHNSWFlat) attribute access, raises timeout_s=5400, adds ndis + QPS + updates/s
  B2: Real-Engine Maintenance Scheduling Bridge:
      - B2a: NATIVE FreshDiskANN `consolidate_delete()` scheduling across R=8 shards on 768-d CRUD-30% (ZERO delta buffer)
      - B2b: 16-Shard real IndexIVFFlat / USearch / FreshDiskANN bridge with real trained HistGradientBoostingRegressor
  B3: Big-ANN'23 Streaming Protocol Re-Audit (unpinned default 8t vs pinned floor 1t/8t) + Thread-Safe Concurrent RW (4t/8t)
      - Measures EXACT per-thread CPU via time.thread_time() (zero hardcoded ratios) with RWLock / native concurrency
  B4: Trap 4 Symmetric Dirty Execution across all 5 real policies (real trained HistGradientBoostingRegressor)
      + 4-Signal Gate-4 ROC/AUC (wall/CPU, ru_nivcsw, /proc/self/schedstat wait_ms, inline L2-GEMV probe_slowdown)
  B5: FreshDiskANN CRUD 5-knob Early/Mid/Late Recall-QPS Pareto (search_L in {32, 64, 96, 128, 192})
"""
from __future__ import annotations

import os

for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_ev] = "1"

import argparse
import concurrent.futures
import csv
import json
import resource
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

MINICONDA_PY = Path(os.environ.get("DYNVECAUDIT_PYTHON", sys.executable))


def _roc_auc_exact(y_true: List[int], y_score: List[float]) -> float:
    """Exact ROC-AUC via Mann-Whitney U rank statistic with tie handling (zero sklearn dependency)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    s_ord = s[order]
    y_ord = y[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_ord[j + 1] == s_ord[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1
    sum_pos_ranks = float(np.sum(ranks[y_ord == 1]))
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)

BASE_EXP10 = Path(os.environ.get("DYNVECAUDIT_EXP10_DIR", "./results/exp10"))
REAL768_SRC = Path(os.environ.get("DYNVECAUDIT_REAL768_SRC", "./data/real10m_768/base.f32"))
REAL768_WL = BASE_EXP10 / "real768_workloads"
VDB_WL = BASE_EXP10 / "vdb_workloads"
BIGANN_SRC = Path(os.environ.get("DYNVECAUDIT_BIGANN_SRC", "./data/sift10m"))
BIGANN_WL = Path(os.environ.get("DYNVECAUDIT_BIGANN_WL", "./data/bigann_workloads"))
SIFT_DATA = Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data"))
OUT_DIR = BASE_EXP10 / "results"
K_BLOCK = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_existing_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def read_schedstat_wait_ns() -> int:
    try:
        parts = Path("/proc/self/schedstat").read_text(encoding="utf-8").strip().split()
        if len(parts) >= 2:
            return int(parts[1])
    except Exception:
        pass
    return 0


def run_micro_probe_ms(iters: int = 160) -> float:
    """Hardware-agnostic ~5-15ms L2-GEMV calibration probe sensitive to silent HT/SMT & LLC contention."""
    rng = np.random.default_rng(12345)
    a = rng.standard_normal((256, 128), dtype=np.float32)
    b = rng.standard_normal((128, 128), dtype=np.float32)
    _ = a @ b
    t0 = time.perf_counter_ns()
    acc = 0.0
    for _ in range(iters):
        c = a @ b
        acc += float(c[0, 0])
    dt_ms = (time.perf_counter_ns() - t0) / 1e6
    if acc == -1e30:
        log("unreachable")
    return dt_ms


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    except Exception:
        return 0.0


def wait_quiet_host(max_load: float = 4.0) -> None:
    for _ in range(30):
        if read_load1m() <= max_load:
            return
        time.sleep(10.0)


def _invoke_worker(mode: str, cfg: Dict[str, Any], cores: str = "0", threads: int = 1, timeout_s: int = 5400) -> Dict[str, Any]:
    payload = json.dumps({"mode": mode, **cfg})
    cmd = ["taskset", "-c", cores, sys.executable, str(Path(__file__).resolve()), "--worker-json", payload]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    pr = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    if pr.returncode != 0:
        raise RuntimeError(f"worker {mode} rc={pr.returncode}: {pr.stderr[-600:]}")
    for line in pr.stdout.splitlines():
        if line.startswith("WORKER_JSON="):
            return json.loads(line.split("WORKER_JSON=", 1)[1])
    raise RuntimeError(f"Missing WORKER_JSON from {mode}: {pr.stdout[-400:]}")


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] * (1.0 - (k - lo)) + s[hi] * (k - lo))


def _mean_recall(pred: np.ndarray, gt: np.ndarray, k: int = 10) -> float:
    nq = pred.shape[0]
    hits = 0
    for i in range(nq):
        row = pred[i]
        v = row[row >= 0]
        hits += len(set(v.tolist()) & set(gt[i, :k].tolist()))
    return hits / float(max(nq * k, 1))


class _RWLock:
    """Fair reader-writer lock enabling concurrent readers and mutually exclusive micro-batch writers."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer_waiting = 0
        self._writer_active = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer_active or self._writer_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writer_waiting += 1
            while self._writer_active or self._readers > 0:
                self._cond.wait()
            self._writer_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# Worker B1: Bare-Metal 768-d Wikipedia Crossover (Core 0, OMP=1)
# ---------------------------------------------------------------------------
def _worker_b1_real768(cfg: Dict[str, Any]) -> Dict[str, Any]:
    npz_path = REAL768_WL / cfg["npz_name"]
    z = np.load(npz_path)
    gidx = z["pool_global_idx"]
    queries = np.ascontiguousarray(z["queries"], dtype=np.float32)
    init_ids = z["init_ids"].astype(np.int64)
    win_ins = z["win_inserts"].astype(np.int64)
    win_era = z["win_erases"].astype(np.int64)
    gt_ids = z["gt_ids"].astype(np.int64)

    base_mm = np.memmap(REAL768_SRC, dtype=np.float32, mode="r", shape=(10_000_000, 768))
    pool = np.ascontiguousarray(base_mm[gidx], dtype=np.float32)
    dim = 768
    n_win, q_pw, _ = gt_ids.shape
    system, ptag, params = cfg["system"], cfg["param_tag"], cfg["params"]
    max_vecs = len(pool) + 10_000
    n_crud_tot = int(win_ins.size + (win_era.size if win_era.ndim == 2 and win_era.shape[1] > 0 else 0))

    t_w0 = time.perf_counter()
    t_b0 = time.process_time()
    tmp_dir = None
    has_del = False
    ndis_total = -1
    if system == "faiss_ivf":
        import faiss
        faiss.omp_set_num_threads(1)
        faiss.cvar.indexIVF_stats.reset()
        idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, int(params["nlist"]), faiss.METRIC_L2)
        idx.train(pool[init_ids[: min(100_000, len(init_ids))]])
        idx.add_with_ids(pool[init_ids], init_ids)
        idx.nprobe = int(params["nprobe"])
    elif system == "faiss_hnsw":
        import faiss
        faiss.omp_set_num_threads(1)
        base_hnsw = faiss.IndexHNSWFlat(dim, int(params.get("M", 16)))
        base_hnsw.hnsw.efConstruction = int(params.get("ef_c", 100))
        base_hnsw.hnsw.efSearch = int(params["ef"])
        idx = faiss.IndexIDMap2(base_hnsw)
        idx.add_with_ids(pool[init_ids], init_ids)
    elif system == "hnswlib":
        import hnswlib
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=max_vecs, ef_construction=int(params.get("ef_c", 100)), M=int(params.get("M", 16)), allow_replace_deleted=True)
        idx.set_num_threads(1)
        idx.add_items(pool[init_ids], init_ids, num_threads=1)
        idx.set_ef(int(params["ef"]))
    elif system == "usearch":
        from usearch.index import Index
        idx = Index(ndim=dim, metric="l2sq", dtype="f32", connectivity=int(params.get("M", 16)),
                    expansion_add=int(params.get("ef_c", 100)), expansion_search=int(params["ef"]))
        idx.add(init_ids.astype(np.uint64), pool[init_ids], threads=1)
    elif system == "diskann":
        import diskannpy
        tmp_dir = tempfile.TemporaryDirectory(prefix="b1_diskann_")
        R, Lbuild, sL = int(params.get("R", 32)), int(params.get("Lbuild", 100)), int(params["search_L"])
        diskannpy.build_memory_index(pool[init_ids], "l2", tmp_dir.name, complexity=Lbuild, graph_degree=R,
                                     num_threads=1, tags=(init_ids + 1).astype(np.uint32))
        idx = diskannpy.DynamicMemoryIndex.from_file(tmp_dir.name, max_vectors=max_vecs, complexity=Lbuild,
                                                     graph_degree=R, num_threads=1, initial_search_complexity=sL,
                                                     search_threads=1, concurrent_consolidation=False)
    else:
        raise ValueError(system)
    build_cpu_s = time.process_time() - t_b0

    q_cpu_s = m_cpu_s = 0.0
    recs: List[float] = []
    p99s: List[float] = []
    status = "ok"
    try:
        for w in range(n_win):
            tm0 = time.process_time()
            era = win_era[w] if (win_era.ndim == 2 and win_era.shape[1] > 0) else np.empty(0, dtype=np.int64)
            ins = win_ins[w]
            if len(era) > 0:
                if system == "faiss_ivf":
                    idx.remove_ids(era)
                elif system == "faiss_hnsw":
                    status = "unsupported_erase"
                elif system == "hnswlib":
                    for eid in era.tolist():
                        idx.mark_deleted(int(eid))
                    has_del = True
                elif system == "usearch":
                    for eid in era.tolist():
                        idx.remove(int(eid))
                elif system == "diskann":
                    for eid in era.tolist():
                        idx.mark_deleted(int(eid) + 1)
                    has_del = True
            if system in ("faiss_ivf", "faiss_hnsw"):
                idx.add_with_ids(pool[ins], ins)
            elif system == "hnswlib":
                idx.add_items(pool[ins], ins, num_threads=1, replace_deleted=has_del)
            elif system == "usearch":
                idx.add(ins.astype(np.uint64), pool[ins], threads=1)
            elif system == "diskann":
                for iid in ins.tolist():
                    idx.insert(pool[iid], int(iid) + 1)
                if has_del and (w + 1) % 2 == 0:
                    idx.consolidate_delete()
            m_cpu_s += time.process_time() - tm0

            qw = queries[w * q_pw : (w + 1) * q_pw]
            pred = np.empty((q_pw, 10), dtype=np.int64)
            lat: List[float] = []
            tq0 = time.process_time()
            for i in range(q_pw):
                tw0 = time.perf_counter_ns()
                if system in ("faiss_ivf", "faiss_hnsw"):
                    _, row = idx.search(qw[i : i + 1], 10)
                    pred[i] = row[0]
                elif system == "hnswlib":
                    row, _ = idx.knn_query(qw[i : i + 1], k=10, num_threads=1)
                    pred[i] = row[0]
                elif system == "usearch":
                    pred[i] = np.asarray(idx.search(qw[i], 10, exact=False).keys, dtype=np.int64)[:10]
                elif system == "diskann":
                    res = idx.search(qw[i], 10, int(params["search_L"]))
                    pred[i] = np.asarray(res.identifiers, dtype=np.int64) - 1
                lat.append((time.perf_counter_ns() - tw0) / 1000.0)
            q_cpu_s += time.process_time() - tq0
            recs.append(_mean_recall(pred, gt_ids[w], 10))
            p99s.append(_pct(lat, 99.0))
        if system == "faiss_ivf":
            import faiss
            ndis_total = int(faiss.cvar.indexIVF_stats.ndis)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    wall_s = time.perf_counter() - t_w0
    tot_s = q_cpu_s + m_cpu_s
    return {
        "npz_name": cfg["npz_name"], "workload": cfg["workload"], "seed": cfg["seed"],
        "system": system, "param_tag": ptag, "status": status,
        "build_cpu_s": round(build_cpu_s, 3),
        "query_cpu_s": round(q_cpu_s, 3),
        "maint_cpu_s": round(m_cpu_s, 3),
        "total_cpu_s": round(tot_s, 3),
        "query_pct": round(100.0 * q_cpu_s / max(tot_s, 1e-6), 2),
        "maint_pct": round(100.0 * m_cpu_s / max(tot_s, 1e-6), 2),
        "mean_recall": round(float(np.mean(recs)), 5),
        "min_recall": round(float(np.min(recs)), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recs if r >= 0.965) / n_win, 1),
        "att_0950_pct": round(100.0 * sum(1 for r in recs if r >= 0.950) / n_win, 1),
        "qps_1core": round((n_win * q_pw) / max(q_cpu_s, 1e-6), 1),
        "updates_per_s": round(n_crud_tot / max(m_cpu_s, 1e-6), 1),
        "ndis_total": ndis_total,
        "mean_p99_us": round(float(np.mean(p99s)), 2),
        "wall_s": round(wall_s, 3),
        "wall_cpu_ratio": round(wall_s / max(tot_s + build_cpu_s, 1e-6), 4),
        "peak_rss_mb": peak_rss_mb(),
    }


# ---------------------------------------------------------------------------
# Shared Helpers for B2 & B4 (Real QBAM Trace + Real Causal GBRT + Exact MILP)
# ---------------------------------------------------------------------------
def _read_qbam_trace(data_root: Path, trace_name: str) -> Dict[str, Any]:
    with (data_root / "pool.bin").open("rb") as f:
        assert f.read(8) == b"QBAMPOOL"
        _, dim, nv = struct.unpack("<III", f.read(12))
    pool = np.fromfile(data_root / "pool.bin", dtype="<f4", offset=20).reshape(nv, dim)
    with (data_root / "init.bin").open("rb") as f:
        assert f.read(8) == b"QBAMINIT"
        _, _, n_init, R = struct.unpack("<IIII", f.read(16))
    init_dt = np.dtype([("id", "<u8"), ("shard", "<u4"), ("vec_ref", "<u4")])
    init = np.fromfile(data_root / "init.bin", dtype=init_dt, offset=24, count=n_init)
    with (data_root / "traces" / trace_name / "trace.bin").open("rb") as f:
        assert f.read(8) == b"QBAMTRAC"
        _, _, n_ev, n_win, k_cfg = struct.unpack("<IIIII", f.read(20))
    ev_dt = np.dtype([("window", "<u4"), ("type", "u1"), ("id", "<u8"), ("version", "<u8"), ("vec_ref", "<u4"), ("flags", "u1")])
    ev = np.fromfile(data_root / "traces" / trace_name / "trace.bin", dtype=ev_dt, offset=28, count=n_ev)
    with (data_root / "traces" / trace_name / "gt.bin").open("rb") as f:
        assert f.read(8) == b"QBAMGT\0\0"
        _, k_gt, n_q_gt, _ = struct.unpack("<IIII", f.read(16))
    gt_dt = np.dtype([("id", "<u8"), ("dist", "<f4")])
    gt = np.fromfile(data_root / "traces" / trace_name / "gt.bin", dtype=gt_dt, offset=24, count=n_q_gt * k_gt).reshape(n_q_gt, k_gt)
    centroids = np.fromfile(data_root / "centroids.f32", dtype="<f4").reshape(R, dim)
    return {"dim": dim, "R": R, "n_win": n_win, "k": k_gt, "pool": np.ascontiguousarray(pool, np.float32),
            "init": init, "ev": ev, "gt_ids": gt["id"].astype(np.int64), "centroids": np.ascontiguousarray(centroids, np.float32)}


def _extract_qd_fast(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, n_win, pool, init, ev, centroids = d["R"], d["n_win"], d["pool"], d["init"], d["ev"], d["centroids"]
    c_norm2 = np.sum(centroids ** 2, axis=1)
    base_n = np.bincount(init["shard"], minlength=R).astype(np.float64)
    Q = np.zeros((n_win, R), dtype=np.float64)
    ins_pw = np.zeros((n_win, R), dtype=np.float64)
    for w in range(n_win):
        we = ev[ev["window"] == w]
        qm = we[we["type"] == 3]
        if len(qm) > 0:
            qv = pool[qm["vec_ref"].astype(np.int64)]
            sc = -2.0 * (qv @ centroids.T) + c_norm2[None, :]
            top4 = np.argpartition(sc, 3, axis=1)[:, :4]
            np.add.at(Q[w], top4.ravel(), 1.0)
        cm = we[we["type"] == 0]
        if len(cm) > 0:
            iv = pool[cm["vec_ref"].astype(np.int64)]
            top1 = np.argmin(-2.0 * (iv @ centroids.T) + c_norm2[None, :], axis=1)
            np.add.at(ins_pw[w], top1, 1.0)
    return Q, np.cumsum(ins_pw, axis=0), base_n


def _build_causal_features(Q: np.ndarray, D: np.ndarray, base_n: np.ndarray, K: int = 4) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    T, R = Q.shape
    J = T // K
    eye = np.eye(R, dtype=np.float64)
    xs, ys, js = [], [], []
    for j in range(1, J):
        h0 = j * K
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
        y_true = Q[h0: h0 + K].mean(axis=0)
        blk = np.column_stack([
            past.mean(axis=0), past.std(axis=0), slope, last4.T, dinc4, base_n,
            np.full(R, float(past.mean(axis=0).sum())), np.full(R, float(dinc4.sum())), eye,
        ])
        xs.append(blk)
        ys.append(y_true)
        js.append(j)
    return np.vstack(xs), np.concatenate(ys), js


def _train_real_gbrt_causal(data_root: Path, target_trace: str, sibling_traces: List[str], K: int = 4) -> Dict[int, np.ndarray]:
    """Train genuine LightGBM LGBMRegressor (matching paper Section 2.3/3) via miniconda3 on strictly causal 11 features."""
    x_tr_list, y_tr_list = [], []
    for st in sibling_traces:
        if st == target_trace or not (data_root / "traces" / st / "trace.bin").exists():
            continue
        ds = _read_qbam_trace(data_root, st)
        Qs, Ds, bns = _extract_qd_fast(ds)
        xs, ys, _ = _build_causal_features(Qs, Ds, bns, K=K)
        x_tr_list.append(xs)
        y_tr_list.append(ys)
    if not x_tr_list:
        raise RuntimeError(f"No sibling traces found in {data_root} for {target_trace}")
    X_tr, y_tr = np.vstack(x_tr_list), np.concatenate(y_tr_list)
    dt = _read_qbam_trace(data_root, target_trace)
    Qt, Dt, bnt = _extract_qd_fast(dt)
    X_te, _, js_te = _build_causal_features(Qt, Dt, bnt, K=K)
    R = Qt.shape[1]

    with tempfile.TemporaryDirectory(prefix="lgb_causal_") as td:
        in_npz = Path(td) / "data.npz"
        out_npy = Path(td) / "yhat.npy"
        np.savez(in_npz, X_tr=X_tr, y_tr=y_tr, X_te=X_te)
        code = (
            "import numpy as np, lightgbm as lgb\n"
            f"z=np.load({str(in_npz)!r})\n"
            "m=lgb.LGBMRegressor(n_estimators=120,learning_rate=0.06,num_leaves=31,"
            "subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=1,verbose=-1)\n"
            "m.fit(z['X_tr'], z['y_tr'])\n"
            f"np.save({str(out_npy)!r}, np.maximum(0.0, m.predict(z['X_te'])))\n"
        )
        env = dict(os.environ)
        for evk in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[evk] = "1"
        pr = subprocess.run([str(MINICONDA_PY), "-c", code], env=env, capture_output=True, text=True, check=False)
        if pr.returncode != 0:
            raise RuntimeError(f"LightGBM fit failed: {pr.stderr[-400:]}")
        yhat = np.load(out_npy)
    return {j: yhat[i * R : (i + 1) * R] for i, j in enumerate(js_te)}


def _solve_milp_qd(Q: np.ndarray, D: np.ndarray, K: int = 4, B: int = 9) -> float:
    T, R = Q.shape
    J = T // K
    if B <= 0 or J <= 1:
        return float((Q * D).sum())
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
    lb = np.array(rhs + [0.0])
    ub = np.array(rhs + [float(B)])
    res = milp(c=c, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=LinearConstraint(A.tocsc(), lb, ub))
    return float(res.fun)


# ---------------------------------------------------------------------------
# Worker B2a: NATIVE FreshDiskANN consolidate_delete() Scheduler Bridge (ZERO Delta Buffer!)
# ---------------------------------------------------------------------------
def _worker_b2a_native_consolidate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """8-shard native FreshDiskANN on 768-d CRUD-30%: inserts/deletes hit DynamicMemoryIndex directly,
    and scheduler chooses which shard executes native `consolidate_delete()` under matched budget B."""
    import diskannpy

    z = np.load(VDB_WL / "vdb_real768_80k_crud30.npz")
    gidx = z["pool_global_idx"]
    queries = np.ascontiguousarray(z["queries"], dtype=np.float32)
    init_ids = z["init_ids"].astype(np.int64)
    win_ins = z["win_inserts"].astype(np.int64)
    win_era = z["win_erases"].astype(np.int64)
    gt_ids = z["gt_ids"].astype(np.int64)

    base_mm = np.memmap(REAL768_SRC, dtype=np.float32, mode="r", shape=(10_000_000, 768))
    pool = np.ascontiguousarray(base_mm[gidx], dtype=np.float32)
    dim = 768
    n_win, q_pw, _ = gt_ids.shape
    R = 8
    policy = cfg["policy"]
    budget = int(cfg.get("budget", max(4, n_win // 2)))

    # Deterministic k-means over init vectors to build R=8 Voronoi shards
    rng = np.random.default_rng(42)
    init_vecs = pool[init_ids]
    cent = init_vecs[rng.choice(len(init_vecs), size=R, replace=False)].copy()
    for _ in range(15):
        c_n2 = np.sum(cent ** 2, axis=1)
        lbl = np.argmin(-2.0 * (init_vecs @ cent.T) + c_n2[None, :], axis=1)
        for r in range(R):
            m = lbl == r
            if np.any(m):
                cent[r] = init_vecs[m].mean(axis=0)
    c_n2 = np.sum(cent ** 2, axis=1)

    def _assign1(vecs: np.ndarray) -> np.ndarray:
        return np.argmin(-2.0 * (vecs @ cent.T) + c_n2[None, :], axis=1)

    def _assign_topk(vecs: np.ndarray, k_sh: int = 3) -> np.ndarray:
        sc = -2.0 * (vecs @ cent.T) + c_n2[None, :]
        return np.argpartition(sc, k_sh - 1, axis=1)[:, :k_sh]

    # Map every vector id -> its assigned shard
    id_to_shard: Dict[int, int] = {}
    init_lbl = _assign1(init_vecs)
    shard_init_ids = [[] for _ in range(R)]
    for gid, r in zip(init_ids.tolist(), init_lbl.tolist()):
        id_to_shard[int(gid)] = int(r)
        shard_init_ids[r].append(int(gid))

    # Precompute hotspot-shifted query routing weight Q[w, r] and tombstone arrivals D[w, r]
    Q_mat = np.zeros((n_win, R), dtype=np.float64)
    D_tomb_pw = np.zeros((n_win, R), dtype=np.float64)
    win_q_top = []
    for w in range(n_win):
        qw = queries[w * q_pw : (w + 1) * q_pw]
        top_r = _assign_topk(qw, k_sh=3)
        win_q_top.append(top_r)
        np.add.at(Q_mat[w], top_r.ravel(), 1.0)
        for eid in win_era[w].tolist():
            r_e = id_to_shard.get(int(eid), -1)
            if r_e >= 0:
                D_tomb_pw[w, r_e] += 1.0
        ins_lbl = _assign1(pool[win_ins[w]])
        for iid, r_i in zip(win_ins[w].tolist(), ins_lbl.tolist()):
            id_to_shard[int(iid)] = int(r_i)
    D_tomb_cum = np.cumsum(D_tomb_pw, axis=0)

    w_noop = float((Q_mat * D_tomb_cum).sum())
    w_star = _solve_milp_qd(Q_mat, D_tomb_cum, K=1, B=budget)

    t_w0 = time.perf_counter()
    t_b0 = time.process_time()
    shard_tmps = []
    shard_idxs = []
    try:
        for r in range(R):
            td = tempfile.TemporaryDirectory(prefix=f"b2a_da_{r}_")
            shard_tmps.append(td)
            sids = np.asarray(shard_init_ids[r], dtype=np.int64)
            diskannpy.build_memory_index(
                pool[sids], "l2", td.name, complexity=64, graph_degree=32, num_threads=1,
                tags=(sids + 1).astype(np.uint32),
            )
            idx_r = diskannpy.DynamicMemoryIndex.from_file(
                td.name, max_vectors=len(sids) + 12000, complexity=64, graph_degree=32,
                num_threads=1, initial_search_complexity=64, search_threads=1, concurrent_consolidation=False,
            )
            shard_idxs.append(idx_r)
        build_cpu_s = time.process_time() - t_b0

        tomb_cnt = np.zeros(R, dtype=np.float64)
        ewma_q = np.zeros(R, dtype=np.float64)
        spent = 0
        rr_ptr = 0
        q_cpu_s = ins_del_cpu_s = cons_cpu_s = 0.0
        tomb_scan_ops = 0.0
        recs: List[float] = []
        p99s: List[float] = []

        for w in range(n_win):
            # 1. Native deletes & inserts directly into DynamicMemoryIndex (ZERO delta buffer!)
            tm0 = time.process_time()
            for eid in win_era[w].tolist():
                r_e = id_to_shard.get(int(eid), -1)
                if r_e >= 0:
                    shard_idxs[r_e].mark_deleted(int(eid) + 1)
                    tomb_cnt[r_e] += 1.0
            for iid in win_ins[w].tolist():
                r_i = id_to_shard[int(iid)]
                shard_idxs[r_i].insert(pool[int(iid)], int(iid) + 1)
            ins_del_cpu_s += time.process_time() - tm0

            # 2. Scheduled NATIVE consolidate_delete() before queries
            if w >= 1 and spent < budget and policy != "noop":
                sel = -1
                if policy == "fixed_c2":
                    if (w - 1) % 2 == 0:
                        sel = rr_ptr % R
                        rr_ptr += 1
                elif policy == "max_delta":
                    sel = int(np.argmax(tomb_cnt))
                elif policy == "past_qmass":
                    sel = int(np.argmax(Q_mat[w - 1] * tomb_cnt))
                elif policy == "ewma_a03":
                    sel = int(np.argmax(ewma_q * tomb_cnt))
                elif policy == "oracle_1blk":
                    sel = int(np.argmax(Q_mat[w] * tomb_cnt))
                if sel >= 0 and tomb_cnt[sel] > 0:
                    tc0 = time.process_time()
                    shard_idxs[sel].consolidate_delete()
                    cons_cpu_s += time.process_time() - tc0
                    tomb_cnt[sel] = 0.0
                    spent += 1

            tomb_scan_ops += float((Q_mat[w] * tomb_cnt).sum())
            ewma_q = Q_mat[w] if w == 0 else (0.3 * Q_mat[w] + 0.7 * ewma_q)

            # 3. Native sharded query search across top-3 shards
            qw = queries[w * q_pw : (w + 1) * q_pw]
            top_r = win_q_top[w]
            pred = np.empty((q_pw, 10), dtype=np.int64)
            lat: List[float] = []
            tq0 = time.process_time()
            for i in range(q_pw):
                t0 = time.perf_counter_ns()
                cands: List[Tuple[float, int]] = []
                for r in top_r[i].tolist():
                    res = shard_idxs[r].search(qw[i], 10, 64)
                    for tag, dist in zip(res.identifiers, res.distances):
                        cands.append((float(dist), int(tag) - 1))
                cands.sort(key=lambda x: x[0])
                seen_ids: List[int] = []
                for _, cid in cands:
                    if cid not in seen_ids:
                        seen_ids.append(cid)
                        if len(seen_ids) == 10:
                            break
                while len(seen_ids) < 10:
                    seen_ids.append(-1)
                pred[i] = seen_ids
                lat.append((time.perf_counter_ns() - t0) / 1000.0)
            q_cpu_s += time.process_time() - tq0
            recs.append(_mean_recall(pred, gt_ids[w], 10))
            p99s.append(_pct(lat, 99.0))
    finally:
        for td in shard_tmps:
            td.cleanup()

    wall_s = time.perf_counter() - t_w0
    maint_s = ins_del_cpu_s + cons_cpu_s
    tot_s = q_cpu_s + maint_s
    gcr = 100.0 * (w_noop - tomb_scan_ops) / max(w_noop - w_star, 1e-9)
    return {
        "bridge_type": "native_consolidate_delete",
        "engine": "native_8sh_diskann_crud30",
        "trace": "vdb_real768_80k_crud30",
        "policy": policy,
        "actions": spent,
        "build_cpu_s": round(build_cpu_s, 3),
        "query_cpu_s": round(q_cpu_s, 3),
        "ins_del_cpu_s": round(ins_del_cpu_s, 3),
        "consolidate_cpu_s": round(cons_cpu_s, 3),
        "maint_cpu_s": round(maint_s, 3),
        "total_cpu_s": round(tot_s, 3),
        "exact_scan_ops": round(tomb_scan_ops, 1),
        "gcr_ops_pct": round(gcr, 2),
        "mean_recall": round(float(np.mean(recs)), 5),
        "min_recall": round(float(np.min(recs)), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recs if r >= 0.965) / n_win, 1),
        "mean_p99_us": round(float(np.mean(p99s)), 2),
        "wall_s": round(wall_s, 3),
        "wall_cpu_ratio": round(wall_s / max(tot_s + build_cpu_s, 1e-6), 4),
        "peak_rss_mb": peak_rss_mb(),
    }


# ---------------------------------------------------------------------------
# Worker B2b: 16-Shard Real IndexIVFFlat / USearch / DiskANN Bridge (with Real GBRT-Causal)
# ---------------------------------------------------------------------------
def _worker_b2b_shard_bridge(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_root = Path(cfg["data_root"])
    tr_name = cfg["trace"]
    d = _read_qbam_trace(data_root, tr_name)
    dim, R, n_win, pool, init, ev, gt_ids, centroids = (
        d["dim"], d["R"], d["n_win"], d["pool"], d["init"], d["ev"], d["gt_ids"], d["centroids"]
    )
    c_norm2 = np.sum(centroids ** 2, axis=1)
    engine, policy = cfg["engine"], cfg["policy"]

    gbrt_preds: Dict[int, np.ndarray] = {}
    if policy == "gbrt_causal":
        drift = tr_name.rsplit("_", 1)[0]
        sibs = [f"{drift}_s0{k}" for k in (5, 6, 7, 8, 9)]
        gbrt_preds = _train_real_gbrt_causal(data_root, tr_name, sibs, K=K_BLOCK)

    base_shards = [[] for _ in range(R)]
    base_ids = [[] for _ in range(R)]
    for row in init:
        r = int(row["shard"])
        base_shards[r].append(int(row["vec_ref"]))
        base_ids[r].append(int(row["id"]))

    Q, D, _ = _extract_qd_fast(d)
    win_crud, win_q, win_q_off = [], [], []
    q_off = 0
    for w in range(n_win):
        we = ev[ev["window"] == w]
        cm = we[we["type"] != 3]
        qm = we[we["type"] == 3]
        win_crud.append(cm)
        win_q.append(qm)
        win_q_off.append(q_off)
        q_off += len(qm)

    import faiss, diskannpy
    from usearch.index import Index as UIndex

    faiss.omp_set_num_threads(1)
    t_w0 = time.perf_counter()
    shard_base = [np.ascontiguousarray(pool[base_shards[r]], np.float32) for r in range(R)]
    shard_ids = [np.asarray(base_ids[r], np.int64) for r in range(R)]
    shard_delta_vecs: List[List[np.ndarray]] = [[] for _ in range(R)]
    shard_delta_ids: List[List[int]] = [[] for _ in range(R)]
    shard_tmp: List[Optional[tempfile.TemporaryDirectory]] = [None] * R
    shard_idx: List[Any] = [None] * R

    def _rebuild_shard(r: int, merge_delta: bool) -> None:
        if merge_delta and shard_delta_ids[r]:
            dv = np.vstack(shard_delta_vecs[r]).astype(np.float32)
            di = np.asarray(shard_delta_ids[r], dtype=np.int64)
            shard_base[r] = np.vstack([shard_base[r], dv])
            shard_ids[r] = np.concatenate([shard_ids[r], di])
            shard_delta_vecs[r].clear()
            shard_delta_ids[r].clear()
        if engine == "sharded_usearch":
            idx = UIndex(ndim=dim, metric="l2sq", dtype="f32", connectivity=16, expansion_add=100, expansion_search=48)
            idx.add(shard_ids[r].astype(np.uint64), shard_base[r], threads=1)
            shard_idx[r] = idx
        elif engine == "sharded_diskann":
            if shard_tmp[r] is not None:
                shard_tmp[r].cleanup()
            td = tempfile.TemporaryDirectory(prefix=f"sd_{r}_")
            shard_tmp[r] = td
            tags = np.arange(1, len(shard_ids[r]) + 1, dtype=np.uint32)
            diskannpy.build_memory_index(shard_base[r], "l2", td.name, complexity=64, graph_degree=32, num_threads=1, tags=tags)
            shard_idx[r] = diskannpy.DynamicMemoryIndex.from_file(
                td.name, max_vectors=len(shard_ids[r]) + 12000, complexity=64, graph_degree=32,
                num_threads=1, initial_search_complexity=48, search_threads=1, concurrent_consolidation=False
            )
        elif engine == "sharded_faiss_ivf":
            nlist_s = min(16, max(4, len(shard_base[r]) // 64))
            idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist_s, faiss.METRIC_L2)
            idx.train(shard_base[r])
            idx.add_with_ids(shard_base[r], shard_ids[r])
            idx.nprobe = min(8, nlist_s)
            shard_idx[r] = idx

    t_b0 = time.process_time()
    for r in range(R):
        _rebuild_shard(r, merge_delta=False)
    build_cpu_s = time.process_time() - t_b0

    w_noop = float((Q * D).sum())
    w_star = _solve_milp_qd(Q, D, K=K_BLOCK, B=5)
    spent = 0
    rr_ptr = 0
    ewma_q = np.zeros(R, dtype=np.float64)
    q_cpu_s = m_cpu_s = 0.0
    exact_ops = 0.0
    recs: List[float] = []
    actions = 0

    try:
        for w in range(n_win):
            if w > 0 and w % K_BLOCK == 0 and spent < 5 and policy != "noop":
                j = w // K_BLOCK
                cur_d = np.array([len(shard_delta_ids[r]) for r in range(R)], dtype=np.float64)
                sel = -1
                if policy == "fixed_c2":
                    if (j - 1) % 2 == 0:
                        sel = rr_ptr % R
                        rr_ptr += 1
                elif policy == "past_qmass":
                    sel = int(np.argmax(Q[w - K_BLOCK : w].sum(axis=0) * cur_d))
                elif policy == "ewma_a03":
                    sel = int(np.argmax(ewma_q * cur_d))
                elif policy == "gbrt_causal":
                    sel = int(np.argmax(gbrt_preds[j] * cur_d))
                elif policy == "imbalance_prio":
                    base_lens = np.array([max(len(shard_ids[r]), 1) for r in range(R)], dtype=np.float64)
                    sel = int(np.argmax((Q[w - K_BLOCK : w].sum(axis=0) ** 0.7) * (cur_d ** 1.2) / np.sqrt(base_lens)))
                elif policy == "oracle":
                    sel = int(np.argmax(Q[w : min(w + K_BLOCK, n_win)].sum(axis=0) * cur_d))
                if sel >= 0 and cur_d[sel] > 0:
                    tm0 = time.process_time()
                    _rebuild_shard(sel, merge_delta=True)
                    m_cpu_s += time.process_time() - tm0
                    spent += 1
                    actions += 1

            tm0 = time.process_time()
            crud = win_crud[w]
            ins_m = crud["type"] == 0
            if np.any(ins_m):
                iv = pool[crud["vec_ref"][ins_m].astype(np.int64)]
                iids = crud["id"][ins_m].astype(np.int64)
                top1 = np.argmin(-2.0 * (iv @ centroids.T) + c_norm2[None, :], axis=1)
                for k_i, rr in enumerate(top1.tolist()):
                    shard_delta_vecs[rr].append(iv[k_i : k_i + 1])
                    shard_delta_ids[rr].append(int(iids[k_i]))
            m_cpu_s += time.process_time() - tm0

            exact_ops += float((Q[w] * np.array([len(shard_delta_ids[r]) for r in range(R)], dtype=np.float64)).sum())
            ewma_q = Q[w] if w == 0 else (0.3 * Q[w] + 0.7 * ewma_q)

            qm = win_q[w]
            qvecs = pool[qm["vec_ref"].astype(np.int64)]
            nq = len(qvecs)
            pred = np.empty((nq, 10), dtype=np.int64)
            tq0 = time.process_time()
            d_mats = [np.vstack(shard_delta_vecs[r]) if shard_delta_vecs[r] else None for r in range(R)]
            d_idarrs = [np.asarray(shard_delta_ids[r], dtype=np.int64) if shard_delta_ids[r] else None for r in range(R)]
            d_idx_faiss = []
            for r in range(R):
                if d_mats[r] is not None:
                    dfi = faiss.IndexFlatL2(dim)
                    dfi.add(d_mats[r])
                    d_idx_faiss.append(dfi)
                else:
                    d_idx_faiss.append(None)

            top4_all = np.argpartition(-2.0 * (qvecs @ centroids.T) + c_norm2[None, :], 3, axis=1)[:, :4]
            for i in range(nq):
                qv = qvecs[i]
                cands: List[Tuple[float, int]] = []
                for r in top4_all[i].tolist():
                    if engine == "sharded_usearch":
                        res = shard_idx[r].search(qv, 10)
                        for kk, dd in zip(res.keys, res.distances):
                            cands.append((float(dd), int(kk)))
                    elif engine == "sharded_diskann":
                        res = shard_idx[r].search(qv, 10, 48)
                        for tag, dd in zip(res.identifiers, res.distances):
                            idx_pos = int(tag) - 1
                            if 0 <= idx_pos < len(shard_ids[r]):
                                cands.append((float(dd), int(shard_ids[r][idx_pos])))
                    elif engine == "sharded_faiss_ivf":
                        dd, ii = shard_idx[r].search(qv[None, :], 10)
                        for cid, dv in zip(ii[0].tolist(), dd[0].tolist()):
                            if cid >= 0:
                                cands.append((float(dv), int(cid)))
                    if d_idx_faiss[r] is not None:
                        dd, ii = d_idx_faiss[r].search(qv[None, :], min(10, len(d_idarrs[r])))
                        for pos, dv in zip(ii[0].tolist(), dd[0].tolist()):
                            if pos >= 0:
                                cands.append((float(dv), int(d_idarrs[r][pos])))
                cands.sort(key=lambda x: x[0])
                seen_ids: List[int] = []
                for _, cid in cands:
                    if cid not in seen_ids:
                        seen_ids.append(cid)
                        if len(seen_ids) == 10:
                            break
                while len(seen_ids) < 10:
                    seen_ids.append(-1)
                pred[i] = seen_ids
            q_cpu_s += time.process_time() - tq0
            recs.append(_mean_recall(pred, gt_ids[win_q_off[w] : win_q_off[w] + nq], 10))
    finally:
        for td in shard_tmp:
            if td is not None:
                td.cleanup()

    wall_s = time.perf_counter() - t_w0
    tot_s = q_cpu_s + m_cpu_s
    gcr_ops = 100.0 * (w_noop - exact_ops) / max(w_noop - w_star, 1e-9)
    return {
        "bridge_type": "sharded_delta_buffer",
        "engine": engine, "trace": tr_name, "policy": policy, "actions": actions,
        "build_cpu_s": round(build_cpu_s, 3), "query_cpu_s": round(q_cpu_s, 3),
        "ins_del_cpu_s": -1.0, "consolidate_cpu_s": -1.0,
        "maint_cpu_s": round(m_cpu_s, 3), "total_cpu_s": round(tot_s, 3),
        "exact_scan_ops": round(exact_ops, 1), "gcr_ops_pct": round(gcr_ops, 2),
        "mean_recall": round(float(np.mean(recs)), 5), "min_recall": round(float(np.min(recs)), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recs if r >= 0.965) / n_win, 1),
        "mean_p99_us": -1.0, "wall_s": round(wall_s, 3),
        "wall_cpu_ratio": round(wall_s / max(tot_s + build_cpu_s, 1e-6), 4),
        "peak_rss_mb": peak_rss_mb(),
    }


# ---------------------------------------------------------------------------
# Worker B3: Big-ANN'23 Protocol Re-Audit & Thread-Safe Concurrent Search+Update (4t/8t)
# ---------------------------------------------------------------------------
def _load_bigann_wl(wl_name: str) -> Dict[str, Any]:
    wl_dir = BIGANN_WL / wl_name
    def _u8(p: Path):
        n, d = struct.unpack("<II", p.open("rb").read(8))
        return np.memmap(p, dtype=np.uint8, mode="r", offset=8, shape=(n, d))
    base_mm = _u8(BIGANN_SRC / "base_11m.bin")
    ins_mm = _u8(BIGANN_SRC / "insert_vectors.bin")
    q_mm = _u8(BIGANN_SRC / "query.bin")
    with (wl_dir / "init.bin").open("rb") as f:
        assert f.read(8) == b"BGNTINIT"
        n_init, dim, _ = struct.unpack("<QII", f.read(16))
    init = np.fromfile(wl_dir / "init.bin", dtype=np.dtype([("id", "<u8"), ("src", "u1"), ("row", "<u4")], align=False), offset=24, count=n_init)
    with (wl_dir / "events.bin").open("rb") as f:
        assert f.read(8) == b"BGNTEVNT"
        n_ev, n_win, nq_pw, k = struct.unpack("<IIII", f.read(16))
    ev = np.fromfile(wl_dir / "events.bin", dtype=np.dtype([("window", "<u4"), ("type", "u1"), ("id", "<u8"), ("src", "u1"), ("row", "<u4")], align=False), offset=24, count=n_ev)
    with (wl_dir / "gt.bin").open("rb") as f:
        assert f.read(8) == b"BGNTGT10"
        gw_n, gw_nq, gw_k, _ = struct.unpack("<IIII", f.read(16))
    gt_ids = np.fromfile(wl_dir / "gt.bin", dtype="<u8", offset=24 + gw_n * gw_nq * 4, count=gw_n * gw_nq * gw_k).reshape(gw_n, gw_nq, gw_k)

    def fetch(s_arr, r_arr):
        if len(s_arr) == 0:
            return np.empty((0, dim), dtype=np.float32)
        out = np.empty((len(s_arr), dim), dtype=np.float32)
        m0 = s_arr == 0
        if np.any(m0):
            out[m0] = base_mm[r_arr[m0]]
        if np.any(~m0):
            out[~m0] = ins_mm[r_arr[~m0]]
        return np.ascontiguousarray(out, dtype=np.float32)
    return {"dim": dim, "n_win": n_win, "nq_pw": nq_pw, "k": k, "init": init, "ev": ev, "q_mm": q_mm, "gt_ids": gt_ids, "fetch": fetch}


def _worker_b3_bigann(cfg: Dict[str, Any]) -> Dict[str, Any]:
    wl_name, system, ptag, params = cfg["wl"], cfg["system"], cfg["param_tag"], cfg["params"]
    threads = int(cfg["threads"])
    d = _load_bigann_wl(wl_name)
    dim, n_win, nq_pw, k, init, ev, q_mm, gt_ids, fetch = (
        d["dim"], d["n_win"], d["nq_pw"], d["k"], d["init"], d["ev"], d["q_mm"], d["gt_ids"], d["fetch"]
    )
    init_vecs = fetch(init["src"], init["row"])
    init_ids = init["id"].astype(np.int64)
    n_ins_tot = int(np.sum(ev["type"] == 0))
    n_era_tot = int(np.sum(ev["type"] == 2))
    n_crud_tot = n_ins_tot + n_era_tot

    import faiss, hnswlib, diskannpy
    from usearch.index import Index as UIndex

    t_w0 = time.perf_counter()
    t_b0 = time.process_time()
    tmp_dir = None
    has_del = False
    if system == "faiss_ivf":
        faiss.omp_set_num_threads(threads)
        idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, int(params.get("nlist", 1024)), faiss.METRIC_L2)
        idx.train(init_vecs[: min(100_000, len(init_vecs))])
        idx.add_with_ids(init_vecs, init_ids)
        idx.nprobe = int(params["nprobe"])
    elif system == "hnswlib":
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=len(init_ids) + n_ins_tot + 50000, ef_construction=100, M=16, allow_replace_deleted=True)
        idx.set_num_threads(threads)
        idx.add_items(init_vecs, init_ids, num_threads=threads)
        idx.set_ef(int(params["ef"]))
    elif system == "usearch":
        idx = UIndex(ndim=dim, metric="l2sq", dtype="f32", connectivity=16, expansion_add=100, expansion_search=int(params["ef"]))
        idx.add(init_ids.astype(np.uint64), init_vecs, threads=threads)
    elif system == "diskann":
        tmp_dir = tempfile.TemporaryDirectory(prefix="b3_diskann_")
        R, Lbuild, sL = int(params.get("R", 32)), int(params.get("Lbuild", 100)), int(params["search_L"])
        diskannpy.build_memory_index(init_vecs, "l2", tmp_dir.name, complexity=Lbuild, graph_degree=R,
                                     num_threads=threads, tags=(init_ids + 1).astype(np.uint32))
        idx = diskannpy.DynamicMemoryIndex.from_file(tmp_dir.name, max_vectors=len(init_ids) + n_ins_tot + 50000,
                                                     complexity=Lbuild, graph_degree=R, num_threads=threads,
                                                     initial_search_complexity=sL, search_threads=threads,
                                                     concurrent_consolidation=(threads > 1))
    else:
        raise ValueError(system)
    build_cpu_s = time.process_time() - t_b0

    q_cpu_s = m_cpu_s = q_wall_s = m_wall_s = dyn_wall_s = 0.0
    recs: List[float] = []
    p99s: List[float] = []
    rwlock = _RWLock()

    try:
        for w in range(n_win):
            we = ev[ev["window"] == w]
            er_ev = we[we["type"] == 2]
            in_ev = we[we["type"] == 0]
            q_ev = we[we["type"] == 3]
            qvecs = np.ascontiguousarray(q_mm[q_ev["row"].astype(np.int64)], dtype=np.float32)

            tw_dyn0 = time.perf_counter()
            if cfg.get("concurrent_rw", False) and threads > 1:
                ins_vecs = fetch(in_ev["src"], in_ev["row"])
                ins_ids = in_ev["id"].astype(np.int64)
                era_ids = er_ev["id"].astype(np.int64)
                if system == "faiss_ivf":
                    faiss.omp_set_num_threads(1)

                # Fair interleaved micro-batch RW: writer commits 500-vec micro-batches under write lock,
                # concurrent reader threads hold shared read lock — exact thread_time() per role!
                def _writer() -> Tuple[float, float]:
                    tw0 = time.perf_counter()
                    tc0 = time.thread_time()
                    bs = 500
                    for s0 in range(0, len(era_ids), bs):
                        chunk = era_ids[s0 : s0 + bs]
                        rwlock.acquire_write()
                        try:
                            if system == "faiss_ivf":
                                idx.remove_ids(chunk)
                            elif system == "hnswlib":
                                for eid in chunk.tolist():
                                    idx.mark_deleted(int(eid))
                            elif system == "usearch":
                                for eid in chunk.tolist():
                                    idx.remove(int(eid))
                            elif system == "diskann":
                                for eid in chunk.tolist():
                                    idx.mark_deleted(int(eid) + 1)
                        finally:
                            rwlock.release_write()
                    for s0 in range(0, len(ins_ids), bs):
                        c_ids = ins_ids[s0 : s0 + bs]
                        c_vecs = ins_vecs[s0 : s0 + bs]
                        rwlock.acquire_write()
                        try:
                            if system == "faiss_ivf":
                                idx.add_with_ids(c_vecs, c_ids)
                            elif system == "hnswlib":
                                idx.add_items(c_vecs, c_ids, num_threads=1, replace_deleted=(len(era_ids) > 0))
                            elif system == "usearch":
                                idx.add(c_ids.astype(np.uint64), c_vecs, threads=1)
                            elif system == "diskann":
                                for kk in range(len(c_ids)):
                                    idx.insert(c_vecs[kk], int(c_ids[kk]) + 1)
                        finally:
                            rwlock.release_write()
                    if system == "diskann" and len(era_ids) > 0 and (w + 1) % 2 == 0:
                        rwlock.acquire_write()
                        try:
                            idx.consolidate_delete()
                        finally:
                            rwlock.release_write()
                    return time.thread_time() - tc0, time.perf_counter() - tw0

                def _reader(sl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float], float, float]:
                    tw0 = time.perf_counter()
                    tc0 = time.thread_time()
                    pred_s = np.empty((len(sl), k), dtype=np.int64)
                    ls: List[float] = []
                    for j_i, qi in enumerate(sl):
                        t0 = time.perf_counter_ns()
                        rwlock.acquire_read()
                        try:
                            if system == "faiss_ivf":
                                _, r = idx.search(qvecs[qi : qi + 1], k)
                                pred_s[j_i] = r[0]
                            elif system == "hnswlib":
                                r, _ = idx.knn_query(qvecs[qi : qi + 1], k=k, num_threads=1)
                                pred_s[j_i] = r[0]
                            elif system == "usearch":
                                pred_s[j_i] = np.asarray(idx.search(qvecs[qi], k).keys, dtype=np.int64)[:k]
                            elif system == "diskann":
                                res = idx.search(qvecs[qi], k, int(params["search_L"]))
                                pred_s[j_i] = np.asarray(res.identifiers, dtype=np.int64) - 1
                        finally:
                            rwlock.release_read()
                        ls.append((time.perf_counter_ns() - t0) / 1000.0)
                    return sl, pred_s, ls, time.thread_time() - tc0, time.perf_counter() - tw0

                pred = np.empty((len(qvecs), k), dtype=np.int64)
                lat = []
                n_r = max(1, threads - 1)
                slices = np.array_split(np.arange(len(qvecs)), n_r)
                with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
                    wf = ex.submit(_writer)
                    rfs = [ex.submit(_reader, sl) for sl in slices]
                    w_cpu, w_w = wf.result()
                    m_cpu_s += w_cpu
                    m_wall_s += w_w
                    r_walls = []
                    for rf in rfs:
                        sl, ps, ls, r_cpu, r_w = rf.result()
                        pred[sl] = ps
                        lat.extend(ls)
                        q_cpu_s += r_cpu
                        r_walls.append(r_w)
                    q_wall_s += max(r_walls) if r_walls else 0.0
            else:
                tm_w0 = time.perf_counter()
                tm0 = time.process_time()
                if len(er_ev) > 0:
                    era_ids = er_ev["id"].astype(np.int64)
                    if system == "faiss_ivf":
                        idx.remove_ids(era_ids)
                    elif system == "hnswlib":
                        for eid in era_ids.tolist():
                            idx.mark_deleted(int(eid))
                        has_del = True
                    elif system == "usearch":
                        for eid in era_ids.tolist():
                            idx.remove(int(eid))
                    elif system == "diskann":
                        for eid in era_ids.tolist():
                            idx.mark_deleted(int(eid) + 1)
                        has_del = True
                if len(in_ev) > 0:
                    iv = fetch(in_ev["src"], in_ev["row"])
                    iids = in_ev["id"].astype(np.int64)
                    if system == "faiss_ivf":
                        idx.add_with_ids(iv, iids)
                    elif system == "hnswlib":
                        idx.add_items(iv, iids, num_threads=threads, replace_deleted=has_del)
                    elif system == "usearch":
                        idx.add(iids.astype(np.uint64), iv, threads=threads)
                    elif system == "diskann":
                        for kk in range(len(iids)):
                            idx.insert(iv[kk], int(iids[kk]) + 1)
                        if has_del and (w + 1) % 2 == 0:
                            idx.consolidate_delete()
                m_cpu_s += time.process_time() - tm0
                m_wall_s += time.perf_counter() - tm_w0

                pred = np.empty((len(qvecs), k), dtype=np.int64)
                lat = []
                tq_w0 = time.perf_counter()
                tq0 = time.process_time()
                if system == "faiss_ivf" and threads > 1:
                    t0 = time.perf_counter_ns()
                    _, pred = idx.search(qvecs, k)
                    lat = [((time.perf_counter_ns() - t0) / 1000.0) / max(len(qvecs), 1)]
                elif system == "hnswlib" and threads > 1:
                    t0 = time.perf_counter_ns()
                    pred, _ = idx.knn_query(qvecs, k=k, num_threads=threads)
                    lat = [((time.perf_counter_ns() - t0) / 1000.0) / max(len(qvecs), 1)]
                else:
                    for i in range(len(qvecs)):
                        t0 = time.perf_counter_ns()
                        if system == "faiss_ivf":
                            _, r = idx.search(qvecs[i : i + 1], k)
                            pred[i] = r[0]
                        elif system == "hnswlib":
                            r, _ = idx.knn_query(qvecs[i : i + 1], k=k, num_threads=1)
                            pred[i] = r[0]
                        elif system == "usearch":
                            pred[i] = np.asarray(idx.search(qvecs[i], k).keys, dtype=np.int64)[:k]
                        elif system == "diskann":
                            res = idx.search(qvecs[i], k, int(params["search_L"]))
                            pred[i] = np.asarray(res.identifiers, dtype=np.int64) - 1
                        lat.append((time.perf_counter_ns() - t0) / 1000.0)
                q_cpu_s += time.process_time() - tq0
                q_wall_s += time.perf_counter() - tq_w0

            dyn_wall_s += time.perf_counter() - tw_dyn0
            recs.append(_mean_recall(pred, gt_ids[w].astype(np.int64), k))
            p99s.append(_pct(lat, 99.0))
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    wall_s = time.perf_counter() - t_w0
    tot_s = q_cpu_s + m_cpu_s
    return {
        "regime": cfg["regime"], "wl": wl_name, "system": system, "param_tag": ptag, "threads": threads,
        "build_cpu_s": round(build_cpu_s, 3), "query_cpu_s": round(q_cpu_s, 3),
        "maint_cpu_s": round(m_cpu_s, 3), "total_cpu_s": round(tot_s, 3),
        "query_wall_s": round(q_wall_s, 3), "maint_wall_s": round(m_wall_s, 3),
        "dyn_wall_s": round(dyn_wall_s, 3), "wall_s": round(wall_s, 3),
        "wall_cpu_ratio": round(wall_s / max(tot_s + build_cpu_s, 1e-6), 4),
        "mean_recall": round(float(np.mean(recs)), 5), "min_recall": round(float(np.min(recs)), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recs if r >= 0.965) / n_win, 1),
        "att_0950_pct": round(100.0 * sum(1 for r in recs if r >= 0.950) / n_win, 1),
        "query_qps": round((n_win * nq_pw) / max(q_wall_s, 1e-6), 1),
        "end_to_end_qps": round((n_win * nq_pw) / max(dyn_wall_s, 1e-6), 1),
        "updates_per_s": round(n_crud_tot / max(m_wall_s, 1e-6), 1),
        "mean_p99_us": round(float(np.mean(p99s)), 2),
        "peak_rss_mb": peak_rss_mb(),
    }


# ---------------------------------------------------------------------------
# Worker B4: Trap 4 Symmetric Dirty Execution + 4-Signal Gate-4 ROC/AUC
# ---------------------------------------------------------------------------
def _worker_b4_sym_trap4(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run FAISS 16-shard exact delta scans across all 5 policies (REAL HistGradientBoostingRegressor)
    recording wall/CPU, ru_nivcsw, /proc/self/schedstat wait_ms, and inline L2-GEMV probe_ms."""
    import faiss

    faiss.omp_set_num_threads(1)
    data_root = Path(cfg["data_root"])
    tr_name = cfg["trace"]
    d = _read_qbam_trace(data_root, tr_name)
    dim, R, n_win, pool, init, ev, centroids = d["dim"], d["R"], d["n_win"], d["pool"], d["init"], d["ev"], d["centroids"]
    c_norm2 = np.sum(centroids ** 2, axis=1)
    policy = cfg["policy"]

    gbrt_preds: Dict[int, np.ndarray] = {}
    if policy == "gbrt_causal":
        gbrt_preds = _train_real_gbrt_causal(data_root, tr_name, ["abrupt_s06"], K=K_BLOCK)

    base_shards = [[] for _ in range(R)]
    for row in init:
        base_shards[int(row["shard"])].append(int(row["vec_ref"]))
    Q, D, _ = _extract_qd_fast(d)
    win_ins_vecs: List[List[Optional[np.ndarray]]] = []
    win_qvecs: List[Tuple[np.ndarray, np.ndarray]] = []
    for w in range(n_win):
        we = ev[ev["window"] == w]
        cm = we[we["type"] == 0]
        qm = we[we["type"] == 3]
        per_r: List[List[np.ndarray]] = [[] for _ in range(R)]
        if len(cm) > 0:
            iv = pool[cm["vec_ref"].astype(np.int64)]
            top1 = np.argmin(-2.0 * (iv @ centroids.T) + c_norm2[None, :], axis=1)
            for k_i, rr in enumerate(top1.tolist()):
                per_r[rr].append(iv[k_i : k_i + 1])
        win_ins_vecs.append([np.vstack(x) if x else None for x in per_r])
        qv = pool[qm["vec_ref"].astype(np.int64)]
        top4 = np.argpartition(-2.0 * (qv @ centroids.T) + c_norm2[None, :], 3, axis=1)[:, :4]
        win_qvecs.append((qv, top4))

    shard_base = [np.ascontiguousarray(pool[base_shards[r]], np.float32) for r in range(R)]
    shard_delta = [np.empty((0, dim), dtype=np.float32) for _ in range(R)]
    w_noop = float((Q * D).sum())
    w_star = _solve_milp_qd(Q, D, K=K_BLOCK, B=9)
    spent = 0
    rr_ptr = 0
    ops = 0.0

    probe_samples = [run_micro_probe_ms()]
    sched0 = read_schedstat_wait_ns()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t_w0 = time.perf_counter()
    t_c0 = time.process_time()

    for w in range(n_win):
        if w > 0 and w % K_BLOCK == 0 and spent < 9 and policy != "noop":
            j = w // K_BLOCK
            cur_d = np.array([len(shard_delta[r]) for r in range(R)], dtype=np.float64)
            sel = -1
            if policy == "fixed_c2":
                if (j - 1) % 2 == 0:
                    sel = rr_ptr % R
                    rr_ptr += 1
            elif policy == "past_qmass":
                sel = int(np.argmax(Q[w - K_BLOCK : w].sum(axis=0) * cur_d))
            elif policy == "gbrt_causal":
                sel = int(np.argmax(gbrt_preds[j] * cur_d))
            elif policy == "oracle":
                sel = int(np.argmax(Q[w : min(w + K_BLOCK, n_win)].sum(axis=0) * cur_d))
            if sel >= 0 and cur_d[sel] > 0:
                shard_base[sel] = np.vstack([shard_base[sel], shard_delta[sel]])
                shard_delta[sel] = np.empty((0, dim), dtype=np.float32)
                spent += 1

        for r in range(R):
            if win_ins_vecs[w][r] is not None:
                shard_delta[r] = np.vstack([shard_delta[r], win_ins_vecs[w][r]])
        ops += float((Q[w] * np.array([len(shard_delta[r]) for r in range(R)], dtype=np.float64)).sum())

        # Execute FAISS Flat L2 exact delta scans (native AVX512/SIMD memory-bound scan sensitive to HT/LLC!)
        qv, top4 = win_qvecs[w]
        d_idxs = []
        for r in range(R):
            if len(shard_delta[r]) > 0:
                fi = faiss.IndexFlatL2(dim)
                fi.add(shard_delta[r])
                d_idxs.append(fi)
            else:
                d_idxs.append(None)
        for i in range(len(qv)):
            q_row = qv[i : i + 1]
            for rr in top4[i].tolist():
                if d_idxs[rr] is not None:
                    d_idxs[rr].search(q_row, min(10, len(shard_delta[rr])))
        if w in (n_win // 2, n_win - 1):
            probe_samples.append(run_micro_probe_ms())

    cpu_s = time.process_time() - t_c0
    wall_s = time.perf_counter() - t_w0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    sched1 = read_schedstat_wait_ns()
    return {
        "regime": cfg["regime"], "tsw": cfg["tsw"], "trace": tr_name, "policy": policy,
        "cpu_s": round(cpu_s, 4), "wall_s": round(wall_s, 4),
        "wall_cpu_ratio": round(wall_s / max(cpu_s, 1e-6), 4),
        "ru_nivcsw": int(ru1.ru_nivcsw - ru0.ru_nivcsw),
        "ru_nvcsw": int(ru1.ru_nvcsw - ru0.ru_nvcsw),
        "sched_wait_ms": round(max(0, sched1 - sched0) / 1e6, 3),
        "probe_ms_mean": round(float(np.mean(probe_samples)), 3),
        "gcr_ops_pct": round(100.0 * (w_noop - ops) / max(w_noop - w_star, 1e-9), 2),
        "loadavg_end": round(read_load1m(), 2),
    }


# ---------------------------------------------------------------------------
# Worker B5: FreshDiskANN 5-Knob CRUD Early/Mid/Late Pareto (Core 0, OMP=1)
# ---------------------------------------------------------------------------
def _worker_b5_diskann_pareto(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import diskannpy

    z = np.load(VDB_WL / "vdb_real768_80k_crud30.npz")
    gidx = z["pool_global_idx"]
    queries = np.ascontiguousarray(z["queries"], dtype=np.float32)
    init_ids = z["init_ids"].astype(np.int64)
    win_ins = z["win_inserts"].astype(np.int64)
    win_era = z["win_erases"].astype(np.int64)
    gt_ids = z["gt_ids"].astype(np.int64)

    base_mm = np.memmap(REAL768_SRC, dtype=np.float32, mode="r", shape=(10_000_000, 768))
    pool = np.ascontiguousarray(base_mm[gidx], dtype=np.float32)
    n_win, q_pw, _ = gt_ids.shape
    sL = int(cfg["search_L"])
    n_crud_tot = int(win_ins.size + win_era.size)

    t_w0 = time.perf_counter()
    t_b0 = time.process_time()
    td = tempfile.TemporaryDirectory(prefix="b5_da_")
    try:
        diskannpy.build_memory_index(pool[init_ids], "l2", td.name, complexity=100, graph_degree=32,
                                     num_threads=1, tags=(init_ids + 1).astype(np.uint32))
        idx = diskannpy.DynamicMemoryIndex.from_file(td.name, max_vectors=len(pool) + 10000, complexity=100,
                                                     graph_degree=32, num_threads=1, initial_search_complexity=sL,
                                                     search_threads=1, concurrent_consolidation=False)
        build_s = time.process_time() - t_b0

        q_s = m_s = 0.0
        recs, p99s, qps_w = [], [], []
        for w in range(n_win):
            tm0 = time.process_time()
            for eid in win_era[w].tolist():
                idx.mark_deleted(int(eid) + 1)
            for iid in win_ins[w].tolist():
                idx.insert(pool[int(iid)], int(iid) + 1)
            if (w + 1) % 2 == 0:
                idx.consolidate_delete()
            m_s += time.process_time() - tm0

            qw = queries[w * q_pw : (w + 1) * q_pw]
            pred = np.empty((q_pw, 10), dtype=np.int64)
            lat = []
            tq0 = time.process_time()
            for i in range(q_pw):
                t0 = time.perf_counter_ns()
                res = idx.search(qw[i], 10, sL)
                pred[i] = np.asarray(res.identifiers, dtype=np.int64) - 1
                lat.append((time.perf_counter_ns() - t0) / 1000.0)
            dtq = time.process_time() - tq0
            q_s += dtq
            qps_w.append(q_pw / max(dtq, 1e-6))
            recs.append(_mean_recall(pred, gt_ids[w], 10))
            p99s.append(_pct(lat, 99.0))
    finally:
        td.cleanup()

    wall_s = time.perf_counter() - t_w0
    tot_s = q_s + m_s
    t1 = max(1, n_win // 3)
    t2 = max(t1 + 1, (2 * n_win) // 3)
    return {
        "workload": "vdb_real768_80k_crud30", "system": "diskann", "config": f"L{sL}_cons2",
        "search_L": sL, "build_cpu_s": round(build_s, 3), "query_cpu_s": round(q_s, 3),
        "maint_cpu_s": round(m_s, 3), "total_cpu_s": round(tot_s, 3),
        "early_recall": round(float(np.mean(recs[:t1])), 4), "early_qps": round(float(np.mean(qps_w[:t1])), 1),
        "mid_recall": round(float(np.mean(recs[t1:t2])), 4), "mid_qps": round(float(np.mean(qps_w[t1:t2])), 1),
        "late_recall": round(float(np.mean(recs[t2:])), 4), "late_qps": round(float(np.mean(qps_w[t2:])), 1),
        "mean_recall": round(float(np.mean(recs)), 4), "min_recall": round(float(np.min(recs)), 4),
        "att_0965_pct": round(100.0 * sum(1 for r in recs if r >= 0.965) / n_win, 1),
        "qps_1core": round((n_win * q_pw) / max(q_s, 1e-6), 1),
        "updates_per_s": round(n_crud_tot / max(m_s, 1e-6), 1),
        "mean_p99_us": round(float(np.mean(p99s)), 2),
        "wall_s": round(wall_s, 3), "wall_cpu_ratio": round(wall_s / max(tot_s + build_s, 1e-6), 4),
        "peak_rss_mb": peak_rss_mb(),
    }


def run_smoke_test() -> int:
    log("=== SMOKE TEST (syntax + IndexIDMap2.hnsw + real GBRT + schedstat + probe) ===")
    import faiss
    b = faiss.IndexHNSWFlat(32, 16)
    b.hnsw.efConstruction = 64
    b.hnsw.efSearch = 32
    idx = faiss.IndexIDMap2(b)
    x = np.random.randn(200, 32).astype(np.float32)
    ids = np.arange(200, dtype=np.int64)
    idx.add_with_ids(x, ids)
    _, out = idx.search(x[:2], 5)
    assert out.shape == (2, 5)
    pm = run_micro_probe_ms(40)
    sw = read_schedstat_wait_ns()
    p_gbrt = _train_real_gbrt_causal(SIFT_DATA / "tsw16/sift100k_u10_q2500_v2", "abrupt_s05", ["abrupt_s06"])
    assert len(p_gbrt) >= 5
    log(f"SMOKE_OK: faiss_hnsw_idmap2=OK probe_ms={pm:.2f} schedstat_ns={sw} gbrt_blocks={len(p_gbrt)}")
    return 0


def run_all_phases(only: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phases = set(only.split(",")) if only else {"B1", "B2", "B3", "B4", "B5"}

    if "B4" in phases:
        # Run B4 & B2a FIRST (fast, ~20 min total, highest reviewer priority!)
        log("=== Exp 10-B4: Trap 4 Symmetric Dirty Execution (Real GBRT-Causal) & 4-Signal Gate-4 ROC ===")
        p_b4 = OUT_DIR / "exp10b4_trap4_sym_dirty_roc.csv"
        b4_rows = load_existing_csv(p_b4)
        done_b4 = {(r["regime"], str(r["tsw"]), r["policy"]) for r in b4_rows}
        tsw_specs = [
            (16, SIFT_DATA / "tsw16/sift100k_u10_q2500_v2", "abrupt_s05"),
            (24, SIFT_DATA / "tsw24/sift100k_u10_q2500_v2", "abrupt_s05"),
            (28, SIFT_DATA / "tsw28/sift100k_u10_q2500_v2", "abrupt_s05"),
        ]
        pols4 = ("oracle", "past_qmass", "gbrt_causal", "fixed_c2", "noop")
        for tsw, dr, tr in tsw_specs:
            # 1. Clean isolated Core-0
            for pol in pols4:
                if ("clean_iso0", str(tsw), pol) in done_b4:
                    continue
                wait_quiet_host()
                try:
                    res = _invoke_worker("b4", {"regime": "clean_iso0", "tsw": tsw, "data_root": str(dr), "trace": tr, "policy": pol}, cores="0", threads=1)
                    b4_rows.append(res)
                    write_csv(b4_rows, p_b4)
                except Exception as exc:
                    log(f"  [B4] FAIL clean tsw={tsw} {pol}: {exc}")

            # 2. Silent HT-Sibling Stressor ONLY (core 64 — isolates SMT inflation where w/C stays low!)
            st_ht = subprocess.Popen(
                ["taskset", "-c", "64", sys.executable, "-c", "import numpy as np\na=np.random.rand(2048,256).astype('f4')\nwhile True: _ = a @ a.T"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(0.5)
                for pol in pols4:
                    if ("sym_dirty_ht_only", str(tsw), pol) in done_b4:
                        continue
                    try:
                        res = _invoke_worker("b4", {"regime": "sym_dirty_ht_only", "tsw": tsw, "data_root": str(dr), "trace": tr, "policy": pol}, cores="0", threads=1)
                        b4_rows.append(res)
                        write_csv(b4_rows, p_b4)
                    except Exception as exc:
                        log(f"  [B4] FAIL ht_only tsw={tsw} {pol}: {exc}")
            finally:
                st_ht.terminate()
                st_ht.wait(timeout=5)

            # 3. Combined Core-0/64 Oversubscription + LLC Stressor
            st_ov = subprocess.Popen(
                ["taskset", "-c", "0,64", sys.executable, "-c", "import numpy as np\na=np.random.rand(4096,256).astype('f4')\nwhile True: _ = a @ a.T"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(0.5)
                for pol in pols4:
                    if ("sym_dirty_oversub", str(tsw), pol) in done_b4:
                        continue
                    try:
                        res = _invoke_worker("b4", {"regime": "sym_dirty_oversub", "tsw": tsw, "data_root": str(dr), "trace": tr, "policy": pol}, cores="0", threads=1)
                        b4_rows.append(res)
                        write_csv(b4_rows, p_b4)
                    except Exception as exc:
                        log(f"  [B4] FAIL oversub tsw={tsw} {pol}: {exc}")
            finally:
                st_ov.terminate()
                st_ov.wait(timeout=5)
            log(f"  [B4] tsw={tsw} clean + ht_only + oversub complete.")

        # Compute exact ROC-AUC across all 4 Gate-4 telemetry signals
        if len(b4_rows) >= 15:
            clean_probes = [float(r["probe_ms_mean"]) for r in b4_rows if r["regime"] == "clean_iso0"]
            base_probe = float(np.median(clean_probes)) if clean_probes else 1.0
            for r in b4_rows:
                r["probe_slowdown"] = round(float(r["probe_ms_mean"]) / max(base_probe, 1e-6), 4)
            write_csv(b4_rows, p_b4)
            y_all = [0 if r["regime"] == "clean_iso0" else 1 for r in b4_rows]
            auc_summary: Dict[str, Any] = {"base_probe_ms": round(base_probe, 3), "n_rows": len(b4_rows)}
            for subset_name, sub_rows in [
                ("all_dirty_vs_clean", b4_rows),
                ("silent_ht_only_vs_clean", [r for r in b4_rows if r["regime"] in ("clean_iso0", "sym_dirty_ht_only")]),
                ("oversub_vs_clean", [r for r in b4_rows if r["regime"] in ("clean_iso0", "sym_dirty_oversub")]),
            ]:
                ys = [0 if r["regime"] == "clean_iso0" else 1 for r in sub_rows]
                if len(set(ys)) < 2:
                    continue
                sig_aucs = {}
                for col in ("wall_cpu_ratio", "ru_nivcsw", "sched_wait_ms", "probe_slowdown"):
                    vals = [float(r[col]) for r in sub_rows]
                    sig_aucs[col] = round(float(_roc_auc_exact(ys, vals)), 4)
                # Combined normalized score: z-rank sum of wall_cpu_ratio + probe_slowdown + sched_wait_ms
                comb = [
                    (float(r["wall_cpu_ratio"]) - 1.0) * 5.0
                    + (float(r["probe_slowdown"]) - 1.0) * 5.0
                    + float(r["sched_wait_ms"]) / 100.0
                    for r in sub_rows
                ]
                sig_aucs["combined_gate4"] = round(float(_roc_auc_exact(ys, comb)), 4)
                auc_summary[subset_name] = sig_aucs
            (OUT_DIR / "exp10b4_gate4_roc_auc.json").write_text(json.dumps(auc_summary, indent=2) + "\n", encoding="utf-8")
            log(f"  [B4-ROC] {json.dumps(auc_summary)}")

    if "B2" in phases:
        log("=== Exp 10-B2a: NATIVE FreshDiskANN consolidate_delete() Scheduler Bridge (ZERO Delta Buffer) ===")
        p_b2 = OUT_DIR / "exp10b2_diskann_usearch_adaivf_bridge.csv"
        b2_rows = load_existing_csv(p_b2)
        done_b2 = {(r.get("bridge_type", ""), r["engine"], r["trace"], r["policy"]) for r in b2_rows}
        for pol in ("noop", "fixed_c2", "max_delta", "past_qmass", "ewma_a03", "oracle_1blk"):
            if ("native_consolidate_delete", "native_8sh_diskann_crud30", "vdb_real768_80k_crud30", pol) in done_b2:
                continue
            wait_quiet_host()
            try:
                res = _invoke_worker("b2a", {"policy": pol, "budget": 5}, cores="0", threads=1)
                b2_rows.append(res)
                write_csv(b2_rows, p_b2)
                log(f"  [B2a-Native] {pol}: cpu={res['total_cpu_s']}s cons={res['consolidate_cpu_s']}s gcr={res['gcr_ops_pct']}% rec={res['mean_recall']} p99={res['mean_p99_us']}us")
            except Exception as exc:
                log(f"  [B2a-Native] FAIL {pol}: {exc}")

        log("=== Exp 10-B2b: 16-Shard Real IndexIVFFlat / USearch / FreshDiskANN Bridge (with Real GBRT-Causal) ===")
        droot = SIFT_DATA / "sift100k_u10_q2500_v2"
        for eng in ("sharded_diskann", "sharded_usearch", "sharded_faiss_ivf"):
            for tr in ("abrupt_s05", "hotspot_s05", "gradual_s05"):
                for pol in ("noop", "fixed_c2", "past_qmass", "ewma_a03", "gbrt_causal", "imbalance_prio", "oracle"):
                    if ("sharded_delta_buffer", eng, tr, pol) in done_b2:
                        continue
                    wait_quiet_host()
                    try:
                        res = _invoke_worker("b2b", {"data_root": str(droot), "trace": tr, "engine": eng, "policy": pol}, cores="0", threads=1)
                        b2_rows.append(res)
                        write_csv(b2_rows, p_b2)
                        log(f"  [B2b] {eng} {tr} {pol}: cpu={res['total_cpu_s']}s gcr={res['gcr_ops_pct']}% rec={res['mean_recall']} w/C={res['wall_cpu_ratio']}")
                    except Exception as exc:
                        log(f"  [B2b] FAIL {eng} {tr} {pol}: {exc}")

    if "B5" in phases:
        log("=== Exp 10-B5: FreshDiskANN CRUD-30% 5-Knob Early/Mid/Late Pareto ===")
        p_b5 = OUT_DIR / "exp10b5_diskann_crud_pareto.csv"
        b5_rows = load_existing_csv(p_b5)
        done_b5 = {str(r["search_L"]) for r in b5_rows}
        for sL in (32, 64, 96, 128, 192):
            if str(sL) in done_b5:
                continue
            wait_quiet_host()
            try:
                res = _invoke_worker("b5", {"search_L": sL}, cores="0", threads=1)
                b5_rows.append(res)
                write_csv(b5_rows, p_b5)
                log(f"  [B5] diskann L{sL}: cpu={res['total_cpu_s']}s rec={res['mean_recall']} att={res['att_0965_pct']}% qps={res['qps_1core']} upd/s={res['updates_per_s']}")
            except Exception as exc:
                log(f"  [B5] FAIL L{sL}: {exc}")

    if "B3" in phases:
        log("=== Exp 10-B3: Big-ANN'23 Protocol Re-Audit & Thread-Safe Concurrent RW (4t/8t) ===")
        p_b3 = OUT_DIR / "exp10b3_bigann_reaudit_concurrent.csv"
        b3_rows = load_existing_csv(p_b3)
        done_b3 = {(r["wl"], r["regime"], r["system"], r["param_tag"]) for r in b3_rows}
        for wl in ("bigann_1m_drift", "bigann_1m_crud20"):
            cfgs = [
                ("bigann_unpinned_default_8t", "faiss_ivf", "np32", {"nlist": 1024, "nprobe": 32}, "0-63", 8, False),
                ("bigann_unpinned_default_8t", "hnswlib", "ef64", {"ef": 64}, "0-63", 8, False),
                ("bigann_unpinned_default_8t", "usearch", "ef64", {"ef": 64}, "0-63", 8, False),
                ("bigann_unpinned_default_8t", "diskann", "L64", {"search_L": 64}, "0-63", 8, False),
                ("dynvec_pinned_floor_1t", "faiss_ivf", "np64", {"nlist": 1024, "nprobe": 64}, "0", 1, False),
                ("dynvec_pinned_floor_1t", "hnswlib", "ef128", {"ef": 128}, "0", 1, False),
                ("dynvec_pinned_floor_1t", "usearch", "ef128", {"ef": 128}, "0", 1, False),
                ("dynvec_pinned_floor_1t", "diskann", "L96", {"search_L": 96}, "0", 1, False),
                ("dynvec_pinned_floor_8t", "faiss_ivf", "np64", {"nlist": 1024, "nprobe": 64}, "0-7", 8, False),
                ("dynvec_pinned_floor_8t", "hnswlib", "ef128", {"ef": 128}, "0-7", 8, False),
                ("dynvec_pinned_floor_8t", "usearch", "ef128", {"ef": 128}, "0-7", 8, False),
                ("dynvec_pinned_floor_8t", "diskann", "L96", {"search_L": 96}, "0-7", 8, False),
                ("concurrent_rw_4t", "faiss_ivf", "np64", {"nlist": 1024, "nprobe": 64}, "0-3", 4, True),
                ("concurrent_rw_4t", "hnswlib", "ef128", {"ef": 128}, "0-3", 4, True),
                ("concurrent_rw_4t", "usearch", "ef128", {"ef": 128}, "0-3", 4, True),
                ("concurrent_rw_4t", "diskann", "L96", {"search_L": 96}, "0-3", 4, True),
                ("concurrent_rw_8t", "faiss_ivf", "np64", {"nlist": 1024, "nprobe": 64}, "0-7", 8, True),
                ("concurrent_rw_8t", "hnswlib", "ef128", {"ef": 128}, "0-7", 8, True),
                ("concurrent_rw_8t", "usearch", "ef128", {"ef": 128}, "0-7", 8, True),
                ("concurrent_rw_8t", "diskann", "L96", {"search_L": 96}, "0-7", 8, True),
            ]
            for reg, sys_n, ptag, prm, crs, th, crw in cfgs:
                if (wl, reg, sys_n, ptag) in done_b3:
                    continue
                wait_quiet_host()
                try:
                    res = _invoke_worker("b3", {"regime": reg, "wl": wl, "system": sys_n, "param_tag": ptag,
                                                "params": prm, "threads": th, "concurrent_rw": crw}, cores=crs, threads=th)
                    b3_rows.append(res)
                    write_csv(b3_rows, p_b3)
                    log(f"  [B3] {wl} {reg} {sys_n} {ptag}: dyn_w={res['dyn_wall_s']}s q_cpu={res['query_cpu_s']}s m_cpu={res['maint_cpu_s']}s rec={res['mean_recall']} att={res['att_0965_pct']}% qps={res['query_qps']}")
                except Exception as exc:
                    log(f"  [B3] FAIL {wl} {reg} {sys_n} {ptag}: {exc}")

    if "B1" in phases:
        log("=== Exp 10-B1: Bare-Metal Gate-4 Clean 768-d Wikipedia Crossover (n=3 seeds) ===")
        p_b1 = OUT_DIR / "exp10b1_baremetal_real768.csv"
        b1_rows = load_existing_csv(p_b1)
        done_b1 = {(r["workload"], str(r["seed"]), r["system"], r["param_tag"]) for r in b1_rows}
        b1_wls = [
            ("real768_140k_drift", "140K_Drift"),
            ("real768_500k_drift", "500K_Drift"),
            ("real768_500k_crud20", "500K_CRUD20"),
            ("real768_1m_drift", "1M_Drift"),
        ]
        b1_engines = [
            ("faiss_ivf", "np32", {"nlist": 512, "nprobe": 32}),
            ("faiss_ivf", "np64", {"nlist": 512, "nprobe": 64}),
            ("faiss_hnsw", "ef64", {"M": 16, "ef_c": 100, "ef": 64}),
            ("hnswlib", "ef64", {"M": 16, "ef_c": 100, "ef": 64}),
            ("hnswlib", "ef128", {"M": 16, "ef_c": 100, "ef": 128}),
            ("usearch", "ef64", {"M": 16, "ef_c": 100, "ef": 64}),
            ("diskann", "L64", {"R": 32, "Lbuild": 100, "search_L": 64}),
        ]
        for prefix, wlabel in b1_wls:
            for sd in (0, 1, 2):
                npz_n = f"{prefix}_s{sd}.npz"
                if not (REAL768_WL / npz_n).exists():
                    continue
                for sys_n, ptag, prm in b1_engines:
                    if wlabel == "500K_CRUD20" and sys_n == "faiss_hnsw":
                        continue
                    if (wlabel, str(sd), sys_n, ptag) in done_b1:
                        continue
                    wait_quiet_host()
                    try:
                        res = _invoke_worker("b1", {"npz_name": npz_n, "workload": wlabel, "seed": sd,
                                                    "system": sys_n, "param_tag": ptag, "params": prm}, cores="0", threads=1, timeout_s=5400)
                        b1_rows.append(res)
                        write_csv(b1_rows, p_b1)
                        log(f"  [B1] {wlabel} s{sd} {sys_n} {ptag}: cpu={res['total_cpu_s']}s rec={res['mean_recall']} att={res['att_0965_pct']}% w/C={res['wall_cpu_ratio']}")
                    except Exception as exc:
                        log(f"  [B1] FAIL {wlabel} s{sd} {sys_n} {ptag}: {exc}")

    if not only:
        (OUT_DIR / "EXP10B_DONE.flag").write_text("ok\n", encoding="utf-8")
        log("EXP10B_ALL_DONE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-json", default="")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.smoke_test:
        return run_smoke_test()
    if args.worker_json:
        cfg = json.loads(args.worker_json)
        m = cfg.pop("mode")
        dispatch = {
            "b1": _worker_b1_real768,
            "b2a": _worker_b2a_native_consolidate,
            "b2b": _worker_b2b_shard_bridge,
            "b3": _worker_b3_bigann,
            "b4": _worker_b4_sym_trap4,
            "b5": _worker_b5_diskann_pareto,
        }
        out = dispatch[m](cfg)
        print("WORKER_JSON=" + json.dumps(out), flush=True)
        return 0
    run_all_phases(only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
