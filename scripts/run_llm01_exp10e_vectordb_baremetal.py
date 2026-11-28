#!/usr/bin/env python3
"""Exp 10-E v2 Bare-Metal Core-0 Native VectorDB + Multi-Slice Real 768-d Wikipedia Audit (Xeon-8352V):
Queued cleanly after EXP10B_DONE.flag so every run executes in an isolated Core-0 subprocess
(OMP_NUM_THREADS=1, Gate-4 w/C checked, unified L2 metric across ALL engines, ZERO pure-Python mocks):
  - E1: Native C++/Rust Full VectorDB CRUD-30% lifecycle on vdb_real768_80k_crud30.npz:
        * Milvus-Lite with EXPLICIT IVF_FLAT (nprobe 16/32/64) and EXPLICIT HNSW (ef 64/128) via prepare_index_params
        * LanceDB native Rust storage with IVF_PQ + refine_factor=10 (nprobes 16/32/64) and IVF_HNSW_SQ + refine_factor=10
        * Core library anchors: FAISS-IVF (np16/32/64), hnswlib (ef64/128), USearch (ef64/128)
        (Pure-Python `QdrantClient(":memory:")` intentionally removed because it bypasses Rust HNSW!)
  - E2: n=5 disjoint 85K chronological 768-d Wikipedia slices across FAISS-IVF (exact hardware-invariant ndis),
        hnswlib, and USearch on Core-0.
"""
from __future__ import annotations

import os

for _ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_ev] = "1"

import argparse
import csv
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BASE_EXP10 = Path(os.environ.get("DYNVECAUDIT_EXP10_DIR", "./results/exp10"))
VDB_PKGS = BASE_EXP10 / "vdb_pkgs"
REAL768_SRC = Path(os.environ.get("DYNVECAUDIT_REAL768_SRC", "./data/real10m_768/base.f32"))
WL_DIR = BASE_EXP10 / "vdb_workloads"
OUT_DIR = BASE_EXP10 / "results"

if VDB_PKGS.exists() and str(VDB_PKGS) not in sys.path:
    sys.path.insert(0, str(VDB_PKGS))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    return float(s[lo] * (1.0 - (k - lo)) + s[min(lo + 1, len(s) - 1)] * (k - lo))


def _mean_recall(pred: List[List[int]] | np.ndarray, gt: np.ndarray, k: int = 10) -> float:
    hits = 0
    nq = len(gt)
    for i in range(nq):
        row = pred[i]
        hits += len({int(x) for x in row if int(x) >= 0} & set(gt[i, :k].tolist()))
    return hits / float(max(nq * k, 1))


def _invoke_worker(mode: str, cfg: Dict[str, Any], timeout_s: int = 3600) -> Dict[str, Any]:
    payload = json.dumps({"mode": mode, **cfg})
    cmd = ["taskset", "-c", "0", sys.executable, str(Path(__file__).resolve()), "--worker-json", payload]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(VDB_PKGS)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    pr = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    if pr.returncode != 0:
        raise RuntimeError(f"worker {mode} rc={pr.returncode}: {pr.stderr[-600:]}")
    for line in pr.stdout.splitlines():
        if line.startswith("WORKER_JSON="):
            return json.loads(line.split("WORKER_JSON=", 1)[1])
    raise RuntimeError(f"Missing WORKER_JSON: {pr.stdout[-400:]}")


def _worker_e1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    z = np.load(WL_DIR / "vdb_real768_80k_crud30.npz")
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
    system, ctag, param = cfg["system"], cfg["config"], int(cfg["param"])
    n_crud_tot = int(win_ins.size + win_era.size)

    t_w0 = time.perf_counter()
    t_b0 = time.process_time()
    q_s = m_s = 0.0
    recs: List[float] = []
    p99s: List[float] = []
    qps_phases: List[float] = []
    tmpdir = tempfile.TemporaryDirectory(prefix=f"e1_{system}_")
    index_verified = system

    try:
        if system in ("milvus_lite_ivf", "milvus_lite_hnsw"):
            from pymilvus import MilvusClient
            mc = MilvusClient(f"{tmpdir.name}/milvus.db")
            ip = mc.prepare_index_params()
            if system == "milvus_lite_ivf":
                ip.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 256})
                sp = {"metric_type": "L2", "params": {"nprobe": param}}
            else:
                ip.add_index(field_name="vector", index_type="HNSW", metric_type="L2", params={"M": 16, "efConstruction": 100})
                sp = {"metric_type": "L2", "params": {"ef": param}}
            mc.create_collection("c", dimension=dim, metric_type="L2", auto_id=False, index_params=ip)
            for cs in range(0, len(init_ids), 5000):
                sl = init_ids[cs:cs + 5000]
                mc.insert("c", [{"id": int(i), "vector": pool[i].tolist()} for i in sl])
            index_verified = f"{system}:{mc.list_indexes('c')}"
            build_s = time.process_time() - t_b0
            for w in range(n_win):
                tm0 = time.process_time()
                mc.delete("c", ids=win_era[w].tolist())
                for cs in range(0, len(win_ins[w]), 5000):
                    sl = win_ins[w][cs:cs + 5000]
                    mc.insert("c", [{"id": int(i), "vector": pool[i].tolist()} for i in sl])
                m_s += time.process_time() - tm0
                qw = queries[w * q_pw : (w + 1) * q_pw]
                preds, lat = [], []
                tq0 = time.process_time()
                for i in range(q_pw):
                    t0 = time.perf_counter_ns()
                    res = mc.search("c", data=[qw[i].tolist()], limit=10, search_params=sp)
                    lat.append((time.perf_counter_ns() - t0) / 1000.0)
                    preds.append([int(x["id"]) for x in res[0]])
                dtq = time.process_time() - tq0
                q_s += dtq
                qps_phases.append(q_pw / max(dtq, 1e-6))
                recs.append(_mean_recall(preds, gt_ids[w]))
                p99s.append(_pct(lat, 99.0))
            mc.close()
        elif system in ("lancedb_ivfpq_ref10", "lancedb_ivf_hnsw_sq"):
            import lancedb, pyarrow as pa
            db = lancedb.connect(tmpdir.name)
            tbl = db.create_table("c", pa.table({"id": init_ids.tolist(), "vector": pool[init_ids].tolist()}))
            if system == "lancedb_ivfpq_ref10":
                tbl.create_index(metric="l2", num_partitions=128, num_sub_vectors=64, index_type="IVF_PQ", replace=True)
            else:
                tbl.create_index(metric="l2", num_partitions=16, m=16, ef_construction=100, index_type="IVF_HNSW_SQ", replace=True)
            build_s = time.process_time() - t_b0
            for w in range(n_win):
                tm0 = time.process_time()
                tbl.delete("id IN (" + ",".join(str(int(x)) for x in win_era[w]) + ")")
                tbl.add(pa.table({"id": win_ins[w].tolist(), "vector": pool[win_ins[w]].tolist()}))
                if (w + 1) % 3 == 0:
                    if system == "lancedb_ivfpq_ref10":
                        tbl.create_index(metric="l2", num_partitions=128, num_sub_vectors=64, index_type="IVF_PQ", replace=True)
                    else:
                        tbl.create_index(metric="l2", num_partitions=16, m=16, ef_construction=100, index_type="IVF_HNSW_SQ", replace=True)
                m_s += time.process_time() - tm0
                qw = queries[w * q_pw : (w + 1) * q_pw]
                preds, lat = [], []
                tq0 = time.process_time()
                for i in range(q_pw):
                    t0 = time.perf_counter_ns()
                    q_obj = tbl.search(qw[i]).nprobes(param)
                    if hasattr(q_obj, "refine_factor"):
                        q_obj = q_obj.refine_factor(10)
                    df_r = q_obj.limit(10).to_pandas()
                    lat.append((time.perf_counter_ns() - t0) / 1000.0)
                    preds.append(df_r["id"].astype(int).tolist())
                dtq = time.process_time() - tq0
                q_s += dtq
                qps_phases.append(q_pw / max(dtq, 1e-6))
                recs.append(_mean_recall(preds, gt_ids[w]))
                p99s.append(_pct(lat, 99.0))
        elif system == "faiss_ivf":
            import faiss
            faiss.omp_set_num_threads(1)
            idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 256, faiss.METRIC_L2)
            idx.train(pool[init_ids])
            idx.add_with_ids(pool[init_ids], init_ids)
            idx.nprobe = param
            build_s = time.process_time() - t_b0
            for w in range(n_win):
                tm0 = time.process_time()
                idx.remove_ids(win_era[w])
                idx.add_with_ids(pool[win_ins[w]], win_ins[w])
                m_s += time.process_time() - tm0
                qw = queries[w * q_pw : (w + 1) * q_pw]
                pred = np.empty((q_pw, 10), dtype=np.int64)
                lat = []
                tq0 = time.process_time()
                for i in range(q_pw):
                    t0 = time.perf_counter_ns()
                    _, row = idx.search(qw[i : i + 1], 10)
                    lat.append((time.perf_counter_ns() - t0) / 1000.0)
                    pred[i] = row[0]
                dtq = time.process_time() - tq0
                q_s += dtq
                qps_phases.append(q_pw / max(dtq, 1e-6))
                recs.append(_mean_recall(pred, gt_ids[w]))
                p99s.append(_pct(lat, 99.0))
        elif system == "hnswlib":
            import hnswlib
            idx = hnswlib.Index(space="l2", dim=dim)
            idx.init_index(max_elements=len(pool) + 5000, ef_construction=100, M=16, allow_replace_deleted=True)
            idx.set_num_threads(1)
            idx.add_items(pool[init_ids], init_ids, num_threads=1)
            idx.set_ef(param)
            build_s = time.process_time() - t_b0
            for w in range(n_win):
                tm0 = time.process_time()
                for eid in win_era[w].tolist():
                    idx.mark_deleted(int(eid))
                idx.add_items(pool[win_ins[w]], win_ins[w], num_threads=1, replace_deleted=True)
                m_s += time.process_time() - tm0
                qw = queries[w * q_pw : (w + 1) * q_pw]
                pred = np.empty((q_pw, 10), dtype=np.int64)
                lat = []
                tq0 = time.process_time()
                for i in range(q_pw):
                    t0 = time.perf_counter_ns()
                    row, _ = idx.knn_query(qw[i : i + 1], k=10, num_threads=1)
                    lat.append((time.perf_counter_ns() - t0) / 1000.0)
                    pred[i] = row[0]
                dtq = time.process_time() - tq0
                q_s += dtq
                qps_phases.append(q_pw / max(dtq, 1e-6))
                recs.append(_mean_recall(pred, gt_ids[w]))
                p99s.append(_pct(lat, 99.0))
        elif system == "usearch":
            from usearch.index import Index as UIdx
            idx = UIdx(ndim=dim, metric="l2sq", dtype="f32", connectivity=16, expansion_add=100, expansion_search=param)
            idx.add(init_ids.astype(np.uint64), pool[init_ids], threads=1)
            build_s = time.process_time() - t_b0
            for w in range(n_win):
                tm0 = time.process_time()
                for eid in win_era[w].tolist():
                    idx.remove(int(eid))
                idx.add(win_ins[w].astype(np.uint64), pool[win_ins[w]], threads=1)
                m_s += time.process_time() - tm0
                qw = queries[w * q_pw : (w + 1) * q_pw]
                pred = np.empty((q_pw, 10), dtype=np.int64)
                lat = []
                tq0 = time.process_time()
                for i in range(q_pw):
                    t0 = time.perf_counter_ns()
                    pred[i] = np.asarray(idx.search(qw[i], 10).keys, dtype=np.int64)[:10]
                    lat.append((time.perf_counter_ns() - t0) / 1000.0)
                dtq = time.process_time() - tq0
                q_s += dtq
                qps_phases.append(q_pw / max(dtq, 1e-6))
                recs.append(_mean_recall(pred, gt_ids[w]))
                p99s.append(_pct(lat, 99.0))
        else:
            raise ValueError(system)
    finally:
        tmpdir.cleanup()

    wall_s = time.perf_counter() - t_w0
    tot_s = q_s + m_s
    t1 = max(1, n_win // 3)
    t2 = max(t1 + 1, (2 * n_win) // 3)
    return {
        "system": system, "config": ctag, "metric": "L2", "index_verified": index_verified,
        "build_cpu_s": round(build_s, 3), "query_cpu_s": round(q_s, 3), "maint_cpu_s": round(m_s, 3),
        "total_cpu_s": round(tot_s, 3),
        "query_pct": round(100.0 * q_s / max(tot_s, 1e-6), 2),
        "maint_pct": round(100.0 * m_s / max(tot_s, 1e-6), 2),
        "early_recall": round(float(np.mean(recs[:t1])), 4), "early_qps": round(float(np.mean(qps_phases[:t1])), 1),
        "mid_recall": round(float(np.mean(recs[t1:t2])), 4), "mid_qps": round(float(np.mean(qps_phases[t1:t2])), 1),
        "late_recall": round(float(np.mean(recs[t2:])), 4), "late_qps": round(float(np.mean(qps_phases[t2:])), 1),
        "mean_recall": round(float(np.mean(recs)), 4), "min_recall": round(float(np.min(recs)), 4),
        "att_0950_pct": round(100.0 * sum(1 for r in recs if r >= 0.950) / n_win, 1),
        "att_0965_pct": round(100.0 * sum(1 for r in recs if r >= 0.965) / n_win, 1),
        "qps_1core": round((n_win * q_pw) / max(q_s, 1e-6), 1),
        "updates_per_s": round(n_crud_tot / max(m_s, 1e-6), 1),
        "mean_p99_us": round(float(np.mean(p99s)), 2),
        "wall_s": round(wall_s, 3), "wall_cpu_ratio": round(wall_s / max(tot_s + build_s, 1e-6), 4),
        "peak_rss_mb": peak_rss_mb(),
    }


def _worker_e2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    s_idx = int(cfg["slice"])
    z = np.load(WL_DIR / f"real768_n5_slice_{s_idx}.npz")
    gidx = z["pool_global_idx"]
    queries = np.ascontiguousarray(z["queries"], dtype=np.float32)
    init_ids = z["init_ids"].astype(np.int64)
    win_ins = z["win_inserts"].astype(np.int64)
    gt_ids = z["gt_ids"].astype(np.int64)
    base_mm = np.memmap(REAL768_SRC, dtype=np.float32, mode="r", shape=(10_000_000, 768))
    pool = np.ascontiguousarray(base_mm[gidx], dtype=np.float32)
    dim = 768
    n_win, q_pw, _ = gt_ids.shape
    system, ctag, param = cfg["system"], cfg["config"], int(cfg["param"])

    t_w0 = time.perf_counter()
    t_b0 = time.process_time()
    ndis = -1
    if system == "faiss_ivf":
        import faiss
        faiss.omp_set_num_threads(1)
        faiss.cvar.indexIVF_stats.reset()
        idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 256, faiss.METRIC_L2)
        idx.train(pool[init_ids])
        idx.add_with_ids(pool[init_ids], init_ids)
        idx.nprobe = param
    elif system == "hnswlib":
        import hnswlib
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=len(pool) + 5000, ef_construction=100, M=16)
        idx.set_num_threads(1)
        idx.add_items(pool[init_ids], init_ids, num_threads=1)
        idx.set_ef(param)
    elif system == "usearch":
        from usearch.index import Index as UIdx
        idx = UIdx(ndim=dim, metric="l2sq", dtype="f32", connectivity=16, expansion_add=100, expansion_search=param)
        idx.add(init_ids.astype(np.uint64), pool[init_ids], threads=1)
    else:
        raise ValueError(system)
    build_s = time.process_time() - t_b0

    q_s = m_s = 0.0
    recs = []
    for w in range(n_win):
        tm0 = time.process_time()
        if system == "faiss_ivf":
            idx.add_with_ids(pool[win_ins[w]], win_ins[w])
        elif system == "hnswlib":
            idx.add_items(pool[win_ins[w]], win_ins[w], num_threads=1)
        elif system == "usearch":
            idx.add(win_ins[w].astype(np.uint64), pool[win_ins[w]], threads=1)
        m_s += time.process_time() - tm0
        qw = queries[w * q_pw : (w + 1) * q_pw]
        tq0 = time.process_time()
        if system == "faiss_ivf":
            _, pred = idx.search(qw, 10)
        elif system == "hnswlib":
            pred, _ = idx.knn_query(qw, k=10, num_threads=1)
        elif system == "usearch":
            pred = np.stack([np.asarray(idx.search(qv, 10).keys, dtype=np.int64)[:10] for qv in qw], axis=0)
        q_s += time.process_time() - tq0
        recs.append(_mean_recall(pred, gt_ids[w]))
    if system == "faiss_ivf":
        import faiss
        ndis = int(faiss.cvar.indexIVF_stats.ndis)
    wall_s = time.perf_counter() - t_w0
    tot_s = q_s + m_s
    return {
        "slice": s_idx, "system": system, "config": ctag,
        "build_cpu_s": round(build_s, 3), "query_cpu_s": round(q_s, 3), "maint_cpu_s": round(m_s, 3),
        "total_cpu_s": round(tot_s, 3), "maint_pct": round(100.0 * m_s / max(tot_s, 1e-6), 2),
        "mean_recall": round(float(np.mean(recs)), 4),
        "att_0965_pct": round(100.0 * sum(1 for r in recs if r >= 0.965) / n_win, 1),
        "ndis_total": ndis, "wall_cpu_ratio": round(wall_s / max(tot_s + build_s, 1e-6), 4),
        "peak_rss_mb": peak_rss_mb(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-json", default="")
    args = ap.parse_args()
    if args.worker_json:
        cfg = json.loads(args.worker_json)
        m = cfg.pop("mode")
        out = _worker_e1(cfg) if m == "e1" else _worker_e2(cfg)
        print("WORKER_JSON=" + json.dumps(out), flush=True)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== Exp 10-E1 v2: Bare-Metal Core-0 Native VectorDB CRUD-30% (Unified L2, Zero Python Mocks) ===")
    e1_cfgs = [
        ("faiss_ivf", "np16", 16), ("faiss_ivf", "np32", 32), ("faiss_ivf", "np64", 64),
        ("hnswlib", "ef64", 64), ("hnswlib", "ef128", 128),
        ("usearch", "ef64", 64), ("usearch", "ef128", 128),
        ("milvus_lite_ivf", "ivf_np16", 16), ("milvus_lite_ivf", "ivf_np32", 32), ("milvus_lite_ivf", "ivf_np64", 64),
        ("milvus_lite_hnsw", "hnsw_ef64", 64), ("milvus_lite_hnsw", "hnsw_ef128", 128),
        ("lancedb_ivfpq_ref10", "ivfpq_ref10_np16", 16), ("lancedb_ivfpq_ref10", "ivfpq_ref10_np32", 32),
        ("lancedb_ivfpq_ref10", "ivfpq_ref10_np64", 64),
        ("lancedb_ivf_hnsw_sq", "ivfhnsw_ref10_np4", 4), ("lancedb_ivf_hnsw_sq", "ivfhnsw_ref10_np8", 8),
    ]
    e1_rows = []
    for sys_n, ct, prm in e1_cfgs:
        try:
            r = _invoke_worker("e1", {"system": sys_n, "config": ct, "param": prm})
            e1_rows.append(r)
            write_csv(e1_rows, OUT_DIR / "exp10e1_baremetal_vectordb_crud30.csv")
            log(f"  [E1] {sys_n} {ct}: cpu={r['total_cpu_s']}s rec={r['mean_recall']} att965={r['att_0965_pct']}% qps={r['qps_1core']} upd/s={r['updates_per_s']} w/C={r['wall_cpu_ratio']}")
        except Exception as exc:
            log(f"  [E1] SKIP {sys_n} {ct}: {exc}")

    log("=== Exp 10-E2: n=5 Chronological 768-d Wikipedia Slices (ndis + Bare-Metal Core-0) ===")
    e2_rows = []
    e2_cfgs = [("faiss_ivf", "np16", 16), ("faiss_ivf", "np32", 32), ("hnswlib", "ef64", 64), ("hnswlib", "ef128", 128), ("usearch", "ef64", 64)]
    for s_idx in range(5):
        for sys_n, ct, prm in e2_cfgs:
            try:
                r = _invoke_worker("e2", {"slice": s_idx, "system": sys_n, "config": ct, "param": prm})
                e2_rows.append(r)
                write_csv(e2_rows, OUT_DIR / "exp10e2_baremetal_real768_n5.csv")
            except Exception as exc:
                log(f"  [E2] SKIP slice={s_idx} {sys_n} {ct}: {exc}")
        log(f"  [E2] slice={s_idx} done.")

    (OUT_DIR / "EXP10E_DONE.flag").write_text("ok\n", encoding="utf-8")
    log("EXP10E_ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
