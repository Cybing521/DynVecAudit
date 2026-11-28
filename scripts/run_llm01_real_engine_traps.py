#!/usr/bin/env python3
"""Real-Engine Measurement Hygiene Traps & Multi-Thread/NUMA Benchmark on Xeon-8352V.

Implements 4 real-engine hygiene experiments across 5 open-source vector engines
(FAISS-IVF, hnswlib, USearch, FreshDiskANN via diskannpy, Annoy) on SIFT-100K
QBAM binary traces (`pool.bin`, `init.bin`, `trace.bin`, `gt.bin`):

  - A1+D (Real-Engine Trap 4 & MT NUMA):
      Regimes: iso_numa0_1w (core 0), ht_sibling_2w (cores 0,64),
               same_numa_4w (cores 0-3), cross_numa_8w (cores 0-3,32-35),
               iso_numa0_1t_batch (core 0, batch query),
               mt_same_numa_4t (pinned node0 0-3, 4 threads),
               mt_unpinned_8t (unpinned, 8 threads).
      Computes CPU/p99 inflation, Kendall tau, pairwise flip %, top-1 winner flip.
  - A2 (Real-Engine Trap 1 Cadence Brackets):
      Pinned Core 0, OMP=1.
      FAISS-IVF retrain_every in {0,1,2,4,8,16},
      Annoy rebuild_every in {1,2,3,4,6,8},
      FreshDiskANN consolidate_every in {0,1,2,4,8} (on CRUD-30% and insert-only).
      Audits linear interpolation vs executed cadence brackets.
  - A3 (Real-Engine Trap 3 CRUD + FreshDiskANN CRUD):
      Pinned Core 0, OMP=1 on CRUD-30%, CRUD-50%, and Deletion Runbook.
      Benchmarks FreshDiskANN (consolidate_delete ON vs OFF), FAISS-IVF,
      hnswlib, USearch, and Annoy.
  - A4 (Recall-Matched >=95% Attainment Floor):
      Pinned Core 0, OMP=1 across FAISS-IVF nprobe {8,16,24,32,48},
      hnswlib ef {32,64,96,128}, USearch ef {32,64,96,128},
      FreshDiskANN search_L {32,64,96,128}.
      Identifies each engine's minimum-CPU config attaining >=95% of windows.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import resource
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import audit_hygiene as ah  # noqa: E402

EV_INSERT, EV_UPDATE, EV_ERASE, EV_QUERY = 0, 1, 2, 3

DATA_ROOTS_LLM01 = {
    "drift": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_v2",
    "crud30v2": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_crud30v2",
    "crud50v2": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_crud50v2",
    "runbook": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_runbook",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    except Exception:
        return 0.0


def wait_quiet_host(max_load: float, checks: int = 2, sleep_s: float = 10.0) -> None:
    streak = 0
    while streak < checks:
        lv = read_load1m()
        if lv <= max_load:
            streak += 1
            if streak < checks:
                time.sleep(1.5)
        else:
            streak = 0
            time.sleep(sleep_s)


def _read_header(path: Path, magic: bytes, fmt: str) -> Tuple[Tuple[int, ...], int]:
    with path.open("rb") as f:
        got = f.read(8)
        if got != magic:
            raise ValueError(f"{path}: bad magic {got!r} != {magic!r}")
        vals = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        off = f.tell()
    return vals, off


def load_qbam_trace(data_root: Path, trace_name: str) -> Dict[str, Any]:
    """Load QBAM pool.bin, init.bin, trace.bin, and gt.bin."""
    tr_dir = data_root / "traces" / trace_name
    (_, dim, nvec), p_off = _read_header(data_root / "pool.bin", b"QBAMPOOL", "<III")
    pool = np.fromfile(data_root / "pool.bin", dtype="<f4", offset=p_off).reshape(nvec, dim)

    (_, _, n_init, _), i_off = _read_header(data_root / "init.bin", b"QBAMINIT", "<IIII")
    init_dtype = np.dtype([("id", "<u8"), ("shard", "<u4"), ("vec_ref", "<u4")])
    init = np.fromfile(data_root / "init.bin", dtype=init_dtype, offset=i_off, count=n_init)

    (_, _, n_ev, n_win, k_cfg), t_off = _read_header(tr_dir / "trace.bin", b"QBAMTRAC", "<IIIII")
    ev_dtype = np.dtype(
        [("window", "<u4"), ("type", "u1"), ("id", "<u8"), ("version", "<u8"), ("vec_ref", "<u4"), ("flags", "u1")]
    )
    ev = np.fromfile(tr_dir / "trace.bin", dtype=ev_dtype, offset=t_off, count=n_ev)

    gt_path = tr_dir / "gt.bin"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing gt.bin at {gt_path}; precompute with GPU first.")
    (_, k_gt, n_q_gt, _), g_off = _read_header(gt_path, b"QBAMGT\0\0", "<IIII")
    gt_dtype = np.dtype([("id", "<u8"), ("dist", "<f4")])
    gt = np.fromfile(gt_path, dtype=gt_dtype, offset=g_off, count=n_q_gt * k_gt).reshape(n_q_gt, k_gt)

    win_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    q_offsets: List[int] = []
    q_cur = 0
    for w in range(n_win):
        w_mask = ev["window"] == w
        w_ev = ev[w_mask]
        crud = w_ev[w_ev["type"] != EV_QUERY]
        q_ev = w_ev[w_ev["type"] == EV_QUERY]
        win_splits.append((crud, q_ev))
        q_offsets.append(q_cur)
        q_cur += len(q_ev)

    return {
        "dim": int(dim),
        "n_windows": int(n_win),
        "k": int(k_gt),
        "K_cfg": int(k_cfg),
        "pool": np.ascontiguousarray(pool, dtype=np.float32),
        "init": init,
        "ev": ev,
        "gt_ids": gt["id"].astype(np.int64),
        "win_splits": win_splits,
        "q_offsets": q_offsets,
    }


def _cpu_ms() -> float:
    return time.process_time() * 1000.0


def _pct(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * p / 100.0)))
    return float(sorted_vals[idx])


def _mean_recall(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    nq = pred.shape[0]
    if nq == 0:
        return 0.0
    hits = 0
    for i in range(nq):
        row = pred[i]
        valid = row[row >= 0]
        if valid.size == 0:
            continue
        hits += len(set(valid.tolist()) & set(gt[i, :k].tolist()))
    return hits / float(nq * k)


def _set_thread_env(threads: int) -> None:
    for k_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k_env] = str(threads)


def _build_initial_index(
    system: str,
    params: Dict[str, Any],
    dim: int,
    pool: np.ndarray,
    init: np.ndarray,
    ev: np.ndarray,
    threads: int,
) -> Dict[str, Any]:
    init_ids = init["id"].astype(np.int64)
    init_vrefs = init["vec_ref"].astype(np.int64)
    init_vecs = np.ascontiguousarray(pool[init_vrefs], dtype=np.float32)

    types_all = ev["type"]
    n_updates_total = int(np.sum(types_all == EV_UPDATE))
    n_erases_total = int(np.sum(types_all == EV_ERASE))
    n_inserts_total = int(np.sum(types_all == EV_INSERT))
    has_churn = (n_updates_total + n_erases_total) > 0

    state: Dict[str, Any] = {
        "live_map": dict(zip(init_ids.tolist(), init_vrefs.tolist())),
        "has_deleted_hnsw": False,
        "usearch_synth_ids": None,
        "usearch_label_of": {},
        "usearch_label_base": 0,
        "usearch_next_label": 0,
        "usearch_n_synth": 0,
        "diskann_tmpdir": None,
        "idx_obj": None,
    }

    if system == "faiss_ivf":
        import faiss
        faiss.omp_set_num_threads(threads)
        nlist = int(params.get("nlist", 256))
        quant = faiss.IndexFlatL2(dim)
        idx_obj = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
        idx_obj.train(init_vecs[: min(100_000, len(init_vecs))])
        idx_obj.add_with_ids(init_vecs, init_ids)
        idx_obj.nprobe = int(params.get("nprobe", 16))
        state["idx_obj"] = idx_obj
    elif system == "hnswlib":
        import hnswlib
        max_el = len(init_ids) + n_inserts_total + n_updates_total + 50_000
        if has_churn:
            elem_ids = ev["id"][types_all != EV_QUERY].astype(np.int64)
            max_id = max(int(elem_ids.max()) if elem_ids.size else -1, int(init_ids.max()) if init_ids.size else -1)
            state["usearch_label_base"] = max_id + 1
            state["usearch_next_label"] = max_id + 1
            state["usearch_synth_ids"] = np.empty(n_updates_total + 16, dtype=np.int64)
        idx_obj = hnswlib.Index(space="l2", dim=dim)
        idx_obj.init_index(
            max_elements=max_el,
            ef_construction=int(params.get("ef_construction", 200)),
            M=int(params.get("M", 16)),
            allow_replace_deleted=True,
        )
        idx_obj.set_num_threads(threads)
        idx_obj.add_items(init_vecs, init_ids, num_threads=threads)
        idx_obj.set_ef(int(params.get("ef", 64)))
        state["idx_obj"] = idx_obj
    elif system == "usearch":
        from usearch.index import Index
        if has_churn:
            elem_ids = ev["id"][types_all != EV_QUERY].astype(np.int64)
            max_id = max(int(elem_ids.max()) if elem_ids.size else -1, int(init_ids.max()) if init_ids.size else -1)
            state["usearch_label_base"] = max_id + 1
            state["usearch_next_label"] = max_id + 1
            state["usearch_synth_ids"] = np.empty(n_updates_total + 16, dtype=np.int64)
        idx_obj = Index(
            ndim=dim,
            metric="l2sq",
            dtype="f32",
            connectivity=int(params.get("M", 16)),
            expansion_add=int(params.get("ef_construction", 200)),
            expansion_search=int(params.get("ef", 64)),
        )
        idx_obj.add(init_ids.astype(np.uint64), init_vecs, threads=threads)
        state["idx_obj"] = idx_obj
    elif system == "diskann":
        import diskannpy
        R = int(params.get("R", 32))
        Lbuild = int(params.get("Lbuild", 100))
        search_L = int(params.get("search_L", 64))
        max_vecs = len(init_ids) + n_inserts_total + n_updates_total + 100_000
        diskann_tmpdir = tempfile.TemporaryDirectory(prefix="pvldb_diskann_")
        tags0 = (init_ids + 1).astype(np.uint32)
        diskannpy.build_memory_index(
            init_vecs,
            "l2",
            diskann_tmpdir.name,
            complexity=Lbuild,
            graph_degree=R,
            num_threads=threads,
            tags=tags0,
        )
        idx_obj = diskannpy.DynamicMemoryIndex.from_file(
            diskann_tmpdir.name,
            max_vectors=max_vecs,
            complexity=Lbuild,
            graph_degree=R,
            num_threads=threads,
            initial_search_complexity=search_L,
            search_threads=threads,
            concurrent_consolidation=False,
        )
        state["diskann_tmpdir"] = diskann_tmpdir
        state["idx_obj"] = idx_obj
    elif system == "annoy":
        from annoy import AnnoyIndex
        n_trees = int(params.get("n_trees", 5))
        idx_obj = AnnoyIndex(dim, "euclidean")
        for tid, vref in zip(init_ids.tolist(), init_vrefs.tolist()):
            idx_obj.add_item(int(tid), pool[int(vref)])
        idx_obj.build(n_trees, n_jobs=threads)
        state["idx_obj"] = idx_obj
    else:
        raise ValueError(f"Unknown engine: {system}")

    return state


def _apply_window_crud(
    system: str,
    params: Dict[str, Any],
    state: Dict[str, Any],
    dim: int,
    pool: np.ndarray,
    crud: np.ndarray,
    w: int,
    threads: int,
) -> str:
    ctypes = crud["type"]
    ins_mask = ctypes == EV_INSERT
    upd_mask = ctypes == EV_UPDATE
    ers_mask = ctypes == EV_ERASE

    ins_ids = crud["id"][ins_mask].astype(np.int64)
    ins_vrefs = crud["vec_ref"][ins_mask].astype(np.int64)
    upd_ids = crud["id"][upd_mask].astype(np.int64)
    upd_vrefs = crud["vec_ref"][upd_mask].astype(np.int64)
    ers_ids = crud["id"][ers_mask].astype(np.int64)

    idx_obj = state["idx_obj"]
    live_map = state["live_map"]
    w_status = "ok"

    if system == "faiss_ivf":
        import faiss
        faiss.omp_set_num_threads(threads)
        if ers_ids.size:
            idx_obj.remove_ids(ers_ids)
            for tid in ers_ids.tolist():
                live_map.pop(int(tid), None)
        if upd_ids.size:
            idx_obj.remove_ids(upd_ids)
            idx_obj.add_with_ids(np.ascontiguousarray(pool[upd_vrefs], dtype=np.float32), upd_ids)
            for tid, vr in zip(upd_ids.tolist(), upd_vrefs.tolist()):
                live_map[int(tid)] = int(vr)
        if ins_ids.size:
            idx_obj.add_with_ids(np.ascontiguousarray(pool[ins_vrefs], dtype=np.float32), ins_ids)
            for tid, vr in zip(ins_ids.tolist(), ins_vrefs.tolist()):
                live_map[int(tid)] = int(vr)
        ret_ev = int(params.get("retrain_every", 0))
        if ret_ev > 0 and (w + 1) % ret_ev == 0:
            cur_ids = np.fromiter(live_map.keys(), dtype=np.int64)
            cur_vrefs = np.fromiter(live_map.values(), dtype=np.int64)
            cur_vecs = np.ascontiguousarray(pool[cur_vrefs], dtype=np.float32)
            nlist = int(params.get("nlist", 256))
            quant = faiss.IndexFlatL2(dim)
            idx_obj = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
            idx_obj.train(cur_vecs[: min(100_000, len(cur_vecs))])
            idx_obj.add_with_ids(cur_vecs, cur_ids)
            idx_obj.nprobe = int(params.get("nprobe", 16))
            state["idx_obj"] = idx_obj

    elif system == "hnswlib":
        idx_obj.set_num_threads(threads)
        label_of = state["usearch_label_of"]
        for tid in ers_ids.tolist():
            cur_lab = int(label_of.pop(int(tid), int(tid)))
            idx_obj.mark_deleted(cur_lab)
            state["has_deleted_hnsw"] = True
        if upd_ids.size:
            synth_ids = state["usearch_synth_ids"]
            assert synth_ids is not None
            n_u = upd_ids.size
            for tid in upd_ids.tolist():
                cur_lab = int(label_of.get(int(tid), int(tid)))
                idx_obj.mark_deleted(cur_lab)
            state["has_deleted_hnsw"] = True
            new_labs = np.arange(state["usearch_next_label"], state["usearch_next_label"] + n_u, dtype=np.int64)
            state["usearch_next_label"] += n_u
            lo = state["usearch_n_synth"]
            if lo + n_u > len(synth_ids):
                synth_ids = np.resize(synth_ids, lo + n_u + 1024)
                state["usearch_synth_ids"] = synth_ids
            synth_ids[lo : lo + n_u] = upd_ids
            state["usearch_n_synth"] = lo + n_u
            for tid, nl in zip(upd_ids.tolist(), new_labs.tolist()):
                label_of[int(tid)] = int(nl)
            idx_obj.add_items(
                np.ascontiguousarray(pool[upd_vrefs], dtype=np.float32),
                new_labs,
                num_threads=threads,
                replace_deleted=True,
            )
        if ins_ids.size:
            idx_obj.add_items(
                np.ascontiguousarray(pool[ins_vrefs], dtype=np.float32),
                ins_ids,
                num_threads=threads,
                replace_deleted=state["has_deleted_hnsw"],
            )

    elif system == "usearch":
        if ins_ids.size:
            idx_obj.add(ins_ids.astype(np.uint64), np.ascontiguousarray(pool[ins_vrefs], dtype=np.float32), threads=threads)
        if upd_ids.size:
            synth_ids = state["usearch_synth_ids"]
            label_of = state["usearch_label_of"]
            n_u = upd_ids.size
            old_labs = np.array([int(label_of.get(int(t), int(t))) for t in upd_ids.tolist()], dtype=np.uint64)
            try:
                idx_obj.remove(old_labs)
            except Exception:
                for ol in old_labs.tolist():
                    try:
                        idx_obj.remove(ol)
                    except Exception:
                        pass
            new_labs = np.arange(state["usearch_next_label"], state["usearch_next_label"] + n_u, dtype=np.uint64)
            state["usearch_next_label"] += n_u
            idx_obj.add(new_labs, np.ascontiguousarray(pool[upd_vrefs], dtype=np.float32), threads=threads)
            lo = state["usearch_n_synth"]
            if lo + n_u > len(synth_ids):
                synth_ids = np.resize(synth_ids, lo + n_u + 1024)
                state["usearch_synth_ids"] = synth_ids
            synth_ids[lo : lo + n_u] = upd_ids
            state["usearch_n_synth"] = lo + n_u
            for t, nl in zip(upd_ids.tolist(), new_labs.tolist()):
                label_of[int(t)] = int(nl)
        if ers_ids.size:
            label_of = state["usearch_label_of"]
            cur_labs = np.array([int(label_of.pop(int(t), int(t))) for t in ers_ids.tolist()], dtype=np.uint64)
            try:
                idx_obj.remove(cur_labs)
            except Exception:
                for cl in cur_labs.tolist():
                    try:
                        idx_obj.remove(cl)
                    except Exception:
                        pass

    elif system == "diskann":
        for tid in ers_ids.tolist():
            idx_obj.mark_deleted(np.uint32(int(tid) + 1))
        if upd_ids.size:
            upd_vecs = np.ascontiguousarray(pool[upd_vrefs], dtype=np.float32)
            for tid, vec in zip(upd_ids.tolist(), upd_vecs):
                tag = np.uint32(int(tid) + 1)
                idx_obj.mark_deleted(tag)
                idx_obj.insert(vec, tag)
        if ins_ids.size:
            ins_vecs = np.ascontiguousarray(pool[ins_vrefs], dtype=np.float32)
            for tid, vec in zip(ins_ids.tolist(), ins_vecs):
                idx_obj.insert(vec, np.uint32(int(tid) + 1))
        cons_ev = int(params.get("consolidate_every", 0))
        if cons_ev > 0 and (w + 1) % cons_ev == 0:
            idx_obj.consolidate_delete()

    elif system == "annoy":
        if ers_ids.size or upd_ids.size:
            w_status = "unsupported_erase"
            for tid in ers_ids.tolist():
                live_map.pop(int(tid), None)
            for tid, vr in zip(upd_ids.tolist(), upd_vrefs.tolist()):
                live_map[int(tid)] = int(vr)
        for tid, vr in zip(ins_ids.tolist(), ins_vrefs.tolist()):
            live_map[int(tid)] = int(vr)
        reb_ev = int(params.get("rebuild_every", 4))
        if reb_ev > 0 and (w + 1) % reb_ev == 0:
            from annoy import AnnoyIndex
            idx_obj = AnnoyIndex(dim, "euclidean")
            for tid, vr in live_map.items():
                idx_obj.add_item(int(tid), pool[int(vr)])
            idx_obj.build(int(params.get("n_trees", 5)), n_jobs=threads)
            state["idx_obj"] = idx_obj

    return w_status


def _run_queries(
    system: str,
    params: Dict[str, Any],
    state: Dict[str, Any],
    q_vecs: np.ndarray,
    k: int,
    threads: int,
    batch_query: bool,
) -> Tuple[np.ndarray, List[float]]:
    nq = len(q_vecs)
    pred_ids = np.full((nq, k), -1, dtype=np.int64)
    walls_us: List[float] = []
    idx_obj = state["idx_obj"]

    if batch_query:
        t0 = time.perf_counter_ns()
        if system == "faiss_ivf":
            import faiss
            faiss.omp_set_num_threads(threads)
            idx_obj.nprobe = int(params.get("nprobe", 16))
            _, I = idx_obj.search(q_vecs, k)
            pred_ids[:] = I
        elif system == "hnswlib":
            idx_obj.set_ef(int(params.get("ef", 64)))
            lbls, _ = idx_obj.knn_query(q_vecs, k=k, num_threads=threads)
            raw = np.asarray(lbls, dtype=np.int64)
            if state["usearch_n_synth"] and state["usearch_synth_ids"] is not None:
                base_l = int(state["usearch_label_base"])
                mask = raw >= base_l
                if bool(mask.any()):
                    raw[mask] = state["usearch_synth_ids"][raw[mask] - base_l]
            pred_ids[:] = raw
        elif system == "usearch":
            idx_obj.expansion_search = int(params.get("ef", 64))
            matches = idx_obj.search(q_vecs, k, threads=threads)
            raw_keys = np.asarray(matches.keys, dtype=np.uint64).reshape(nq, -1)[:, :k]
            if state["usearch_n_synth"] and state["usearch_synth_ids"] is not None:
                out = raw_keys.astype(np.int64)
                base_u = np.uint64(state["usearch_label_base"])
                mask = raw_keys >= base_u
                if bool(mask.any()):
                    out[mask] = state["usearch_synth_ids"][(raw_keys[mask] - base_u).astype(np.int64)]
                pred_ids[:, : out.shape[1]] = out
            else:
                pred_ids[:, : raw_keys.shape[1]] = raw_keys.astype(np.int64)
        elif system == "diskann":
            res = idx_obj.batch_search(q_vecs, k, int(params.get("search_L", 64)), threads)
            pred_ids[:] = np.asarray(res.identifiers, dtype=np.int64) - 1
        elif system == "annoy":
            sk = int(params.get("search_k", 8000))
            for i in range(nq):
                r = idx_obj.get_nns_by_vector(q_vecs[i], k, search_k=sk)
                pred_ids[i, : len(r)] = r[:k]
        per_q_us = ((time.perf_counter_ns() - t0) / 1000.0) / max(nq, 1)
        walls_us = [per_q_us]
        return pred_ids, walls_us

    if system == "faiss_ivf":
        import faiss
        faiss.omp_set_num_threads(threads)
        idx_obj.nprobe = int(params.get("nprobe", 16))
        for i in range(nq):
            t0 = time.perf_counter_ns()
            _, I = idx_obj.search(q_vecs[i : i + 1], k)
            walls_us.append((time.perf_counter_ns() - t0) / 1000.0)
            pred_ids[i] = I[0]
    elif system == "hnswlib":
        idx_obj.set_ef(int(params.get("ef", 64)))
        n_synth = state["usearch_n_synth"]
        synth_ids = state["usearch_synth_ids"]
        base_l = int(state["usearch_label_base"])
        for i in range(nq):
            t0 = time.perf_counter_ns()
            lbls, _ = idx_obj.knn_query(q_vecs[i : i + 1], k=k, num_threads=threads)
            walls_us.append((time.perf_counter_ns() - t0) / 1000.0)
            row = np.asarray(lbls[0], dtype=np.int64)
            if n_synth and synth_ids is not None:
                mask = row >= base_l
                if bool(mask.any()):
                    row[mask] = synth_ids[row[mask] - base_l]
            pred_ids[i] = row
    elif system == "usearch":
        idx_obj.expansion_search = int(params.get("ef", 64))
        base_u = np.uint64(state["usearch_label_base"])
        synth_ids = state["usearch_synth_ids"]
        n_synth = state["usearch_n_synth"]
        for i in range(nq):
            t0 = time.perf_counter_ns()
            matches = idx_obj.search(q_vecs[i], k, threads=threads)
            walls_us.append((time.perf_counter_ns() - t0) / 1000.0)
            raw = np.asarray(matches.keys, dtype=np.uint64).reshape(-1)[:k]
            if n_synth and synth_ids is not None:
                out = raw.astype(np.int64)
                mask = raw >= base_u
                if bool(mask.any()):
                    out[mask] = synth_ids[(raw[mask] - base_u).astype(np.int64)]
                pred_ids[i, : out.size] = out
            else:
                pred_ids[i, : raw.size] = raw.astype(np.int64)
    elif system == "diskann":
        sL = int(params.get("search_L", 64))
        for i in range(nq):
            t0 = time.perf_counter_ns()
            res = idx_obj.search(q_vecs[i], k, sL)
            walls_us.append((time.perf_counter_ns() - t0) / 1000.0)
            ids_i = np.asarray(res.identifiers, dtype=np.int64) - 1
            pred_ids[i, : min(k, ids_i.size)] = ids_i[:k]
    elif system == "annoy":
        sk = int(params.get("search_k", 8000))
        for i in range(nq):
            t0 = time.perf_counter_ns()
            r = idx_obj.get_nns_by_vector(q_vecs[i], k, search_k=sk)
            walls_us.append((time.perf_counter_ns() - t0) / 1000.0)
            pred_ids[i, : len(r)] = r[:k]

    return pred_ids, walls_us


# ---------------------------------------------------------------------------
# Single-Engine & Multi-Knob Sweep Workers
# ---------------------------------------------------------------------------
def run_engine_worker(
    data_root: Path,
    trace_name: str,
    system: str,
    param_tag: str,
    params: Dict[str, Any],
    threads: int = 1,
    batch_query: bool = False,
    max_windows: Optional[int] = None,
) -> Dict[str, Any]:
    _set_thread_env(threads)
    load_start = read_load1m()
    t_wall0 = time.perf_counter()
    data = load_qbam_trace(data_root, trace_name)
    n_win = min(data["n_windows"], max_windows) if max_windows else data["n_windows"]

    t_b0 = _cpu_ms()
    state = _build_initial_index(system, params, data["dim"], data["pool"], data["init"], data["ev"], threads)
    build_cpu_ms = _cpu_ms() - t_b0

    q_cpu_sum = 0.0
    m_cpu_sum = 0.0
    recalls: List[float] = []
    p95_list: List[float] = []
    p99_list: List[float] = []
    win_statuses: List[str] = []

    try:
        for w in range(n_win):
            crud, qe = data["win_splits"][w]
            t_m0 = _cpu_ms()
            w_status = _apply_window_crud(system, params, state, data["dim"], data["pool"], crud, w, threads)
            m_cpu_sum += _cpu_ms() - t_m0
            win_statuses.append(w_status)

            q_vecs = np.ascontiguousarray(data["pool"][qe["vec_ref"].astype(np.int64)], dtype=np.float32)
            t_q0 = _cpu_ms()
            pred_ids, walls_us = _run_queries(system, params, state, q_vecs, data["k"], threads, batch_query)
            q_cpu_sum += _cpu_ms() - t_q0

            q_off = data["q_offsets"][w]
            gt_w = data["gt_ids"][q_off : q_off + len(q_vecs)]
            recalls.append(_mean_recall(pred_ids, gt_w, data["k"]))
            sw = sorted(walls_us)
            p95_v, p99_v = _pct(sw, 95.0), _pct(sw, 99.0)
            if p95_v is not None:
                p95_list.append(p95_v)
            if p99_v is not None:
                p99_list.append(p99_v)
    finally:
        if state.get("diskann_tmpdir") is not None:
            state["diskann_tmpdir"].cleanup()

    wall_s = time.perf_counter() - t_wall0
    tot_cpu_ms = q_cpu_sum + m_cpu_sum
    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    att0965 = sum(1 for r in recalls if r >= 0.965)
    att0950 = sum(1 for r in recalls if r >= 0.950)
    status_str = "unsupported_erase" if "unsupported_erase" in win_statuses else "ok"

    return {
        "system": system,
        "param_tag": param_tag,
        "threads": threads,
        "batch_query": batch_query,
        "n_windows": n_win,
        "status": status_str,
        "build_cpu_ms": round(build_cpu_ms, 2),
        "query_cpu_ms": round(q_cpu_sum, 2),
        "maint_cpu_ms": round(m_cpu_sum, 2),
        "total_cpu_ms": round(tot_cpu_ms, 2),
        "wall_s": round(wall_s, 3),
        "wall_cpu_ratio": round((wall_s * 1000.0) / max(tot_cpu_ms + build_cpu_ms, 1.0), 4),
        "mean_recall": round(statistics.fmean(recalls), 4) if recalls else 0.0,
        "min_recall": round(min(recalls), 4) if recalls else 0.0,
        "attained_0965": f"{att0965}/{len(recalls)}",
        "attainment_0965_pct": round(100.0 * att0965 / max(len(recalls), 1), 1),
        "attained_0950": f"{att0950}/{len(recalls)}",
        "attainment_0950_pct": round(100.0 * att0950 / max(len(recalls), 1), 1),
        "mean_p95_us": round(statistics.fmean(p95_list), 2) if p95_list else None,
        "mean_p99_us": round(statistics.fmean(p99_list), 2) if p99_list else None,
        "peak_rss_mb": peak_rss_mb,
        "loadavg_start": round(load_start, 2),
        "loadavg_end": round(read_load1m(), 2),
        "params_json": json.dumps(params, sort_keys=True),
    }


def run_a4_sweep_worker(
    data_root: Path,
    trace_name: str,
    system: str,
    knobs: List[Tuple[str, Dict[str, Any]]],
    max_windows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate read-only query-parameter sweeps on one shared index build/insert lifecycle."""
    _set_thread_env(1)
    load_start = read_load1m()
    t_load0 = time.perf_counter()
    data = load_qbam_trace(data_root, trace_name)
    load_wall_s = time.perf_counter() - t_load0
    n_win = min(data["n_windows"], max_windows) if max_windows else data["n_windows"]

    base_params = knobs[0][1]
    t_bw0 = time.perf_counter()
    t_bc0 = _cpu_ms()
    state = _build_initial_index(system, base_params, data["dim"], data["pool"], data["init"], data["ev"], 1)
    build_cpu_ms = _cpu_ms() - t_bc0
    build_wall_ms = (time.perf_counter() - t_bw0) * 1000.0

    maint_cpu_sum = 0.0
    maint_wall_ms = 0.0
    n_k = len(knobs)
    q_cpu_sums = [0.0] * n_k
    q_wall_ms = [0.0] * n_k
    recalls_k: List[List[float]] = [[] for _ in range(n_k)]
    p95_k: List[List[float]] = [[] for _ in range(n_k)]
    p99_k: List[List[float]] = [[] for _ in range(n_k)]

    try:
        for w in range(n_win):
            crud, qe = data["win_splits"][w]
            tw0 = time.perf_counter()
            tc0 = _cpu_ms()
            _apply_window_crud(system, base_params, state, data["dim"], data["pool"], crud, w, 1)
            maint_cpu_sum += _cpu_ms() - tc0
            maint_wall_ms += (time.perf_counter() - tw0) * 1000.0

            q_vecs = np.ascontiguousarray(data["pool"][qe["vec_ref"].astype(np.int64)], dtype=np.float32)
            q_off = data["q_offsets"][w]
            gt_w = data["gt_ids"][q_off : q_off + len(q_vecs)]

            for ki, (_, kparams) in enumerate(knobs):
                tqw0 = time.perf_counter()
                tqc0 = _cpu_ms()
                pred_ids, walls_us = _run_queries(system, kparams, state, q_vecs, data["k"], 1, False)
                q_cpu_sums[ki] += _cpu_ms() - tqc0
                q_wall_ms[ki] += (time.perf_counter() - tqw0) * 1000.0
                recalls_k[ki].append(_mean_recall(pred_ids, gt_w, data["k"]))
                sw = sorted(walls_us)
                p95_v, p99_v = _pct(sw, 95.0), _pct(sw, 99.0)
                if p95_v is not None:
                    p95_k[ki].append(p95_v)
                if p99_v is not None:
                    p99_k[ki].append(p99_v)
    finally:
        if state.get("diskann_tmpdir") is not None:
            state["diskann_tmpdir"].cleanup()

    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    load_end = round(read_load1m(), 2)
    results: List[Dict[str, Any]] = []
    for ki, (ptag, kparams) in enumerate(knobs):
        tot_cpu_ms = q_cpu_sums[ki] + maint_cpu_sum
        tot_wall_ms = (load_wall_s * 1000.0) + build_wall_ms + maint_wall_ms + q_wall_ms[ki]
        recs = recalls_k[ki]
        att0965 = sum(1 for r in recs if r >= 0.965)
        att0950 = sum(1 for r in recs if r >= 0.950)
        results.append({
            "system": system,
            "param_tag": ptag,
            "threads": 1,
            "batch_query": False,
            "n_windows": n_win,
            "status": "ok",
            "build_cpu_ms": round(build_cpu_ms, 2),
            "query_cpu_ms": round(q_cpu_sums[ki], 2),
            "maint_cpu_ms": round(maint_cpu_sum, 2),
            "total_cpu_ms": round(tot_cpu_ms, 2),
            "wall_s": round(tot_wall_ms / 1000.0, 3),
            "wall_cpu_ratio": round(tot_wall_ms / max(tot_cpu_ms + build_cpu_ms, 1.0), 4),
            "mean_recall": round(statistics.fmean(recs), 4) if recs else 0.0,
            "min_recall": round(min(recs), 4) if recs else 0.0,
            "attained_0965": f"{att0965}/{len(recs)}",
            "attainment_0965_pct": round(100.0 * att0965 / max(len(recs), 1), 1),
            "attained_0950": f"{att0950}/{len(recs)}",
            "attainment_0950_pct": round(100.0 * att0950 / max(len(recs), 1), 1),
            "mean_p95_us": round(statistics.fmean(p95_k[ki]), 2) if p95_k[ki] else None,
            "mean_p99_us": round(statistics.fmean(p99_k[ki]), 2) if p99_k[ki] else None,
            "peak_rss_mb": peak_rss_mb,
            "loadavg_start": round(load_start, 2),
            "loadavg_end": load_end,
            "params_json": json.dumps(kparams, sort_keys=True),
        })
    return results


# ---------------------------------------------------------------------------
# Subprocess Launchers & Hygiene Retry
# ---------------------------------------------------------------------------
def _cache_key(
    workload: str,
    trace: str,
    system: str,
    param_tag: str,
    cores: Optional[str],
    threads: int,
    batch_query: bool,
    max_windows: Optional[int],
    regime: str = "isolated",
) -> str:
    raw = f"{regime}__{workload}__{trace}__{system}__{param_tag}__c={cores or 'unpinned'}__t={threads}__bq={int(batch_query)}__w={max_windows or 'all'}"
    return raw.replace(",", "_")


def invoke_worker(
    cache_dir: Path,
    workload: str,
    data_root: Path,
    trace: str,
    system: str,
    param_tag: str,
    params: Dict[str, Any],
    cores: Optional[str],
    threads: int = 1,
    batch_query: bool = False,
    max_windows: Optional[int] = None,
    worker_id: int = 0,
    timeout_s: int = 2400,
    reuse_cache: bool = True,
    enforce_clean_iso: bool = False,
    regime: str = "isolated",
) -> Dict[str, Any]:
    key = _cache_key(workload, trace, system, param_tag, cores, threads, batch_query, max_windows, regime=regime) + f"__wid{worker_id}"
    cache_file = cache_dir / f"{key}.json"
    if reuse_cache and cache_file.exists():
        try:
            obj = json.loads(cache_file.read_text(encoding="utf-8"))
            if obj.get("wall_cpu_ratio") is not None and (not enforce_clean_iso or float(obj["wall_cpu_ratio"]) <= 1.18):
                return obj
        except Exception:
            pass

    core_candidates = [cores]
    if enforce_clean_iso and cores == "0":
        core_candidates = ["0", "0", "32"]

    last_res: Optional[Dict[str, Any]] = None
    for attempt, try_core in enumerate(core_candidates):
        cmd: List[str] = []
        if try_core is not None and shutil.which("taskset"):
            cmd.extend(["taskset", "-c", str(try_core)])
        cmd.extend([
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--data-root", str(data_root),
            "--trace", trace,
            "--system", system,
            "--param-tag", param_tag,
            "--params-json", json.dumps(params, sort_keys=True),
            "--threads", str(threads),
        ])
        if batch_query:
            cmd.append("--batch-query")
        if max_windows is not None:
            cmd.extend(["--max-windows", str(max_windows)])

        env = os.environ.copy()
        for k_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k_env] = str(threads)

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Worker failed ({key}): {proc.stderr[-400:]}")
        res = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("WORKER_JSON="):
                res = json.loads(line.split("=", 1)[1])
                break
        if res is None:
            raise RuntimeError(f"Missing WORKER_JSON ({key}): {proc.stdout[-300:]}")

        res["cores_pinned"] = try_core or "unpinned"
        res["workload"] = workload
        res["trace"] = trace
        last_res = res
        if not enforce_clean_iso or float(res["wall_cpu_ratio"]) <= 1.18:
            break
        log(f"  [retry {attempt + 1}] {key} on core={try_core} got wall_cpu_ratio={res['wall_cpu_ratio']} > 1.18; retrying...")
        time.sleep(5.0)

    assert last_res is not None
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(last_res, indent=2) + "\n", encoding="utf-8")
    return last_res


def invoke_a4_sweep(
    cache_dir: Path,
    data_root: Path,
    trace: str,
    system: str,
    knobs: List[Tuple[str, Dict[str, Any]]],
    max_windows: Optional[int],
) -> List[Dict[str, Any]]:
    cache_file = cache_dir / f"a4sweep__{trace}__{system}__w={max_windows or 'all'}.json"
    if cache_file.exists():
        try:
            arr = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(arr, list) and len(arr) == len(knobs) and all(float(r["wall_cpu_ratio"]) <= 1.18 for r in arr):
                return arr
        except Exception:
            pass

    for attempt, try_core in enumerate(["0", "0", "32"]):
        cmd = [
            "taskset", "-c", try_core,
            sys.executable, str(Path(__file__).resolve()),
            "--a4-sweep-worker",
            "--data-root", str(data_root),
            "--trace", trace,
            "--system", system,
            "--params-json", json.dumps(knobs),
        ]
        if max_windows is not None:
            cmd.extend(["--max-windows", str(max_windows)])
        env = os.environ.copy()
        for k_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k_env] = "1"
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"A4 sweep failed ({trace}/{system}): {proc.stderr[-400:]}")
        arr = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("A4_SWEEP_JSON="):
                arr = json.loads(line.split("=", 1)[1])
                break
        if arr is None:
            raise RuntimeError(f"Missing A4_SWEEP_JSON ({trace}/{system})")
        for r in arr:
            r["cores_pinned"] = try_core
            r["workload"] = "sift100k_drift"
            r["trace"] = trace
        if all(float(r["wall_cpu_ratio"]) <= 1.18 for r in arr):
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(arr, indent=2) + "\n", encoding="utf-8")
            return arr
        log(f"  [A4 retry {attempt + 1}] {trace}/{system} on core={try_core} max_ratio={max(r['wall_cpu_ratio'] for r in arr)} > 1.18")
        time.sleep(5.0)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(arr, indent=2) + "\n", encoding="utf-8")
    return arr


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def compute_ranking_flips(
    scores_base: Dict[str, float],
    scores_alt: Dict[str, float],
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    from scipy.stats import kendalltau

    keys = sorted(set(scores_base.keys()) & set(scores_alt.keys()))
    n = len(keys)
    if n < 2:
        return {"n": n, "kendall_tau": None, "pairwise_flip_pct": None, "top1_flip": None}
    v_a = [scores_base[k] for k in keys]
    v_b = [scores_alt[k] for k in keys]
    tau, pval = kendalltau(v_a, v_b)
    flips = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = v_a[i] - v_a[j]
            db = v_b[i] - v_b[j]
            if abs(da) <= 1e-12 or abs(db) <= 1e-12:
                continue
            pairs += 1
            if da * db < 0:
                flips += 1
    w_a = min(keys, key=lambda k: scores_base[k]) if lower_is_better else max(keys, key=lambda k: scores_base[k])
    w_b = min(keys, key=lambda k: scores_alt[k]) if lower_is_better else max(keys, key=lambda k: scores_alt[k])
    return {
        "n": n,
        "kendall_tau": round(float(tau), 4),
        "p_value": round(float(pval), 4),
        "pairwise_flips": flips,
        "total_pairs": pairs,
        "pairwise_flip_pct": round(100.0 * flips / max(pairs, 1), 1),
        "winner_base": w_a,
        "winner_alt": w_b,
        "top1_flip": bool(w_a != w_b),
    }


# ---------------------------------------------------------------------------
# Suite A1+D: Real-Engine Trap 4 & MT NUMA
# ---------------------------------------------------------------------------
def run_a1d(out_dir: Path, cache_dir: Path, max_load: float, max_windows: int) -> None:
    log(f"=== Starting Suite A1+D (Real-Engine Trap 4 & MT NUMA, max_windows={max_windows}) ===")
    data_root = DATA_ROOTS_LLM01["drift"]
    traces = ["abrupt_s05", "hotspot_s05"]
    engines = [
        ("faiss_ivf", "np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}),
        ("hnswlib", "ef128", {"M": 16, "ef_construction": 200, "ef": 128}),
        ("usearch", "ef128", {"M": 16, "ef_construction": 200, "ef": 128}),
        ("diskann", "L64", {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": 0}),
        ("annoy", "sk8000_re4", {"n_trees": 5, "search_k": 8000, "rebuild_every": 4}),
    ]
    regimes = [
        ("iso_numa0_1w", ["0"], 1, False),
        ("ht_sibling_2w", ["0", "64"], 1, False),
        ("same_numa_4w", ["0", "1", "2", "3"], 1, False),
        ("cross_numa_8w", ["0", "1", "2", "3", "32", "33", "34", "35"], 1, False),
        ("iso_numa0_1t_batch", ["0"], 1, True),
        ("mt_same_numa_4t", ["0-3"], 4, True),
        ("mt_unpinned_8t", [None], 8, True),
    ]

    raw_rows: List[Dict[str, Any]] = []
    for trace in traces:
        for sys_name, ptag, params in engines:
            for reg_name, core_list, n_thr, bq in regimes:
                is_iso = len(core_list) == 1 and n_thr == 1
                all_cached = all(
                    (cache_dir / f"{_cache_key('sift100k_drift', trace, sys_name, ptag, c, n_thr, bq, max_windows, regime=reg_name)}__wid{wid}.json").exists()
                    for wid, c in enumerate(core_list)
                )
                if not all_cached:
                    wait_quiet_host(max_load)
                log(f"[A1+D] {trace} / {sys_name}_{ptag} / {reg_name} (ways={len(core_list)}, thr={n_thr}, batch={bq})")

                def _launch(item: Tuple[int, Optional[str]]) -> Dict[str, Any]:
                    wid, core_str = item
                    return invoke_worker(
                        cache_dir, "sift100k_drift", data_root, trace, sys_name, ptag, params,
                        cores=core_str, threads=n_thr, batch_query=bq,
                        max_windows=max_windows, worker_id=wid, enforce_clean_iso=is_iso, regime=reg_name,
                    )

                if len(core_list) == 1:
                    res_list = [_launch((0, core_list[0]))]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(core_list)) as ex:
                        res_list = list(ex.map(_launch, list(enumerate(core_list))))

                for wid, r in enumerate(res_list):
                    raw_rows.append({"suite": "A1D", "regime": reg_name, "ways": len(core_list), "worker_id": wid, **r})
                write_csv(raw_rows, out_dir / "a1d_raw_runs.csv")

    import pandas as pd
    df = pd.DataFrame(raw_rows)
    agg = df.groupby(["trace", "regime", "ways", "threads", "batch_query", "system", "param_tag"], as_index=False).agg(
        total_cpu_ms=("total_cpu_ms", "mean"),
        query_cpu_ms=("query_cpu_ms", "mean"),
        maint_cpu_ms=("maint_cpu_ms", "mean"),
        wall_s=("wall_s", "mean"),
        wall_cpu_ratio=("wall_cpu_ratio", "mean"),
        mean_recall=("mean_recall", "mean"),
        attainment_0965_pct=("attainment_0965_pct", "mean"),
        attainment_0950_pct=("attainment_0950_pct", "mean"),
        mean_p95_us=("mean_p95_us", "mean"),
        mean_p99_us=("mean_p99_us", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
    )

    base_1w = agg[agg["regime"] == "iso_numa0_1w"].set_index(["trace", "system"])
    base_1tb = agg[agg["regime"] == "iso_numa0_1t_batch"].set_index(["trace", "system"])
    cpu_inf_list, wall_inf_list, p99_inf_list = [], [], []
    for _, row in agg.iterrows():
        ref_tbl = base_1tb if row["batch_query"] else base_1w
        key = (row["trace"], row["system"])
        if key in ref_tbl.index:
            ref = ref_tbl.loc[key]
            cpu_inf_list.append(round(float(row["total_cpu_ms"]) / max(float(ref["total_cpu_ms"]), 1.0), 4))
            wall_inf_list.append(round(float(row["wall_s"]) / max(float(ref["wall_s"]), 1e-3), 4))
            p99_inf_list.append(round(float(row["mean_p99_us"]) / float(ref["mean_p99_us"]), 4) if (row["mean_p99_us"] and ref["mean_p99_us"]) else None)
        else:
            cpu_inf_list.append(None)
            wall_inf_list.append(None)
            p99_inf_list.append(None)
    agg["cpu_inflation_vs_iso"] = cpu_inf_list
    agg["wall_ratio_vs_iso"] = wall_inf_list
    agg["p99_inflation_vs_iso"] = p99_inf_list
    agg.to_csv(out_dir / "a1d_summary.csv", index=False)

    flip_summary: Dict[str, Any] = {"per_trace": {}, "mean_across_traces": {}}
    mean_reg_sys = agg.groupby(["regime", "system"], as_index=False)[["total_cpu_ms", "wall_s", "mean_p99_us"]].mean()
    base_map_cpu = {r["system"]: float(r["total_cpu_ms"]) for _, r in mean_reg_sys[mean_reg_sys["regime"] == "iso_numa0_1w"].iterrows()}
    base_map_wall = {r["system"]: float(r["wall_s"]) for _, r in mean_reg_sys[mean_reg_sys["regime"] == "iso_numa0_1w"].iterrows()}
    for reg_name in ("ht_sibling_2w", "same_numa_4w", "cross_numa_8w"):
        sub = mean_reg_sys[mean_reg_sys["regime"] == reg_name]
        flip_summary["mean_across_traces"][f"{reg_name}_vs_iso1w_cpu"] = compute_ranking_flips(
            base_map_cpu, {r["system"]: float(r["total_cpu_ms"]) for _, r in sub.iterrows()}, True
        )
        flip_summary["mean_across_traces"][f"{reg_name}_vs_iso1w_wall"] = compute_ranking_flips(
            base_map_wall, {r["system"]: float(r["wall_s"]) for _, r in sub.iterrows()}, True
        )
    base_bt_wall = {r["system"]: float(r["wall_s"]) for _, r in mean_reg_sys[mean_reg_sys["regime"] == "iso_numa0_1t_batch"].iterrows()}
    mt4_wall = {r["system"]: float(r["wall_s"]) for _, r in mean_reg_sys[mean_reg_sys["regime"] == "mt_same_numa_4t"].iterrows()}
    mt8_wall = {r["system"]: float(r["wall_s"]) for _, r in mean_reg_sys[mean_reg_sys["regime"] == "mt_unpinned_8t"].iterrows()}
    mt4_cpu = {r["system"]: float(r["total_cpu_ms"]) for _, r in mean_reg_sys[mean_reg_sys["regime"] == "mt_same_numa_4t"].iterrows()}
    mt8_cpu = {r["system"]: float(r["total_cpu_ms"]) for _, r in mean_reg_sys[mean_reg_sys["regime"] == "mt_unpinned_8t"].iterrows()}
    flip_summary["mean_across_traces"]["mt4_vs_iso1t_wall"] = compute_ranking_flips(base_bt_wall, mt4_wall, True)
    flip_summary["mean_across_traces"]["mt8_vs_iso1t_wall"] = compute_ranking_flips(base_bt_wall, mt8_wall, True)
    flip_summary["mean_across_traces"]["mt8_vs_mt4_wall"] = compute_ranking_flips(mt4_wall, mt8_wall, True)
    flip_summary["mean_across_traces"]["mt8_vs_mt4_cpu"] = compute_ranking_flips(mt4_cpu, mt8_cpu, True)

    for tr in traces:
        sub_tr = agg[agg["trace"] == tr]
        b_cpu = {r["system"]: float(r["total_cpu_ms"]) for _, r in sub_tr[sub_tr["regime"] == "iso_numa0_1w"].iterrows()}
        flip_summary["per_trace"][tr] = {}
        for reg_name in ("ht_sibling_2w", "same_numa_4w", "cross_numa_8w"):
            a_cpu = {r["system"]: float(r["total_cpu_ms"]) for _, r in sub_tr[sub_tr["regime"] == reg_name].iterrows()}
            flip_summary["per_trace"][tr][f"{reg_name}_vs_iso1w_cpu"] = compute_ranking_flips(b_cpu, a_cpu, True)
        m4_w = {r["system"]: float(r["wall_s"]) for _, r in sub_tr[sub_tr["regime"] == "mt_same_numa_4t"].iterrows()}
        m8_w = {r["system"]: float(r["wall_s"]) for _, r in sub_tr[sub_tr["regime"] == "mt_unpinned_8t"].iterrows()}
        flip_summary["per_trace"][tr]["mt8_vs_mt4_wall"] = compute_ranking_flips(m4_w, m8_w, True)

    (out_dir / "a1d_ranking_flips.json").write_text(json.dumps(flip_summary, indent=2) + "\n", encoding="utf-8")
    log(f"A1+D complete: {len(raw_rows)} rows -> a1d_summary.csv & a1d_ranking_flips.json")


# ---------------------------------------------------------------------------
# Suite A2: Real-Engine Trap 1 Cadence Brackets
# ---------------------------------------------------------------------------
def run_a2(out_dir: Path, cache_dir: Path, max_load: float, max_windows: Optional[int]) -> None:
    log("=== Starting Suite A2 (Real-Engine Trap 1 Cadence Brackets) ===")
    jobs: List[Tuple[str, Path, str, str, str, int, Dict[str, Any]]] = []
    for tr in ("abrupt_s05", "hotspot_s05", "gradual_s05"):
        for cad in (0, 1, 2, 4, 8, 16):
            jobs.append(("sift100k_drift", DATA_ROOTS_LLM01["drift"], tr, "faiss_ivf", f"np16_ret{cad}", cad, {"nlist": 256, "nprobe": 16, "retrain_every": cad}))

    for tr in ("abrupt_s05", "hotspot_s05"):
        for cad in (1, 2, 3, 4, 6, 8):
            jobs.append(("sift100k_drift", DATA_ROOTS_LLM01["drift"], tr, "annoy", f"sk8000_re{cad}", cad, {"n_trees": 5, "search_k": 8000, "rebuild_every": cad}))

    for cad in (0, 1, 2, 4, 8):
        jobs.append(("crud30v2", DATA_ROOTS_LLM01["crud30v2"], "abrupt_s05", "diskann", f"L64_cons{cad}", cad, {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": cad}))
    for cad in (1, 2, 4, 8):
        jobs.append(("sift100k_drift", DATA_ROOTS_LLM01["drift"], "abrupt_s05", "diskann", f"L64_cons{cad}", cad, {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": cad}))

    rows: List[Dict[str, Any]] = []
    for idx, (wl, droot, tr, sys_name, ptag, cad, params) in enumerate(jobs, 1):
        key = _cache_key(wl, tr, sys_name, ptag, "0", 1, False, max_windows) + "__wid0"
        if not (cache_dir / f"{key}.json").exists():
            wait_quiet_host(max_load)
        log(f"[A2 {idx}/{len(jobs)}] {wl}/{tr} {sys_name} {ptag}")
        r = invoke_worker(
            cache_dir, wl, droot, tr, sys_name, ptag, params,
            cores="0", threads=1, batch_query=False, max_windows=max_windows, enforce_clean_iso=True,
        )
        rows.append({"suite": "A2", "cadence": cad, **r})
        write_csv(rows, out_dir / "a2_cadence_brackets.csv")

    import pandas as pd
    df = pd.DataFrame(rows)
    bracket_checks: List[Dict[str, Any]] = []
    for (wl, tr, sys_name), grp in df.groupby(["workload", "trace", "system"]):
        grp_pos = grp[grp["cadence"] > 0].sort_values("cadence")
        if len(grp_pos) < 3:
            continue
        c_min, c_max = int(grp_pos["cadence"].min()), int(grp_pos["cadence"].max())
        s_min = float(grp_pos[grp_pos["cadence"] == c_min]["maint_cpu_ms"].iloc[0])
        s_max = float(grp_pos[grp_pos["cadence"] == c_max]["maint_cpu_ms"].iloc[0])
        spends_dict = {f"c{int(r['cadence'])}": float(r["maint_cpu_ms"]) for _, r in grp_pos.iterrows()}
        for _, r in grp_pos.iterrows():
            c = int(r["cadence"])
            if c in (c_min, c_max):
                continue
            inv_frac = (1.0 / c - 1.0 / c_max) / max(1.0 / c_min - 1.0 / c_max, 1e-9)
            interp_maint = s_max + inv_frac * (s_min - s_max)
            actual_maint = float(r["maint_cpu_ms"])
            rel_err_pct = 100.0 * (interp_maint - actual_maint) / max(actual_maint, 1e-6)
            br = ah.check_cadence_bracket(actual_maint, {k: v for k, v in spends_dict.items() if k != f"c{c}"})
            bracket_checks.append({
                "workload": wl,
                "trace": tr,
                "system": sys_name,
                "cadence": c,
                "actual_maint_cpu_ms": round(actual_maint, 2),
                "linear_inv_interp_maint_ms": round(interp_maint, 2),
                "interp_err_pct": round(rel_err_pct, 2),
                "actual_total_cpu_ms": float(r["total_cpu_ms"]),
                "mean_recall": float(r["mean_recall"]),
                "bracketed_by_neighbors": br.get("bracketed"),
            })
    write_csv(bracket_checks, out_dir / "a2_interp_vs_executed.csv")
    log(f"A2 complete: {len(rows)} runs -> a2_cadence_brackets.csv & a2_interp_vs_executed.csv")


# ---------------------------------------------------------------------------
# Suite A3: Real-Engine Trap 3 CRUD + FreshDiskANN CRUD
# ---------------------------------------------------------------------------
def run_a3(out_dir: Path, cache_dir: Path, max_load: float, max_windows: Optional[int]) -> None:
    log("=== Starting Suite A3 (Real-Engine Trap 3 CRUD + FreshDiskANN CRUD) ===")
    suites = [
        ("crud30v2", DATA_ROOTS_LLM01["crud30v2"], ["abrupt_s05", "hotspot_s05"]),
        ("crud50v2", DATA_ROOTS_LLM01["crud50v2"], ["abrupt_s05", "hotspot_s05"]),
        ("runbook", DATA_ROOTS_LLM01["runbook"], ["clustered_s05", "sliding_s05"]),
    ]
    engines = [
        ("diskann", "L64_cons0", {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": 0}),
        ("diskann", "L64_cons1", {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": 1}),
        ("faiss_ivf", "np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}),
        ("hnswlib", "ef64", {"M": 16, "ef_construction": 200, "ef": 64}),
        ("usearch", "ef64", {"M": 16, "ef_construction": 200, "ef": 64}),
        ("annoy", "sk8000_re4", {"n_trees": 5, "search_k": 8000, "rebuild_every": 4}),
    ]
    rows: List[Dict[str, Any]] = []
    total = sum(len(trs) for _, _, trs in suites) * len(engines)
    idx = 0
    for wl, droot, tr_list in suites:
        for tr in tr_list:
            for sys_name, ptag, params in engines:
                idx += 1
                key = _cache_key(wl, tr, sys_name, ptag, "0", 1, False, max_windows) + "__wid0"
                if not (cache_dir / f"{key}.json").exists():
                    wait_quiet_host(max_load)
                log(f"[A3 {idx}/{total}] {wl}/{tr} {sys_name} {ptag}")
                r = invoke_worker(
                    cache_dir, wl, droot, tr, sys_name, ptag, params,
                    cores="0", threads=1, batch_query=False, max_windows=max_windows, enforce_clean_iso=True,
                )
                rows.append({"suite": "A3", **r})
                write_csv(rows, out_dir / "a3_crud_engines.csv")
    log(f"A3 complete: {len(rows)} runs -> a3_crud_engines.csv")


# ---------------------------------------------------------------------------
# Suite A4: Recall-Matched >=95% Attainment Floor
# ---------------------------------------------------------------------------
def run_a4(out_dir: Path, cache_dir: Path, max_load: float, max_windows: Optional[int]) -> None:
    log("=== Starting Suite A4 (Recall-Matched >=95% Attainment Floor) ===")
    data_root = DATA_ROOTS_LLM01["drift"]
    traces = ["abrupt_s05", "hotspot_s05", "gradual_s05"]
    engine_sweeps: List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]] = [
        ("faiss_ivf", [
            ("np8", {"nlist": 256, "nprobe": 8, "retrain_every": 0}),
            ("np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}),
            ("np24", {"nlist": 256, "nprobe": 24, "retrain_every": 0}),
            ("np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}),
            ("np48", {"nlist": 256, "nprobe": 48, "retrain_every": 0}),
        ]),
        ("hnswlib", [
            ("ef32", {"M": 16, "ef_construction": 200, "ef": 32}),
            ("ef64", {"M": 16, "ef_construction": 200, "ef": 64}),
            ("ef96", {"M": 16, "ef_construction": 200, "ef": 96}),
            ("ef128", {"M": 16, "ef_construction": 200, "ef": 128}),
        ]),
        ("usearch", [
            ("ef32", {"M": 16, "ef_construction": 200, "ef": 32}),
            ("ef64", {"M": 16, "ef_construction": 200, "ef": 64}),
            ("ef96", {"M": 16, "ef_construction": 200, "ef": 96}),
            ("ef128", {"M": 16, "ef_construction": 200, "ef": 128}),
        ]),
        ("diskann", [
            ("L32", {"R": 32, "Lbuild": 100, "search_L": 32, "consolidate_every": 0}),
            ("L64", {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": 0}),
            ("L96", {"R": 32, "Lbuild": 100, "search_L": 96, "consolidate_every": 0}),
            ("L128", {"R": 32, "Lbuild": 100, "search_L": 128, "consolidate_every": 0}),
        ]),
    ]

    rows: List[Dict[str, Any]] = []
    total = len(traces) * len(engine_sweeps)
    idx = 0
    for tr in traces:
        for sys_name, knobs in engine_sweeps:
            idx += 1
            wait_quiet_host(max_load)
            log(f"[A4 {idx}/{total}] {tr} {sys_name} ({len(knobs)} knobs)")
            res_list = invoke_a4_sweep(cache_dir, data_root, tr, sys_name, knobs, max_windows)
            for r in res_list:
                rows.append({"suite": "A4", **r})
            write_csv(rows, out_dir / "a4_attainment_sweep.csv")

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby(["system", "param_tag"], as_index=False).agg(
        n_traces=("trace", "count"),
        query_cpu_ms=("query_cpu_ms", "mean"),
        maint_cpu_ms=("maint_cpu_ms", "mean"),
        total_cpu_ms=("total_cpu_ms", "mean"),
        mean_recall=("mean_recall", "mean"),
        min_recall=("min_recall", "min"),
        attainment_0950_pct=("attainment_0950_pct", "mean"),
        attainment_0965_pct=("attainment_0965_pct", "mean"),
        mean_p95_us=("mean_p95_us", "mean"),
        mean_p99_us=("mean_p99_us", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
        max_wall_cpu_ratio=("wall_cpu_ratio", "max"),
    )
    agg.to_csv(out_dir / "a4_summary_by_config.csv", index=False)

    floor_rows: List[Dict[str, Any]] = []
    for target_col, target_name in [("attainment_0965_pct", "recall_0.965"), ("attainment_0950_pct", "recall_0.950")]:
        for sys_name, grp in agg.groupby("system"):
            passing = grp[grp[target_col] >= 95.0].sort_values("total_cpu_ms")
            if not passing.empty:
                best = passing.iloc[0]
                floor_rows.append({"target": target_name, "hit_95pct_floor": True, **best.to_dict()})
            else:
                best = grp.sort_values(target_col, ascending=False).iloc[0]
                floor_rows.append({"target": target_name, "hit_95pct_floor": False, **best.to_dict()})
    write_csv(floor_rows, out_dir / "a4_min_cpu_floor.csv")
    log(f"A4 complete: {len(rows)} rows -> a4_attainment_sweep.csv & a4_min_cpu_floor.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="PVLDB Real-Engine Hygiene Traps on Xeon-8352V")
    ap.add_argument("--suite", default="ALL", choices=["ALL", "A1D", "A2", "A3", "A4"])
    ap.add_argument("--out-dir", default=os.environ.get("DYNVECAUDIT_RESULTS_DIR", "./results"))
    ap.add_argument("--max-load", type=float, default=25.0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--a1d-windows", type=int, default=10)

    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--a4-sweep-worker", action="store_true")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--system", default="")
    ap.add_argument("--param-tag", default="")
    ap.add_argument("--params-json", default="{}")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch-query", action="store_true")
    args = ap.parse_args()

    max_w = args.max_windows if args.max_windows > 0 else None

    if args.worker:
        res = run_engine_worker(
            Path(args.data_root),
            args.trace,
            args.system,
            args.param_tag,
            json.loads(args.params_json),
            threads=args.threads,
            batch_query=args.batch_query,
            max_windows=max_w,
        )
        print("WORKER_JSON=" + json.dumps(res), flush=True)
        return 0

    if args.a4_sweep_worker:
        knobs = [(str(k), dict(v)) for k, v in json.loads(args.params_json)]
        arr = run_a4_sweep_worker(Path(args.data_root), args.trace, args.system, knobs, max_windows=max_w)
        print("A4_SWEEP_JSON=" + json.dumps(arr), flush=True)
        return 0

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.suite in ("ALL", "A1D"):
        run_a1d(out_dir, cache_dir, args.max_load, args.a1d_windows)
    if args.suite in ("ALL", "A4"):
        run_a4(out_dir, cache_dir, args.max_load, max_w)
    if args.suite in ("ALL", "A2"):
        run_a2(out_dir, cache_dir, args.max_load, max_w)
    if args.suite in ("ALL", "A3"):
        run_a3(out_dir, cache_dir, args.max_load, max_w)

    isolated_checks = []
    for fname in ("a1d_raw_runs.csv", "a2_cadence_brackets.csv", "a3_crud_engines.csv", "a4_attainment_sweep.csv"):
        fp = out_dir / fname
        if not fp.exists():
            continue
        import pandas as pd
        df = pd.read_csv(fp)
        mask = df["threads"] == 1
        if "ways" in df.columns:
            mask = mask & (df["ways"] == 1)
        sub = df[mask]
        max_wc = float(sub["wall_cpu_ratio"].max()) if not sub.empty else None
        n_bad = int((sub["wall_cpu_ratio"] > 1.18).sum()) if not sub.empty else 0
        isolated_checks.append({"file": fname, "n_isolated": int(len(sub)), "max_wall_cpu_ratio": max_wc, "n_gt_1_18": n_bad})
    (out_dir / "hygiene_gate_summary.json").write_text(json.dumps(isolated_checks, indent=2) + "\n", encoding="utf-8")
    log("ALL SELECTED SUITES COMPLETE. Hygiene summary: " + json.dumps(isolated_checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
