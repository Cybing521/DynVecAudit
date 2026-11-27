#!/usr/bin/env python3
"""GPU Real-Timestamp arXiv CS Stream (B3) & 1M Multi-Engine Audit (B1) on Cloud GPU.

Runs on a cloud GPU node (L4/A100) using:
  - System CUDA Python for GPU MiniLM-384 encoding (`all-MiniLM-L6-v2`) and
    exact PyTorch FP32 GEMM Top-10 Ground Truth generation.
  - Python 3.11 venv for isolated single-core
    (`taskset -c 0`, `OMP_NUM_THREADS=1`) execution across ALL 5 engines:
    FAISS-IVF, hnswlib, USearch, FreshDiskANN (`diskannpy==0.7.0`), and Annoy.

Strict Gate-4 / Hygiene Guard:
  Cloud vCPU wall/CPU timings are NEVER exported for Gate-4 CPU timing tables.
  Summary CSVs report strictly:
    - GPU-verified Recall@10 (mean, min, >=0.965 & >=0.950 attainment %)
    - Isolated per-engine Peak RSS (MB)
    - Within-run relative CPU split ratios (`query_cpu_pct`, `maint_cpu_pct`,
      `maint_to_query_ratio`, `rel_cpu_vs_faiss_np16`)
    - Hardware-invariant `exact_scan_ops`, Theorem-1 MILP GCR, and Gate 2 Spearman audit.
"""
from __future__ import annotations

import argparse
import ast
import csv
import email.utils
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

K_BLOCK = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cpu_ms_now() -> float:
    return time.process_time() * 1000.0


def set_single_thread_env() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def get_peak_rss_mb() -> float:
    """Return true per-process Peak RSS (VmHWM on Linux/gVisor), falling back to ru_maxrss."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                kb = float(line.split(":", 1)[1].strip().split()[0])
                return round(kb / 1024.0, 2)
    except Exception:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def discover_environment(out_dir: Path, worker_py: str) -> Dict[str, Any]:
    import platform
    import torch

    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    mem_gb = 0.0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_gb = round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
                break
    except Exception:
        pass

    probe_code = (
        "import importlib, json\n"
        "res = {}\n"
        "for mod, name in [('faiss','faiss'),('hnswlib','hnswlib'),('usearch.index','usearch'),"
        "('annoy','annoy'),('diskannpy','diskann')]:\n"
        "  try:\n"
        "    m = importlib.import_module(mod)\n"
        "    res[name] = getattr(m, '__version__', 'installed')\n"
        "  except Exception as e:\n"
        "    res[name] = f'unavailable: {e}'\n"
        "print(json.dumps(res))\n"
    )
    pr = subprocess.run([worker_py, "-c", probe_code], capture_output=True, text=True, check=True)
    engines = json.loads(pr.stdout.strip().splitlines()[-1])

    meta = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "driver_python": sys.version.split()[0],
        "worker_python": worker_py,
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "ram_gb": mem_gb,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "engines": engines,
        "gate4_note": "Cloud vCPU timings withheld from Gate-4 CPU tables; only ratios, Peak RSS, GPU recall, and hardware-invariant ops reported.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp7_env_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"Environment: {meta['cpu_count']} vCPUs ({cpu_model}), {mem_gb} GB RAM, GPU={meta['gpu']}")
    log(f"Worker engines ({worker_py}): {engines}")
    return meta


# ---------------------------------------------------------------------------
# Exact GPU Top-10 Ground Truth + NumPy float64 Cross-Verification
# ---------------------------------------------------------------------------
def compute_exact_gt_gpu(
    base_vecs: np.ndarray,
    init_ids: np.ndarray,
    windows_ops: List[Dict[str, Any]],
    queries: np.ndarray,
    window_q_indices: List[np.ndarray],
    k: int = 10,
    metric: str = "l2",
    verify_n: int = 8,
) -> np.ndarray:
    """Compute exact Top-k live ground truth for every window query on GPU."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_t = torch.from_numpy(base_vecs).to(device=device, dtype=torch.float32)
    base_norms = (base_t * base_t).sum(dim=1) if metric == "l2" else None

    live_mask = np.zeros(len(base_vecs), dtype=bool)
    live_mask[init_ids] = True

    n_win = len(windows_ops)
    q_per_win = len(window_q_indices[0])
    gt_out = np.empty((n_win, q_per_win, k), dtype=np.int32)

    for w_idx, ops in enumerate(windows_ops):
        if len(ops["erase"]) > 0:
            live_mask[ops["erase"]] = False
        if len(ops["insert"]) > 0:
            live_mask[ops["insert"]] = True

        live_idx = np.flatnonzero(live_mask)
        live_idx_t = torch.from_numpy(live_idx).to(device=device, dtype=torch.long)
        q_idx = window_q_indices[w_idx]
        q_t = torch.from_numpy(queries[q_idx]).to(device=device, dtype=torch.float32)

        chunk_sz = 200_000
        best_scores = None
        best_ids = None
        q_norms = (q_t * q_t).sum(dim=1, keepdim=True) if metric == "l2" else None

        for s in range(0, len(live_idx), chunk_sz):
            sub_ids = live_idx_t[s : s + chunk_sz]
            xb = base_t[sub_ids]
            if metric == "l2":
                d2 = q_norms + base_norms[sub_ids].unsqueeze(0) - 2.0 * (q_t @ xb.T)
                vals, pos = torch.topk(d2, k=k, dim=1, largest=False)
            else:
                sims = q_t @ xb.T
                vals, pos = torch.topk(sims, k=k, dim=1, largest=True)
            gids = sub_ids[pos]
            if best_scores is None:
                best_scores, best_ids = vals, gids
            else:
                cat_s = torch.cat([best_scores, vals], dim=1)
                cat_i = torch.cat([best_ids, gids], dim=1)
                top_v, top_p = torch.topk(cat_s, k=k, dim=1, largest=(metric != "l2"))
                best_scores = top_v
                best_ids = torch.gather(cat_i, 1, top_p)

        gt_out[w_idx] = best_ids.cpu().numpy().astype(np.int32)

    # Cross-verify sample queries on final window against NumPy float64
    live_idx = np.flatnonzero(live_mask)
    x_live = base_vecs[live_idx].astype(np.float64)
    for qi in range(min(verify_n, q_per_win)):
        q_vec = queries[window_q_indices[-1][qi]].astype(np.float64)
        if metric == "l2":
            dists = ((x_live - q_vec[None, :]) ** 2).sum(axis=1)
            true_top = set(live_idx[np.argsort(dists)[:k]].tolist())
        else:
            sims = x_live @ q_vec
            true_top = set(live_idx[np.argsort(-sims)[:k]].tolist())
        got_top = set(gt_out[-1, qi].tolist())
        overlap = len(true_top & got_top)
        assert overlap >= k - 1, f"GT verification mismatch on q={qi}: overlap={overlap}/{k}"
    log(f"Verified exact Top-{k} GPU GT across {n_win} windows x {q_per_win} queries ({metric}).")
    return gt_out


# ---------------------------------------------------------------------------
# B3 Builder: Real Chronological arXiv CS Abstracts (MiniLM-384)
# ---------------------------------------------------------------------------
def _parse_v1_timestamp(versions_val: Any) -> Optional[float]:
    if versions_val is None:
        return None
    if isinstance(versions_val, np.ndarray):
        parsed = versions_val.tolist()
    elif isinstance(versions_val, str):
        s = versions_val.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                return None
    else:
        parsed = versions_val
    if not isinstance(parsed, (list, tuple)) or len(parsed) == 0:
        return None
    created = None
    for item in parsed:
        if isinstance(item, dict) and item.get("version") == "v1":
            created = item.get("created")
            break
    if created is None and isinstance(parsed[0], dict):
        created = parsed[0].get("created")
    if not created:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(str(created))
        return dt.timestamp()
    except Exception:
        return None


def build_arxiv_cs_minilm384(work_dir: Path, out_dir: Path) -> Path:
    """Download real chronological arXiv CS abstracts (2023-2024), encode on GPU, compute GT."""
    wdir = work_dir / "workloads"
    wdir.mkdir(parents=True, exist_ok=True)
    out_npz = wdir / "arxiv_cs_minilm384.npz"
    prov_path = out_dir / "arxiv_provenance.json"
    if out_npz.exists() and prov_path.exists():
        log("Reusing pre-built arXiv CS MiniLM-384 workload.")
        return out_npz

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    repo_id = "bluuebunny/arxiv_metadata_by_year"
    shards = ["data/23.parquet", "data/24.parquet"]
    log(f"Downloading real arXiv metadata shards {shards} from {repo_id}...")
    records: List[Tuple[float, str, str, str]] = []
    cat_counts: Dict[str, int] = {}

    for shard in shards:
        local_pq = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset")
        df = pd.read_parquet(local_pq, columns=["id", "title", "abstract", "categories", "versions"])
        for aid, title, abst, cats, vers in zip(
            df["id"].tolist(),
            df["title"].tolist(),
            df["abstract"].tolist(),
            df["categories"].tolist(),
            df["versions"].tolist(),
        ):
            if not cats or not abst:
                continue
            primary_cat = str(cats).strip().split()[0]
            if not primary_cat.startswith("cs."):
                continue
            abst_clean = " ".join(str(abst).split())
            if len(abst_clean) < 50:
                continue
            ts = _parse_v1_timestamp(vers)
            if ts is None:
                continue
            title_clean = " ".join(str(title or "").split())
            text = f"{title_clean}. {abst_clean}" if title_clean else abst_clean
            records.append((ts, str(aid), primary_cat, text))

    # Strict chronological sort by true v1 submission timestamp
    records.sort(key=lambda x: (x[0], x[1]))
    n_needed = 140_000  # 60K init + 40*(1K base insert + 1K held-out chronological queries)
    if len(records) < n_needed:
        raise RuntimeError(f"Insufficient chronological CS records: {len(records)} < {n_needed}")
    records = records[:n_needed]

    for _, _, pcat, _ in records:
        cat_counts[pcat] = cat_counts.get(pcat, 0) + 1

    # Split chronologically: first 60K = init base, next 80K = 40 windows x 2K papers
    # In each window w, even indices (1K) = indexed base inserts, odd indices (1K) = held-out chronological queries
    init_recs = records[:60_000]
    stream_recs = records[60_000:]

    base_texts = [r[3] for r in init_recs]
    query_texts: List[str] = []
    win_q_idx: List[np.ndarray] = []
    n_win = 40
    q_per_win = 1000
    ins_per_win = 1000

    win_ins = np.empty((n_win, ins_per_win), dtype=np.int32)
    win_del = np.empty((n_win, 0), dtype=np.int32)
    ops: List[Dict[str, Any]] = []

    for w in range(n_win):
        chunk = stream_recs[w * 2000 : (w + 1) * 2000]
        ins_chunk = chunk[0::2]  # 1,000 papers inserted chronologically
        q_chunk = chunk[1::2]    # 1,000 held-out papers queried in same window
        b_start = len(base_texts)
        base_texts.extend([r[3] for r in ins_chunk])
        ins_ids = np.arange(b_start, b_start + ins_per_win, dtype=np.int32)
        win_ins[w] = ins_ids
        ops.append({"insert": ins_ids, "erase": np.empty(0, dtype=np.int32)})

        q_start = len(query_texts)
        query_texts.extend([r[3] for r in q_chunk])
        win_q_idx.append(np.arange(q_start, q_start + q_per_win, dtype=np.int32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Encoding {len(base_texts)} base + {len(query_texts)} query arXiv CS abstracts with all-MiniLM-L6-v2 on {device}...")
    t_enc = time.perf_counter()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    base_emb = model.encode(
        base_texts, batch_size=512, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)
    query_emb = model.encode(
        query_texts, batch_size=512, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)
    enc_s = time.perf_counter() - t_enc
    log(f"Encoded {base_emb.shape[0] + query_emb.shape[0]} arXiv abstracts in {enc_s:.1f}s.")

    init_ids = np.arange(60_000, dtype=np.int32)
    # Unit-normalized embeddings: squared L2 = 2 - 2*IP, so L2 and IP yield identical exact Top-10
    gt = compute_exact_gt_gpu(base_emb, init_ids, ops, query_emb, win_q_idx, k=10, metric="l2", verify_n=10)

    np.savez(
        out_npz,
        base=base_emb,
        queries=query_emb,
        init_ids=init_ids,
        win_q_idx=np.stack(win_q_idx, axis=0),
        win_inserts=win_ins,
        win_erases=win_del,
        gt=gt,
        metric=np.array(["l2"]),
    )

    def _iso(ts: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    top_cats = dict(sorted(cat_counts.items(), key=lambda kv: -kv[1])[:12])
    prov = {
        "source_dataset": repo_id,
        "shards": shards,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "normalized": True,
        "metric": "l2_on_unit_norm_equiv_cosine",
        "total_papers": len(records),
        "n_init": 60_000,
        "n_stream_inserts": 40_000,
        "n_final_live": 100_000,
        "n_queries": 40_000,
        "n_windows": n_win,
        "queries_per_window": q_per_win,
        "inserts_per_window": ins_per_win,
        "init_date_start_utc": _iso(init_recs[0][0]),
        "init_date_end_utc": _iso(init_recs[-1][0]),
        "stream_date_start_utc": _iso(stream_recs[0][0]),
        "stream_date_mid_utc": _iso(stream_recs[len(stream_recs) // 2][0]),
        "stream_date_end_utc": _iso(stream_recs[-1][0]),
        "top_primary_cs_categories": top_cats,
        "gpu_encode_s": round(enc_s, 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    log(f"Built arXiv CS MiniLM-384 workload ({prov['init_date_start_utc']} .. {prov['stream_date_end_utc']}).")
    return out_npz


# ---------------------------------------------------------------------------
# B1 Builder: SIFT-1M Drift (600K->1M) & CRUD-30%
# ---------------------------------------------------------------------------
def download_sift1m(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    import h5py

    data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = data_dir / "sift-128-euclidean.hdf5"
    urls = [
        "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    ]
    if not (h5_path.exists() and h5_path.stat().st_size > 400_000_000):
        for url in urls:
            try:
                log(f"Downloading SIFT-1M from {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
                with urllib.request.urlopen(req, timeout=300) as resp, open(h5_path, "wb") as out_f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out_f.write(chunk)
                if h5_path.stat().st_size > 400_000_000:
                    break
            except Exception as exc:
                log(f"Download failed from {url}: {exc}")
                if h5_path.exists():
                    h5_path.unlink()
    if not (h5_path.exists() and h5_path.stat().st_size > 400_000_000):
        raise RuntimeError("Failed to download sift-128-euclidean.hdf5 from all sources")

    with h5py.File(h5_path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
    log(f"Loaded SIFT-1M: base={train.shape}, queries={test.shape}")
    return train, test


def build_sift1m_workloads(work_dir: Path) -> Dict[str, Path]:
    """Build 1M Drift (600K->1.0M) and 1M CRUD-30% workloads on disk."""
    wdir = work_dir / "workloads"
    wdir.mkdir(parents=True, exist_ok=True)
    drift_npz = wdir / "sift1m_drift.npz"
    crud_npz = wdir / "sift1m_crud30.npz"
    if drift_npz.exists() and crud_npz.exists():
        log("Reusing pre-built SIFT-1M workloads.")
        return {"sift1m_drift": drift_npz, "sift1m_crud30": crud_npz}

    train, test = download_sift1m(work_dir / "raw")
    rng = np.random.default_rng(42)

    c0 = train[0]
    c1 = train[500_000]
    d0 = ((test - c0[None, :]) ** 2).sum(axis=1)
    d1 = ((test - c1[None, :]) ** 2).sum(axis=1)
    group_a = np.flatnonzero(d0 <= d1)
    group_b = np.flatnonzero(d0 > d1)
    if len(group_a) < 1000 or len(group_b) < 1000:
        group_a = np.arange(0, len(test) // 2)
        group_b = np.arange(len(test) // 2, len(test))

    n_win = 40
    q_per_win = 1000
    win_q_idx = []
    for w in range(n_win):
        pool_q = group_a if w < 20 else group_b
        win_q_idx.append(rng.choice(pool_q, size=q_per_win, replace=True).astype(np.int32))
    win_q_arr = np.stack(win_q_idx, axis=0)

    # 1. SIFT-1M Drift: 600K init + 10K inserts/window -> 1.0M live
    init_ids = np.arange(600_000, dtype=np.int32)
    perm_cand = rng.permutation(np.arange(600_000, 1_000_000, dtype=np.int32))
    drift_ops = []
    drift_ins_mat = np.empty((n_win, 10_000), dtype=np.int32)
    drift_del_mat = np.empty((n_win, 0), dtype=np.int32)
    for w in range(n_win):
        ins = perm_cand[w * 10_000 : (w + 1) * 10_000]
        drift_ins_mat[w] = ins
        drift_ops.append({"insert": ins, "erase": np.empty(0, dtype=np.int32)})

    gt_drift = compute_exact_gt_gpu(train, init_ids, drift_ops, test, win_q_idx, k=10, metric="l2")
    np.savez(
        drift_npz,
        base=train,
        queries=test,
        init_ids=init_ids,
        win_q_idx=win_q_arr,
        win_inserts=drift_ins_mat,
        win_erases=drift_del_mat,
        gt=gt_drift,
        metric=np.array(["l2"]),
    )

    # 2. SIFT-1M CRUD-30%: 600K init, 10K ops/window (7,000 inserts + 3,000 erases)
    live_list = list(range(600_000))
    cand_pool = perm_cand
    c_ptr = 0
    crud_ops = []
    crud_ins_mat = np.empty((n_win, 7_000), dtype=np.int32)
    crud_del_mat = np.empty((n_win, 3_000), dtype=np.int32)
    for w in range(n_win):
        del_pos = rng.choice(len(live_list), size=3_000, replace=False)
        del_ids = np.array([live_list[p] for p in del_pos], dtype=np.int32)
        del_set = set(del_ids.tolist())
        live_list = [x for x in live_list if x not in del_set]
        ins_ids = cand_pool[c_ptr : c_ptr + 7_000]
        c_ptr += 7_000
        live_list.extend(ins_ids.tolist())
        crud_ins_mat[w] = ins_ids
        crud_del_mat[w] = del_ids
        crud_ops.append({"insert": ins_ids, "erase": del_ids})

    gt_crud = compute_exact_gt_gpu(train, init_ids, crud_ops, test, win_q_idx, k=10, metric="l2")
    np.savez(
        crud_npz,
        base=train,
        queries=test,
        init_ids=init_ids,
        win_q_idx=win_q_arr,
        win_inserts=crud_ins_mat,
        win_erases=crud_del_mat,
        gt=gt_crud,
        metric=np.array(["l2"]),
    )
    log("Built SIFT-1M Drift and CRUD-30% workloads.")
    return {"sift1m_drift": drift_npz, "sift1m_crud30": crud_npz}


# ---------------------------------------------------------------------------
# Subprocess 5-Engine Worker (FAISS-IVF, hnswlib, USearch, FreshDiskANN, Annoy)
# ---------------------------------------------------------------------------
def run_single_engine_worker(
    npz_path: Path,
    system: str,
    param_tag: str,
    params: Dict[str, Any],
    max_windows: Optional[int] = None,
) -> Dict[str, Any]:
    set_single_thread_env()
    d = np.load(npz_path, allow_pickle=False)
    base = np.ascontiguousarray(d["base"], dtype=np.float32)
    queries = np.ascontiguousarray(d["queries"], dtype=np.float32)
    init_ids = np.asarray(d["init_ids"], dtype=np.int32)
    win_q_idx = np.asarray(d["win_q_idx"], dtype=np.int32)
    win_ins = np.asarray(d["win_inserts"], dtype=np.int32)
    win_del = np.asarray(d["win_erases"], dtype=np.int32)
    gt = np.asarray(d["gt"], dtype=np.int32)
    metric = str(d["metric"][0])
    dim = base.shape[1]
    k = gt.shape[2]
    n_win = min(len(win_q_idx), max_windows) if max_windows else len(win_q_idx)

    has_erases = win_del.shape[1] > 0 and np.any(win_del[:n_win].size > 0)
    if system == "annoy" and has_erases:
        return {
            "system": system,
            "param_tag": param_tag,
            "n_windows": n_win,
            "mean_recall": 0.0,
            "min_recall": 0.0,
            "attainment_0965_pct": 0.0,
            "attainment_095_pct": 0.0,
            "query_cpu_pct": 0.0,
            "maint_cpu_pct": 0.0,
            "maint_to_query_ratio": None,
            "peak_rss_mb": get_peak_rss_mb(),
            "_internal_total_cpu_ms": 0.0,
            "status": "unsupported_erase",
        }

    t_b0 = cpu_ms_now()
    idx_obj: Any = None
    diskann_tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
    live_set = set(init_ids.tolist())
    has_deleted_hnsw = False

    if system == "faiss_ivf":
        import faiss
        faiss.omp_set_num_threads(1)
        m_type = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        nlist = int(params.get("nlist", 512))
        quantizer = faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
        idx_obj = faiss.IndexIVFFlat(quantizer, dim, nlist, m_type)
        idx_obj.train(base[init_ids[: min(100_000, len(init_ids))]])
        idx_obj.add_with_ids(base[init_ids], init_ids.astype(np.int64))
        idx_obj.nprobe = int(params.get("nprobe", 16))
    elif system == "hnswlib":
        import hnswlib
        sp = "l2" if metric == "l2" else "ip"
        idx_obj = hnswlib.Index(space=sp, dim=dim)
        idx_obj.init_index(
            max_elements=len(base) + 50_000,
            ef_construction=int(params.get("ef_construction", 100)),
            M=int(params.get("M", 16)),
            allow_replace_deleted=True,
        )
        idx_obj.set_num_threads(1)
        idx_obj.add_items(base[init_ids], init_ids, num_threads=1)
        idx_obj.set_ef(int(params.get("ef", 64)))
    elif system == "usearch":
        from usearch.index import Index
        um = "l2sq" if metric == "l2" else "ip"
        idx_obj = Index(
            ndim=dim,
            metric=um,
            dtype="f32",
            connectivity=int(params.get("M", 16)),
            expansion_add=int(params.get("ef_construction", 100)),
            expansion_search=int(params.get("ef", 64)),
        )
        idx_obj.add(init_ids.astype(np.uint64), base[init_ids], threads=1)
    elif system == "diskann":
        import diskannpy
        R = int(params.get("R", 32))
        Lbuild = int(params.get("Lbuild", 100))
        search_L = int(params.get("search_L", 64))
        max_vecs = len(base) + 50_000
        diskann_tmpdir = tempfile.TemporaryDirectory(prefix="pvldb_diskann_")
        tags0 = (init_ids + 1).astype(np.uint32)
        d_metric = "l2" if metric == "l2" else "mips"
        diskannpy.build_memory_index(
            base[init_ids],
            d_metric,
            diskann_tmpdir.name,
            complexity=Lbuild,
            graph_degree=R,
            num_threads=1,
            tags=tags0,
        )
        idx_obj = diskannpy.DynamicMemoryIndex.from_file(
            diskann_tmpdir.name,
            max_vectors=max_vecs,
            complexity=Lbuild,
            graph_degree=R,
            num_threads=1,
            initial_search_complexity=search_L,
            search_threads=1,
            concurrent_consolidation=False,
        )
    elif system == "annoy":
        from annoy import AnnoyIndex
        am = "euclidean" if metric == "l2" else "dot"
        n_trees = int(params.get("n_trees", 10))
        idx_obj = AnnoyIndex(dim, am)
        for i in init_ids.tolist():
            idx_obj.add_item(int(i), base[int(i)])
        idx_obj.build(n_trees, n_jobs=1)
    else:
        raise ValueError(f"Unsupported engine: {system}")

    _ = cpu_ms_now() - t_b0
    q_cpu_sum = 0.0
    m_cpu_sum = 0.0
    recalls: List[float] = []

    try:
        for w in range(n_win):
            ins = win_ins[w]
            dels = win_del[w]
            t_m0 = cpu_ms_now()

            if system == "faiss_ivf":
                import faiss
                if len(dels) > 0:
                    idx_obj.remove_ids(dels.astype(np.int64))
                    live_set.difference_update(dels.tolist())
                if len(ins) > 0:
                    idx_obj.add_with_ids(base[ins], ins.astype(np.int64))
                    live_set.update(ins.tolist())
                ret_ev = int(params.get("retrain_every", 0))
                if ret_ev > 0 and (w + 1) % ret_ev == 0:
                    cur_live = np.array(sorted(live_set), dtype=np.int32)
                    m_type = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
                    nlist = int(params.get("nlist", 512))
                    quant = faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
                    idx_obj = faiss.IndexIVFFlat(quant, dim, nlist, m_type)
                    idx_obj.train(base[cur_live[: min(100_000, len(cur_live))]])
                    idx_obj.add_with_ids(base[cur_live], cur_live.astype(np.int64))
                    idx_obj.nprobe = int(params.get("nprobe", 16))
            elif system == "hnswlib":
                for did in dels.tolist():
                    idx_obj.mark_deleted(int(did))
                    has_deleted_hnsw = True
                if len(ins) > 0:
                    idx_obj.add_items(base[ins], ins, num_threads=1, replace_deleted=has_deleted_hnsw)
            elif system == "usearch":
                if len(dels) > 0:
                    idx_obj.remove(dels.astype(np.uint64))
                if len(ins) > 0:
                    idx_obj.add(ins.astype(np.uint64), base[ins], threads=1)
            elif system == "diskann":
                for did in dels.tolist():
                    idx_obj.mark_deleted(np.uint32(int(did) + 1))
                if len(ins) > 0:
                    ins_vecs = base[ins]
                    for iid, vec in zip(ins.tolist(), ins_vecs):
                        idx_obj.insert(vec, np.uint32(int(iid) + 1))
                cons_ev = int(params.get("consolidate_every", 0))
                if cons_ev > 0 and (w + 1) % cons_ev == 0:
                    idx_obj.consolidate_delete()
            elif system == "annoy":
                live_set.update(ins.tolist())
                reb_ev = int(params.get("rebuild_every", 4))
                if (w + 1) % reb_ev == 0 or w == 0:
                    from annoy import AnnoyIndex
                    am = "euclidean" if metric == "l2" else "dot"
                    idx_obj = AnnoyIndex(dim, am)
                    for vid in live_set:
                        idx_obj.add_item(int(vid), base[int(vid)])
                    idx_obj.build(int(params.get("n_trees", 10)), n_jobs=1)

            m_cpu_sum += cpu_ms_now() - t_m0

            q_ids = win_q_idx[w]
            q_vecs = queries[q_ids]
            pred_ids = np.full((len(q_vecs), k), -1, dtype=np.int64)
            t_q0 = cpu_ms_now()

            if system == "faiss_ivf":
                _, I = idx_obj.search(q_vecs, k)
                pred_ids[:] = I
            elif system == "hnswlib":
                labels, _ = idx_obj.knn_query(q_vecs, k=k, num_threads=1)
                pred_ids[:] = labels
            elif system == "usearch":
                for i in range(len(q_vecs)):
                    matches = idx_obj.search(q_vecs[i], k, threads=1)
                    keys = matches.keys
                    pred_ids[i, : min(k, len(keys))] = keys[:k]
            elif system == "diskann":
                sL = int(params.get("search_L", 64))
                res = idx_obj.batch_search(q_vecs, k, sL, 1)
                pred_ids[:] = np.asarray(res.identifiers, dtype=np.int64) - 1
            elif system == "annoy":
                sk = int(params.get("search_k", 16000))
                for i in range(len(q_vecs)):
                    r = idx_obj.get_nns_by_vector(q_vecs[i], k, search_k=sk)
                    pred_ids[i, : min(k, len(r))] = r[:k]

            q_cpu_sum += cpu_ms_now() - t_q0

            gt_w = gt[w]
            hits = 0
            for i in range(len(q_vecs)):
                hits += len(set(pred_ids[i].tolist()) & set(gt_w[i].tolist()))
            recalls.append(hits / (len(q_vecs) * k))
    finally:
        if diskann_tmpdir is not None:
            diskann_tmpdir.cleanup()

    tot_cpu = q_cpu_sum + m_cpu_sum
    peak_rss_mb = get_peak_rss_mb()
    att0965 = sum(1 for r in recalls if r >= 0.965) / len(recalls) * 100.0
    att095 = sum(1 for r in recalls if r >= 0.950) / len(recalls) * 100.0
    q_pct = (q_cpu_sum / tot_cpu * 100.0) if tot_cpu > 0 else 0.0
    m_pct = (m_cpu_sum / tot_cpu * 100.0) if tot_cpu > 0 else 0.0
    m_to_q = (m_cpu_sum / q_cpu_sum) if q_cpu_sum > 0 else None

    return {
        "system": system,
        "param_tag": param_tag,
        "n_windows": n_win,
        "mean_recall": round(statistics.fmean(recalls), 4),
        "min_recall": round(min(recalls), 4),
        "attainment_0965_pct": round(att0965, 1),
        "attainment_095_pct": round(att095, 1),
        "query_cpu_pct": round(q_pct, 2),
        "maint_cpu_pct": round(m_pct, 2),
        "maint_to_query_ratio": round(m_to_q, 4) if m_to_q is not None else None,
        "peak_rss_mb": peak_rss_mb,
        "_internal_total_cpu_ms": tot_cpu,
        "status": "ok",
    }


def _invoke_worker_subproc(
    worker_py: str,
    npz_path: Path,
    system: str,
    param_tag: str,
    params: Dict[str, Any],
    core: Optional[int] = 0,
    max_windows: Optional[int] = None,
    timeout_s: int = 2400,
) -> Dict[str, Any]:
    cmd = []
    if core is not None and shutil.which("taskset"):
        cmd.extend(["taskset", "-c", str(core)])
    cmd.extend([
        worker_py,
        str(Path(__file__).resolve()),
        "--worker",
        "--npz", str(npz_path),
        "--system", system,
        "--param-tag", param_tag,
        "--params-json", json.dumps(params),
    ])
    if max_windows is not None:
        cmd.extend(["--max-windows", str(max_windows)])
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    env.pop("LD_PRELOAD", None)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed ({system}/{param_tag}): {proc.stderr[-400:]}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("WORKER_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"Missing WORKER_JSON from {system}/{param_tag}: {proc.stdout[-300:]}")


def _finalize_engine_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace internal CPU ms with baseline-relative ratio; never export cloud raw CPU ms."""
    base_cpu = None
    for r in raw_rows:
        if r.get("status") == "ok" and r["system"] == "faiss_ivf" and r["param_tag"] == "np16":
            base_cpu = r["_internal_total_cpu_ms"]
            break
    if base_cpu is None:
        for r in raw_rows:
            if r.get("status") == "ok" and r["_internal_total_cpu_ms"] > 0:
                base_cpu = r["_internal_total_cpu_ms"]
                break

    out: List[Dict[str, Any]] = []
    for r in raw_rows:
        rec = {k: v for k, v in r.items() if k != "_internal_total_cpu_ms"}
        if r.get("status") == "ok" and base_cpu and base_cpu > 0:
            rec["rel_cpu_vs_faiss_np16"] = round(r["_internal_total_cpu_ms"] / base_cpu, 4)
        else:
            rec["rel_cpu_vs_faiss_np16"] = None
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# B3 (b): Partitioned Counterfactual Replay, Theorem-1 MILP, & Gate 2 Audit
# ---------------------------------------------------------------------------
def _solve_segment_flow_milp(
    q_mat: np.ndarray,
    d_mat: np.ndarray,
    B: int,
    K: int = K_BLOCK,
) -> Tuple[float, List[Tuple[int, int]]]:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    T, R = q_mat.shape
    J = T // K
    if B <= 0 or J <= 1:
        return float((q_mat * d_mat).sum()), []

    B_eff = min(int(B), J - 1)
    arcs: List[Tuple[int, int, int]] = []
    costs: List[float] = []
    for r in range(R):
        for j1 in range(J):
            t_start = K * j1
            d_reset = float(d_mat[t_start - 1, r]) if j1 > 0 else 0.0
            for j2 in range(j1 + 1, J + 1):
                t_end = min(K * j2, T)
                eff_d = d_mat[t_start:t_end, r] if j1 == 0 else np.maximum(0.0, d_mat[t_start:t_end, r] - d_reset)
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

    sched: List[Tuple[int, int]] = []
    for idx, val in enumerate(res.x):
        if val > 0.5:
            r, j1, _ = arcs[idx]
            if 1 <= j1 <= J - 1:
                sched.append((j1, r))
    sched.sort()
    return float(res.fun), sched


def _solve_myopic_1blk(q_mat: np.ndarray, d_mat: np.ndarray, B: int, K: int = K_BLOCK) -> float:
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


def _ewma_next(q_hist: np.ndarray, alpha: float) -> np.ndarray:
    if q_hist.shape[0] == 0:
        return np.zeros(q_hist.shape[1], dtype=np.float64)
    m = q_hist[0].astype(np.float64).copy()
    for w in range(1, q_hist.shape[0]):
        m = alpha * q_hist[w] + (1.0 - alpha) * m
    return m


def _cusum_next(q_hist: np.ndarray, k_sigma: float = 3.0, alpha: float = 0.5) -> np.ndarray:
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


def _simulate_policy_ops(
    q_mat: np.ndarray,
    d_noop: np.ndarray,
    policy: str,
    *,
    cadence: int = 1,
    q_hat: Optional[np.ndarray] = None,
    budget_cap: int = 999,
    K: int = K_BLOCK,
) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Exact hardware-invariant scan work simulation matching qbam_replay merge semantics."""
    T, R = q_mat.shape
    J = T // K
    last_reset = np.full(R, -1, dtype=np.int64)
    actions: List[Tuple[int, int]] = []
    fixed_counter = 0
    total_ops = 0.0

    for j in range(J):
        t0, t1 = j * K, min((j + 1) * K, T)
        if j >= 1:
            cur_delta = np.array(
                [float(d_noop[t0 - 1, r]) - (float(d_noop[last_reset[r], r]) if last_reset[r] >= 0 else 0.0) for r in range(R)],
                dtype=np.float64,
            )
            chosen_r = -1
            if policy == "noop":
                chosen_r = -1
            elif policy == "fixed":
                if (j - 1) % cadence == 0:
                    cand_r = fixed_counter % R
                    fixed_counter += 1
                    if cur_delta[cand_r] > 0:
                        chosen_r = cand_r
            elif policy == "qhat":
                assert q_hat is not None
                if len(actions) < budget_cap:
                    scores = np.where(cur_delta > 0, q_hat[j] * cur_delta, -1.0)
                    best_r = int(np.argmax(scores))
                    if scores[best_r] > 0.0:
                        chosen_r = best_r
            else:
                raise ValueError(f"Unknown policy: {policy}")

            if chosen_r >= 0:
                last_reset[chosen_r] = t0 - 1
                actions.append((j, chosen_r))

        for r in range(R):
            d_prev = float(d_noop[last_reset[r], r]) if last_reset[r] >= 0 else 0.0
            eff_d = np.maximum(0.0, d_noop[t0:t1, r] - d_prev)
            total_ops += float((q_mat[t0:t1, r] * eff_d).sum())

    return total_ops, len(actions), actions


def run_arxiv_counterfactual_replay(npz_path: Path, out_dir: Path) -> None:
    """Run B3 (b): R=16 Voronoi routing, 10 predictors/cadences, MILP GCR, and Gate 2 audit."""
    import audit_hygiene as ah
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = np.load(npz_path, allow_pickle=False)
    base = np.asarray(d["base"], dtype=np.float32)
    queries = np.asarray(d["queries"], dtype=np.float32)
    init_ids = np.asarray(d["init_ids"], dtype=np.int32)
    win_q_idx = np.asarray(d["win_q_idx"], dtype=np.int32)
    win_ins = np.asarray(d["win_inserts"], dtype=np.int32)

    R = 16
    nprobe = 4
    T = len(win_q_idx)
    J = T // K_BLOCK

    # K-Means centroids on initial 60K arXiv CS papers (matching ensure_variant_dir)
    rng = np.random.default_rng(42 + R)
    x_init = base[init_ids]
    cent = x_init[rng.choice(len(x_init), R, replace=False)].copy()
    for _ in range(25):
        cn2 = (cent ** 2).sum(1)
        assign = np.empty(len(x_init), dtype=np.int32)
        for i in range(0, len(x_init), 20000):
            xb = x_init[i : i + 20000]
            d2 = (xb ** 2).sum(1, keepdims=True) + cn2[None, :] - 2.0 * (xb @ cent.T)
            assign[i : i + 20000] = d2.argmin(1)
        new_c = np.zeros_like(cent)
        for r_i in range(R):
            m = assign == r_i
            new_c[r_i] = x_init[m].mean(0) if m.any() else x_init[rng.integers(0, len(x_init))]
        cent = new_c.astype(np.float32)

    cn2 = (cent ** 2).sum(1)
    q_mat = np.zeros((T, R), dtype=np.float64)
    ins_mat = np.zeros((T, R), dtype=np.float64)

    for w in range(T):
        qv = queries[win_q_idx[w]]
        d2q = (qv ** 2).sum(1, keepdims=True) + cn2[None, :] - 2.0 * (qv @ cent.T)
        top_s = np.argpartition(d2q, kth=nprobe - 1, axis=1)[:, :nprobe]
        for s_row in top_s.tolist():
            for s in s_row:
                q_mat[w, s] += 1.0

        iv = base[win_ins[w]]
        d2i = (iv ** 2).sum(1, keepdims=True) + cn2[None, :] - 2.0 * (iv @ cent.T)
        best_s = np.argmin(d2i, axis=1)
        for s in best_s.tolist():
            ins_mat[w, s] += 1.0

    d_noop = np.cumsum(ins_mat, axis=0)
    base_n = np.bincount(
        np.argmin((x_init ** 2).sum(1, keepdims=True) + cn2[None, :] - 2.0 * (x_init @ cent.T), axis=1),
        minlength=R,
    ).astype(np.float64)

    # Build strictly causal features vs leaky features per block j=1..J-1
    causal_names = [
        "qmean8", "qstd8", "slope", "q_lag4", "q_lag3", "q_lag2", "q_lag1",
        "dinc4", "base_n", "glob_qmean8", "glob_dinc4",
    ] + [f"shard_{r}" for r in range(R)]
    leaky_names = causal_names + ["leak_block_j", "leak_dcum_shard", "leak_dcum_total"]

    X_causal_by_j: Dict[int, np.ndarray] = {}
    X_leaky_by_j: Dict[int, np.ndarray] = {}
    y_by_j: Dict[int, np.ndarray] = {}
    eye = np.eye(R, dtype=np.float64)

    preds: Dict[str, np.ndarray] = {
        m: np.zeros((J, R), dtype=np.float64)
        for m in ("past_mean", "ewma_a03", "ewma_a07", "cusum_reset", "ridge_ar", "gbrt_causal", "gbrt_leaky", "oracle_clairvoyant")
    }

    for j in range(1, J):
        h0 = j * K_BLOCK
        past = q_mat[max(0, h0 - 8) : h0]
        L = past.shape[0]
        x_idx = np.arange(L, dtype=np.float64)
        xc = x_idx - x_idx.mean()
        denom = float((xc ** 2).sum())
        slope = (past * xc[:, None]).sum(axis=0) / denom if denom > 0 else np.zeros(R)
        last4 = q_mat[max(0, h0 - 4) : h0]
        if last4.shape[0] < 4:
            last4 = np.concatenate([np.repeat(last4[:1], 4 - last4.shape[0], axis=0), last4], axis=0)
        d_prev = d_noop[h0 - 1]
        d_prev4 = d_noop[h0 - 5] if h0 >= 5 else np.zeros(R)
        dinc4 = d_prev - d_prev4
        y_true = q_mat[h0 : h0 + K_BLOCK].mean(axis=0)
        y_by_j[j] = y_true

        Xc = np.column_stack([
            past.mean(axis=0),
            past.std(axis=0),
            slope,
            last4.T,
            dinc4,
            base_n,
            np.full(R, float(past.mean(axis=0).sum())),
            np.full(R, float(dinc4.sum())),
            eye,
        ])
        Xl = np.hstack([
            Xc,
            np.column_stack([
                np.full(R, float(j)),
                d_prev,
                np.full(R, float(d_prev.sum())),
            ]),
        ])
        X_causal_by_j[j] = Xc
        X_leaky_by_j[j] = Xl

        preds["past_mean"][j] = past.mean(axis=0)
        preds["ewma_a03"][j] = _ewma_next(q_mat[:h0], 0.3)
        preds["ewma_a07"][j] = _ewma_next(q_mat[:h0], 0.7)
        preds["cusum_reset"][j] = _cusum_next(q_mat[:h0], k_sigma=3.0)
        preds["oracle_clairvoyant"][j] = y_true

    # Gate 2 Spearman Audit (check_clock_leak)
    Xc_all = np.vstack([X_causal_by_j[j] for j in range(1, J)])
    Xl_all = np.vstack([X_leaky_by_j[j] for j in range(1, J)])
    y_all = np.concatenate([y_by_j[j] for j in range(1, J)])
    step_vec = [float(j) for j in range(1, J) for _ in range(R)]

    causal_dict = {"_step_j": step_vec}
    for ci, cn in enumerate(causal_names):
        if not cn.startswith("shard_"):
            causal_dict[cn] = Xc_all[:, ci].tolist()
    causal_audit = ah.check_clock_leak(causal_dict, step_col="_step_j", threshold=0.85)

    leaky_dict = {"_step_j": step_vec}
    for li, ln in enumerate(leaky_names):
        if not ln.startswith("shard_"):
            leaky_dict[ln] = Xl_all[:, li].tolist()
    leaky_audit = ah.check_clock_leak(leaky_dict, step_col="_step_j", threshold=0.85)

    # Walk-forward training for Ridge-AR & GBRT-Causal; leaky in-sample + clock features for GBRT-Leaky
    for j in range(1, J):
        if j < 3:
            preds["ridge_ar"][j] = preds["ewma_a03"][j]
            preds["gbrt_causal"][j] = preds["ewma_a03"][j]
        else:
            X_tr = np.vstack([X_causal_by_j[jj] for jj in range(1, j)])
            y_tr = np.concatenate([y_by_j[jj] for jj in range(1, j)])
            mu = X_tr.mean(axis=0)
            sd = np.maximum(X_tr.std(axis=0), 1e-6)
            Z_tr = (X_tr - mu) / sd
            Z_te = (X_causal_by_j[j] - mu) / sd
            y_mu = float(y_tr.mean())
            w_r = np.linalg.solve(Z_tr.T @ Z_tr + 10.0 * np.eye(Z_tr.shape[1]), Z_tr.T @ (y_tr - y_mu))
            preds["ridge_ar"][j] = np.maximum(0.0, y_mu + Z_te @ w_r)

            gb_c = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=5, random_state=42)
            gb_c.fit(X_tr, y_tr)
            preds["gbrt_causal"][j] = np.maximum(0.0, gb_c.predict(X_causal_by_j[j]))

    # GBRT-Leaky: full-stream clock-leaky model matching Trap 2 diagnostic
    gb_l = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=5, random_state=42)
    gb_l.fit(Xl_all, y_all)
    for j in range(1, J):
        preds["gbrt_leaky"][j] = np.maximum(0.0, gb_l.predict(X_leaky_by_j[j]))

    # Predictor diagnostics (Top-1 agreement + JSD)
    diag_rows: List[Dict[str, Any]] = []
    for m_name, p_mat in preds.items():
        agrees, jsds = [], []
        for j in range(1, J):
            dn = d_noop[j * K_BLOCK - 1]
            p_top = int(np.argmax(p_mat[j] * dn))
            t_top = int(np.argmax(y_by_j[j] * dn))
            agrees.append(1.0 if p_top == t_top else 0.0)
            jsds.append(_jsd(p_mat[j].tolist(), y_by_j[j].tolist()))
        diag_rows.append({
            "trace": "arxiv_cs_minilm384",
            "model": m_name,
            "n_blocks": J - 1,
            "top1_agree_all": round(float(np.mean(agrees)), 4),
            "mean_jsd": round(float(np.mean(jsds)), 4),
        })
    _write_csv(diag_rows, out_dir / "arxiv_minilm384_predictor_diagnostics.csv")

    # Counterfactual replay across No-Op, Fixed c=1..8, and Predictors @ B=5 (k5) & B=9 (uncap)
    noop_ops, _, _ = _simulate_policy_ops(q_mat, d_noop, "noop")
    milp_cache: Dict[int, Tuple[float, float]] = {0: (noop_ops, noop_ops)}

    def _get_bounds(B: int) -> Tuple[float, float]:
        if B not in milp_cache:
            opt_ops, _ = _solve_segment_flow_milp(q_mat, d_noop, B=B, K=K_BLOCK)
            myo_ops = _solve_myopic_1blk(q_mat, d_noop, B=B, K=K_BLOCK)
            milp_cache[B] = (opt_ops, myo_ops)
        return milp_cache[B]

    arm_specs: List[Tuple[str, str, Dict[str, Any]]] = [("noop", "noop", {})]
    for c in range(1, 9):
        arm_specs.append((f"fixed_c{c}_merge", "fixed", {"cadence": c}))

    pred_arm_map = [
        ("past_qmass", "past_mean"),
        ("forecast_ewma_a03", "ewma_a03"),
        ("forecast_ewma_a07", "ewma_a07"),
        ("forecast_cusum_reset", "cusum_reset"),
        ("forecast_ridge_ar", "ridge_ar"),
        ("forecast_gbrt_causal", "gbrt_causal"),
        ("forecast_gbrt_leaky", "gbrt_leaky"),
        ("forecast_oracle_clairvoyant", "oracle_clairvoyant"),
    ]
    for b_tag, b_cap in [("k5", 5), ("uncap", 9)]:
        for arm_prefix, m_key in pred_arm_map:
            arm_specs.append((f"{arm_prefix}_{b_tag}", "qhat", {"model": m_key, "budget": b_cap}))

    replay_rows: List[Dict[str, Any]] = []
    fixed_ops_by_c: Dict[str, float] = {}
    for arm_name, pol_type, pol_kw in arm_specs:
        if pol_type == "noop":
            ops, n_act, _ = _simulate_policy_ops(q_mat, d_noop, "noop")
        elif pol_type == "fixed":
            ops, n_act, _ = _simulate_policy_ops(q_mat, d_noop, "fixed", cadence=pol_kw["cadence"])
            fixed_ops_by_c[f"c{pol_kw['cadence']}"] = ops
        else:
            ops, n_act, _ = _simulate_policy_ops(
                q_mat, d_noop, "qhat", q_hat=preds[pol_kw["model"]], budget_cap=pol_kw["budget"]
            )
        opt_ops, myo_ops = _get_bounds(n_act)
        denom = noop_ops - opt_ops
        gcr = 1.0 if n_act == 0 else ((noop_ops - ops) / denom if denom > 1e-9 else 1.0)
        cr = ops / max(opt_ops, 1e-9)
        myo_gap = (myo_ops - opt_ops) / max(opt_ops, 1e-9) * 100.0
        replay_rows.append({
            "trace": "arxiv_cs_minilm384",
            "arm": arm_name,
            "n_actions": n_act,
            "exact_scan_ops": round(ops, 1),
            "noop_scan_ops": round(noop_ops, 1),
            "global_opt_ops": round(opt_ops, 1),
            "myopic_opt_ops": round(myo_ops, 1),
            "global_capture_ratio": round(gcr, 6),
            "competitive_ratio": round(cr, 6),
            "myopic_vs_global_gap_pct": round(myo_gap, 3),
        })

    _write_csv(
        [r for r in replay_rows if not r["arm"].endswith("_uncap")],
        out_dir / "arxiv_minilm384_exact_ops_and_dp.csv",
    )
    _write_csv(
        [r for r in replay_rows if r["arm"].endswith("_uncap") or r["arm"] == "noop" or r["arm"].startswith("fixed_")],
        out_dir / "arxiv_minilm384_uncap_ops_and_dp.csv",
    )

    # Gate 1 Cadence Bracket check on exact_scan_ops + Gate 2 Clock-Leak report
    gate1_brackets = {
        r["arm"]: ah.check_cadence_bracket(float(r["exact_scan_ops"]), fixed_ops_by_c)
        for r in replay_rows
        if r["arm"].endswith("_k5") or r["arm"].endswith("_uncap")
    }
    gate2_report = {
        "trace": "arxiv_cs_minilm384",
        "R": R,
        "nprobe": nprobe,
        "T": T,
        "K": K_BLOCK,
        "clock_leak_audit": {
            "gbrt_causal": causal_audit,
            "gbrt_leaky": leaky_audit,
        },
        "predictor_diagnostics": diag_rows,
        "gate1_cadence_brackets_ops": gate1_brackets,
    }
    (out_dir / "arxiv_minilm384_gate2_spearman.json").write_text(
        json.dumps(gate2_report, indent=2) + "\n", encoding="utf-8"
    )
    log(
        f"arXiv counterfactual replay complete ({len(replay_rows)} rows): "
        f"causal_passed={causal_audit['passed']}, leaky_flagged={list(leaky_audit['flagged_features'])}"
    )


# ---------------------------------------------------------------------------
# Suite Runners: B3 (arXiv 5-Engine + Replay) & B1 (1M 5-Engine Audit)
# ---------------------------------------------------------------------------
def run_arxiv_suite(work_dir: Path, out_dir: Path, worker_py: str) -> None:
    sem_npz = build_arxiv_cs_minilm384(work_dir, out_dir)
    run_arxiv_counterfactual_replay(sem_npz, out_dir)

    configs_sem = [
        ("faiss_ivf", "np16", {"nlist": 256, "nprobe": 16, "retrain_every": 0}),
        ("faiss_ivf", "np32", {"nlist": 256, "nprobe": 32, "retrain_every": 0}),
        ("faiss_ivf", "np32_ret4", {"nlist": 256, "nprobe": 32, "retrain_every": 4}),
        ("hnswlib", "ef64", {"ef": 64, "M": 16, "ef_construction": 100}),
        ("hnswlib", "ef128", {"ef": 128, "M": 16, "ef_construction": 100}),
        ("usearch", "ef64", {"ef": 64, "M": 16, "ef_construction": 100}),
        ("usearch", "ef128", {"ef": 128, "M": 16, "ef_construction": 100}),
        ("diskann", "L64", {"R": 32, "Lbuild": 100, "search_L": 64}),
        ("diskann", "L96", {"R": 32, "Lbuild": 100, "search_L": 96}),
        ("annoy", "sk8000_re4", {"search_k": 8000, "n_trees": 10, "rebuild_every": 4}),
    ]
    raw_rows: List[Dict[str, Any]] = []
    for sys_name, ptag, params in configs_sem:
        log(f"[B3 / arXiv-384] Running {sys_name}_{ptag} (Core 0)...")
        r = _invoke_worker_subproc(worker_py, sem_npz, sys_name, ptag, params, core=0, timeout_s=1800)
        raw_rows.append({"workload": "arxiv_cs_minilm384", **r})
        _write_csv(_finalize_engine_rows(raw_rows), out_dir / "arxiv_5engines_summary.csv")
        log(
            f"  -> {sys_name}_{ptag}: rec={r['mean_recall']} minR={r['min_recall']} "
            f"att0965={r['attainment_0965_pct']}% rss={r['peak_rss_mb']}MB "
            f"split={r['query_cpu_pct']}%/{r['maint_cpu_pct']}%"
        )


def run_sift1m_suite(
    work_dir: Path, out_dir: Path, worker_py: str, target_workloads: Optional[Sequence[str]] = None
) -> None:
    s1m_paths = build_sift1m_workloads(work_dir)
    if target_workloads:
        s1m_paths = {k: v for k, v in s1m_paths.items() if k in target_workloads}
    configs_s1m = [
        ("faiss_ivf", "np16", {"nlist": 512, "nprobe": 16, "retrain_every": 0}),
        ("faiss_ivf", "np32", {"nlist": 512, "nprobe": 32, "retrain_every": 0}),
        ("hnswlib", "ef64", {"ef": 64, "M": 16, "ef_construction": 100}),
        ("hnswlib", "ef128", {"ef": 128, "M": 16, "ef_construction": 100}),
        ("usearch", "ef64", {"ef": 64, "M": 16, "ef_construction": 100}),
        ("usearch", "ef128", {"ef": 128, "M": 16, "ef_construction": 100}),
        ("diskann", "L64", {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": 0}),
        ("diskann", "L96", {"R": 32, "Lbuild": 100, "search_L": 96, "consolidate_every": 0}),
        ("diskann", "L64_cons10", {"R": 32, "Lbuild": 100, "search_L": 64, "consolidate_every": 10}),
        ("annoy", "sk16000_re10", {"search_k": 16000, "n_trees": 10, "rebuild_every": 10}),
    ]
    csv_path = out_dir / "sift1m_5engines_summary.csv"
    prior_rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        import pandas as pd
        df_old = pd.read_csv(csv_path)
        prior_rows = [r for r in df_old.to_dict(orient="records") if r.get("workload") not in s1m_paths]

    raw_rows: List[Dict[str, Any]] = list(prior_rows)
    for wl_name, npz_p in s1m_paths.items():
        wl_raw: List[Dict[str, Any]] = []
        for sys_name, ptag, params in configs_s1m:
            if wl_name == "sift1m_drift" and ptag == "L64_cons10":
                continue
            if wl_name == "sift1m_crud30" and ptag in ("np32", "L96"):
                continue
            log(f"[B1 / {wl_name}] Running {sys_name}_{ptag} (Core 0)...")
            r = _invoke_worker_subproc(worker_py, npz_p, sys_name, ptag, params, core=0, timeout_s=2400)
            wl_raw.append({"workload": wl_name, **r})
            combined = raw_rows + _finalize_engine_rows(wl_raw)
            _write_csv(combined, csv_path)
            log(
                f"  -> {wl_name} {sys_name}_{ptag}: rec={r['mean_recall']} minR={r['min_recall']} "
                f"att0965={r['attainment_0965_pct']}% rss={r['peak_rss_mb']}MB "
                f"split={r['query_cpu_pct']}%/{r['maint_cpu_pct']}%"
            )
        raw_rows.extend(_finalize_engine_rows(wl_raw))
    _write_csv(raw_rows, csv_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="PVLDB B3 (arXiv MiniLM-384) & B1 (1M 5-Engine Audit)")
    ap.add_argument("--work-dir", default="/content/pvldb_exp7")
    ap.add_argument("--out-dir", default="/content/pvldb_exp7/results")
    ap.add_argument("--phase", choices=["arxiv", "sift1m", "sift1m_drift", "sift1m_crud30", "all"], default="all")
    ap.add_argument("--worker-python", default="/content/venv311/bin/python")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--npz", default="")
    ap.add_argument("--system", default="")
    ap.add_argument("--param-tag", default="")
    ap.add_argument("--params-json", default="{}")
    ap.add_argument("--max-windows", type=int, default=0)
    args = ap.parse_args()

    if args.worker:
        res = run_single_engine_worker(
            Path(args.npz),
            args.system,
            args.param_tag,
            json.loads(args.params_json),
            max_windows=(args.max_windows if args.max_windows > 0 else None),
        )
        print("WORKER_JSON=" + json.dumps(res), flush=True)
        return 0

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    worker_py = args.worker_python if Path(args.worker_python).exists() else sys.executable
    meta = discover_environment(out_dir, worker_py)

    if args.phase in ("arxiv", "all"):
        run_arxiv_suite(work_dir, out_dir, worker_py)
    if args.phase in ("sift1m", "all"):
        run_sift1m_suite(work_dir, out_dir, worker_py)
    elif args.phase == "sift1m_drift":
        run_sift1m_suite(work_dir, out_dir, worker_py, target_workloads=["sift1m_drift"])
    elif args.phase == "sift1m_crud30":
        run_sift1m_suite(work_dir, out_dir, worker_py, target_workloads=["sift1m_crud30"])

    summary = {
        "status": "complete",
        "phase": args.phase,
        "gpu": meta["gpu"],
        "engines": meta["engines"],
        "artifacts": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "exp7_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log("ALL EXP7 GPU 1M & ARXIV AUDITS COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
