#!/usr/bin/env python3
"""Exp 11-B: budget pacing on live 16-shard delta-buffered engines (review round 3, storyline S7).

Same bridge as Exp 10-B2b (unmodified engine per shard + exact-scan delta buffer, real queries,
exact Top-10 GT); each greedy policy additionally has a ``*_paced`` variant whose cumulative
spend at block j is capped at floor(B * j / (J - 1)). Jobs are spread over isolated physical
cores (idle HT siblings), one single-threaded job per core.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_llm01_exp10b_full_audit import (  # noqa: E402
    K_BLOCK, SIFT_DATA, _extract_qd_fast, _mean_recall, _read_qbam_trace, _solve_milp_qd,
    _train_real_gbrt_causal, peak_rss_mb,
)


def worker_paced_bridge(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_root = Path(cfg["data_root"])
    tr_name = cfg["trace"]
    d = _read_qbam_trace(data_root, tr_name)
    dim, R, n_win, pool, init, ev, gt_ids, centroids = (
        d["dim"], d["R"], d["n_win"], d["pool"], d["init"], d["ev"], d["gt_ids"], d["centroids"]
    )
    c_norm2 = np.sum(centroids ** 2, axis=1)
    engine, policy = cfg["engine"], cfg["policy"]
    paced = policy.endswith("_paced")
    base_pol = policy[: -len("_paced")] if paced else policy
    BUDGET = int(cfg.get("budget", 5))

    gbrt_preds: Dict[int, np.ndarray] = {}
    if base_pol == "gbrt_causal":
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

    import faiss, diskannpy, hnswlib
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
        elif engine == "sharded_hnswlib":
            idx = hnswlib.Index(space="l2", dim=dim)
            idx.init_index(max_elements=len(shard_ids[r]) + 1000, ef_construction=100, M=16)
            idx.set_num_threads(1)
            idx.add_items(shard_base[r], shard_ids[r], num_threads=1)
            idx.set_ef(48)
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
    w_star = _solve_milp_qd(Q, D, K=K_BLOCK, B=BUDGET)
    J_blk = n_win // K_BLOCK
    spent = 0
    rr_ptr = 0
    ewma_q = np.zeros(R, dtype=np.float64)
    q_cpu_s = m_cpu_s = 0.0
    exact_ops = 0.0
    recs: List[float] = []
    actions = 0

    try:
        for w in range(n_win):
            j_now = w // K_BLOCK
            cap = BUDGET if not paced else int(np.floor(BUDGET * j_now / max(J_blk - 1, 1)))
            if w > 0 and w % K_BLOCK == 0 and spent < min(BUDGET, cap) and policy != "noop":
                j = j_now
                cur_d = np.array([len(shard_delta_ids[r]) for r in range(R)], dtype=np.float64)
                sel = -1
                if base_pol == "fixed_c2":
                    if (j - 1) % 2 == 0:
                        sel = rr_ptr % R
                        rr_ptr += 1
                elif base_pol == "past_qmass":
                    sel = int(np.argmax(Q[w - K_BLOCK : w].sum(axis=0) * cur_d))
                elif base_pol == "ewma_a03":
                    sel = int(np.argmax(ewma_q * cur_d))
                elif base_pol == "gbrt_causal":
                    sel = int(np.argmax(gbrt_preds[j] * cur_d))
                elif base_pol == "imbalance_prio":
                    base_lens = np.array([max(len(shard_ids[r]), 1) for r in range(R)], dtype=np.float64)
                    sel = int(np.argmax((Q[w - K_BLOCK : w].sum(axis=0) ** 0.7) * (cur_d ** 1.2) / np.sqrt(base_lens)))
                elif base_pol == "oracle":
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
                    elif engine == "sharded_hnswlib":
                        ii, dd = shard_idx[r].knn_query(qv[None, :], k=min(10, len(shard_ids[r])), num_threads=1)
                        for cid, dv in zip(ii[0].tolist(), dd[0].tolist()):
                            if cid >= 0:
                                cands.append((float(dv), int(cid)))
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
        "bridge_type": "sharded_delta_buffer_paced" if paced else "sharded_delta_buffer",
        "budget": BUDGET, "paced": paced,
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


def build_jobs(engines: Tuple[str, ...] = ("sharded_faiss_ivf", "sharded_usearch")) -> List[Dict[str, Any]]:
    traces = [(SIFT_DATA / "sift100k_u10_q2500_v2", f"abrupt_s0{k}", 20) for k in (5, 6, 7, 8, 9)]
    for tsw in (16, 24, 28):
        for k in (5, 6):
            if (SIFT_DATA / f"tsw{tsw}" / "sift100k_u10_q2500_v2" / "traces" / f"abrupt_s0{k}").exists():
                traces.append((SIFT_DATA / f"tsw{tsw}" / "sift100k_u10_q2500_v2", f"abrupt_s0{k}", tsw))
    traces += [(SIFT_DATA / "sift100k_u10_q2500_v2", "hotspot_s05", -1), (SIFT_DATA / "sift100k_u10_q2500_v2", "gradual_s05", -1)]
    pols = ["noop", "fixed_c2", "past_qmass", "past_qmass_paced", "ewma_a03", "ewma_a03_paced", "oracle", "oracle_paced"]
    jobs = []
    for eng in engines:
        for droot, tr, tsw in traces:
            for pol in pols:
                jobs.append({"data_root": str(droot), "trace": tr, "tsw": tsw, "engine": eng, "policy": pol, "budget": 5})
    return jobs


def job_key(j: Dict[str, Any]) -> str:
    return f"{j['engine']}__tsw{j['tsw']}__{j['trace']}__{j['policy']}__B{j['budget']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-json", default="")
    ap.add_argument("--cores", default="40,42,44,46,48,50")
    ap.add_argument("--engines", default="sharded_faiss_ivf,sharded_usearch")
    ap.add_argument("--out-csv", default="exp11b_paced_bridge.csv")
    ap.add_argument("--done-flag", default="EXP11B_DONE.flag")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results_exp11b"))
    args = ap.parse_args()
    if args.worker_json:
        cfg = json.loads(args.worker_json)
        tsw = cfg.pop("tsw")
        out = worker_paced_bridge(cfg)
        out["tsw"] = tsw
        print("WORKER_JSON=" + json.dumps(out), flush=True)
        return 0
    out_dir = Path(args.out_dir)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    engines = tuple(e.strip() for e in args.engines.split(",") if e.strip())
    todo = [j for j in build_jobs(engines) if not (out_dir / "cache" / f"{job_key(j)}.json").exists()]
    cores = [c.strip() for c in args.cores.split(",") if c.strip()]
    running: Dict[str, Tuple[subprocess.Popen, Dict[str, Any]]] = {}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = (out_dir / "exp11b.log").open("a", encoding="utf-8")
    while todo or running:
        for c in cores:
            if c not in running and todo:
                j = todo.pop(0)
                cmd = ["taskset", "-c", c, sys.executable, str(Path(__file__).resolve()), "--worker-json", json.dumps(j)]
                running[c] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env), j)
        time.sleep(2.0)
        for c in list(running):
            p, j = running[c]
            if p.poll() is None:
                continue
            so, se = p.communicate()
            del running[c]
            res = None
            for line in reversed(so.splitlines()):
                if line.startswith("WORKER_JSON="):
                    res = json.loads(line.split("=", 1)[1])
                    break
            if res is None:
                log.write(f"[{time.strftime('%F %T')}] FAIL core={c} {job_key(j)}: {se[-300:]}\n")
            else:
                res["core"] = c
                (out_dir / "cache" / f"{job_key(j)}.json").write_text(json.dumps(res) + "\n", encoding="utf-8")
                log.write(f"[{time.strftime('%F %T')}] core={c} {job_key(j)} cpu={res['total_cpu_s']} gcr={res['gcr_ops_pct']} att={res['att_0965_pct']} w/C={res['wall_cpu_ratio']}\n")
            log.flush()
    rows = [json.loads(p.read_text()) for p in sorted((out_dir / "cache").glob("*.json"))]
    import csv
    with (out_dir / args.out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    (out_dir / args.done_flag).write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
