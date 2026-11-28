#!/usr/bin/env python3
"""Orchestrator for Plan A+B Rust Engines, Zero-FFI Control, and 100x (100K -> 10M) BigANN Scaling on Xeon-8352V."""
from __future__ import annotations
import sys

import argparse
import concurrent.futures
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

RUST_BIN_AUDIT = Path(os.environ.get("DYNVECAUDIT_RUST_AUDIT", str(Path(__file__).resolve().parents[1] / "rust" / "target" / "release" / "dynvec-audit")))
RUST_BIN_BENCH = Path(os.environ.get("DYNVECAUDIT_RUST_BENCH", str(Path(__file__).resolve().parents[1] / "rust" / "target" / "release" / "dynvec-bench")))
PY_ENGINES = Path(os.environ.get("DYNVECAUDIT_PYTHON", sys.executable))

SIFT100K_ROOTS = {
    "drift": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_v2",
    "crud30v2": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_crud30v2",
    "crud50v2": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_crud50v2",
    "runbook": Path(os.environ.get("DYNVECAUDIT_DATA_ROOT", "./data")) / "sift100k_u10_q2500_runbook",
}
BIGANN_SRC = Path(os.environ.get("DYNVECAUDIT_BIGANN_SRC", "./data/sift10m"))
BIGANN_WL_ROOT = Path(os.environ.get("DYNVECAUDIT_BIGANN_WL", "./data/bigann_workloads"))
RESULTS_DIR = Path(os.environ.get("DYNVECAUDIT_RESULTS_DIR", "./results/rust"))


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


def read_load1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    except Exception:
        return 0.0


def wait_quiet_host(max_load: float = 20.0, checks: int = 2) -> None:
    streak = 0
    while streak < checks:
        if read_load1m() <= max_load:
            streak += 1
            if streak < checks:
                time.sleep(1.5)
        else:
            streak = 0
            time.sleep(8.0)


def run_rust_bench_list(
    mode: str,
    engine: str,
    data_dir: Path,
    ef_list: List[int],
    trace: str = "",
    src_dir: Optional[Path] = None,
    m: int = 16,
    ef_c: int = 200,
    threads: int = 1,
    cores: str = "0",
    batch_query: bool = False,
    max_windows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cmd = [
        str(RUST_BIN_BENCH),
        "--mode",
        mode,
        "--engine",
        engine,
        "--data-dir",
        str(data_dir),
        "--m",
        str(m),
        "--ef-construction",
        str(ef_c),
        "--ef-list",
        ",".join(str(x) for x in ef_list),
        "--threads",
        str(threads),
        "--cores",
        cores,
    ]
    if trace:
        cmd.extend(["--trace", trace])
    if src_dir is not None:
        cmd.extend(["--src-dir", str(src_dir)])
    if batch_query:
        cmd.append("--batch-query")
    if max_windows is not None:
        cmd.extend(["--max-windows", str(max_windows)])
    pr = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(pr.stdout)


def run_rust_bench(
    mode: str,
    engine: str,
    data_dir: Path,
    trace: str = "",
    src_dir: Optional[Path] = None,
    m: int = 16,
    ef_c: int = 200,
    ef: int = 64,
    threads: int = 1,
    cores: str = "0",
    batch_query: bool = False,
    max_windows: Optional[int] = None,
) -> Dict[str, Any]:
    return run_rust_bench_list(
        mode=mode,
        engine=engine,
        data_dir=data_dir,
        ef_list=[ef],
        trace=trace,
        src_dir=src_dir,
        m=m,
        ef_c=ef_c,
        threads=threads,
        cores=cores,
        batch_query=batch_query,
        max_windows=max_windows,
    )[0]


# ---------------------------------------------------------------------------
# Stage 1 Verification: Smoke + Rust CLI vs Python Speedup
# ---------------------------------------------------------------------------
def run_smoke_and_speedup() -> None:
    log("=== Running Stage 1 Smoke Test (AVX-512 Native) & CLI Speedup ===")
    smoke = {}
    for eng in ("usearch-rust", "hnsw-rs"):
        res = run_rust_bench(
            "qbam",
            eng,
            SIFT100K_ROOTS["drift"],
            trace="abrupt_s05",
            ef=64,
            threads=1,
            cores="0",
            max_windows=5,
        )
        log(f"Smoke {eng}: rec={res['mean_recall']:.5f}, cpu={res['total_cpu_s']:.2f}s, wall={res['wall_s']:.2f}s, rss={res['peak_rss_mb']} MB, w/c={res['wall_cpu_ratio']}")
        assert res["mean_recall"] >= 0.96, f"{eng} smoke recall too low: {res['mean_recall']}"
        assert res["peak_rss_mb"] > 0.0
        assert res["wall_cpu_ratio"] <= 1.15, f"{eng} smoke wall/CPU={res['wall_cpu_ratio']}"
        smoke[eng] = res

    tr_dir = BIGANN_WL_ROOT / "partition_traces"
    out_scale_json = RESULTS_DIR / "rust_scale_replay_raw.json"
    t0 = time.perf_counter()
    subprocess.run(
        [str(RUST_BIN_AUDIT), "scale-replay", "--trace-dir", str(tr_dir), "--budget", "5", "--out-json", str(out_scale_json)],
        check=True,
    )
    rust_replay_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    files = sorted(tr_dir.glob("*.json"))
    for fp in files:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        R, T, K = obj["R"], obj["T"], obj["K"]
        J = T // K
        Q, D = obj["Q"], obj["D"]
        for pol in ("noop", "fixed_c2", "fixed_c4", "past_qmass", "ewma_a03", "max_delta", "oracle_1blk"):
            d_cum = [0.0] * R
            q_prev = [0.0] * R
            q_ewma = [0.0] * R
            spent = 0
            rr = 0
            for j in range(J):
                if pol.startswith("fixed_c"):
                    c_step = int(pol.replace("fixed_c", ""))
                    if j % c_step == 0:
                        sh = rr % R
                        rr += 1
                        if d_cum[sh] > 0:
                            d_cum[sh] = 0.0
                elif pol != "noop" and j > 0 and spent < 5:
                    best_sh, best_v = None, 0.0
                    for sh in range(R):
                        if d_cum[sh] <= 0.0:
                            continue
                        qh = q_prev[sh] if pol == "past_qmass" else (q_ewma[sh] if pol == "ewma_a03" else (1.0 if pol == "max_delta" else sum(Q[t][sh] for t in range(j * K, (j + 1) * K))))
                        sc = qh * d_cum[sh]
                        if sc > best_v:
                            best_v, best_sh = sc, sh
                    if best_sh is not None:
                        d_cum[best_sh] = 0.0
                        spent += 1
                q_prev = [0.0] * R
                for t in range(j * K, (j + 1) * K):
                    qt, dt = Q[t], D[t]
                    for sh in range(R):
                        d_cum[sh] += dt[sh]
                        q_prev[sh] += qt[sh]
                for sh in range(R):
                    q_ewma[sh] = q_prev[sh] if j == 0 else (0.3 * q_prev[sh] + 0.7 * q_ewma[sh])
    py_replay_ms = (time.perf_counter() - t1) * 1000.0

    canon_csv = Path(os.environ.get("DYNVECAUDIT_RESULTS_DIR", "./results")) / "sprint_v2/canon25_summary.csv"
    rust_audit_ms = None
    py_audit_ms = None
    if canon_csv.exists():
        t2 = time.perf_counter()
        subprocess.run([str(RUST_BIN_AUDIT), "audit-csv", "--csv", str(canon_csv)], check=True, capture_output=True)
        rust_audit_ms = (time.perf_counter() - t2) * 1000.0
        t3 = time.perf_counter()
        with canon_csv.open() as f:
            r = list(csv.DictReader(f))
            _ = max(float(x.get("wall_s", 0)) / max(float(x.get("total_s", 1)), 1e-9) for x in r if x.get("total_s"))
        py_audit_ms = (time.perf_counter() - t3) * 1000.0

    speedup_obj = {
        "smoke": {k: {"recall": v["mean_recall"], "cpu_s": v["total_cpu_s"], "wall_cpu": v["wall_cpu_ratio"], "rss_mb": v["peak_rss_mb"]} for k, v in smoke.items()},
        "scale_replay_36_traces_252_runs": {
            "rust_cli_wall_ms": round(rust_replay_ms, 3),
            "python_ref_wall_ms": round(py_replay_ms, 3),
            "speedup_x": round(py_replay_ms / max(rust_replay_ms, 1e-6), 2),
        },
        "audit_csv_canon25": {
            "rust_cli_wall_ms": round(rust_audit_ms, 3) if rust_audit_ms else None,
            "python_ref_wall_ms": round(py_audit_ms, 3) if py_audit_ms else None,
            "speedup_x": round(py_audit_ms / rust_audit_ms, 2) if (rust_audit_ms and py_audit_ms) else None,
        },
    }
    (RESULTS_DIR / "rust_cli_speedup.json").write_text(json.dumps(speedup_obj, indent=2), encoding="utf-8")
    log(f"Speedup saved: {speedup_obj}")


# ---------------------------------------------------------------------------
# Suite R1: Rust 7th Engine (hnsw_rs) + Zero-FFI (usearch_rust) on SIFT-100K
# ---------------------------------------------------------------------------
def _spawn_rust_worker(cmd_args: List[str]) -> Dict[str, Any]:
    pr = subprocess.run(cmd_args, capture_output=True, text=True, check=True)
    return json.loads(pr.stdout)


def run_suite_r1() -> None:
    log("=== Suite R1: Rust 7th Engine (hnsw_rs) + USearch-Rust (Core 0 & Trap 4) ===")
    wait_quiet_host()

    # R1-A4: Single-build --ef-list 32,64,96,128 across 3 traces (Core 0, full 40w)
    r1_a4_rows: List[Dict[str, Any]] = []
    for trace in ("abrupt_s05", "hotspot_s05", "gradual_s05"):
        for eng in ("usearch-rust", "hnsw-rs"):
            wait_quiet_host()
            res_list = run_rust_bench_list(
                "qbam", eng, SIFT100K_ROOTS["drift"], [32, 64, 96, 128], trace=trace, ef_c=200, threads=1, cores="0"
            )
            for res in res_list:
                res["suite"] = "R1-A4"
                res["regime"] = "iso_numa0_1w"
                res["config"] = f"ef{res['ef']}"
                r1_a4_rows.append(res)
                log(f"[R1-A4] {trace} {res['engine']} ef{res['ef']}: cpu={res['total_cpu_s']:.2f}s rec={res['mean_recall']:.4f} att={res['att_0965_pct']}% p99={res['p99_us']}us w/c={res['wall_cpu_ratio']:.3f}")
            write_csv(r1_a4_rows, RESULTS_DIR / "r1_rust_a4_floor.csv")

    # R1-A3: Single-build CRUD sweep (usearch_rust ef64/ef96; hnsw_rs unsupported_erase 10w verification)
    r1_a3_rows: List[Dict[str, Any]] = []
    crud_specs = [
        ("crud30v2", "abrupt_s05"),
        ("crud30v2", "hotspot_s05"),
        ("crud50v2", "abrupt_s05"),
        ("crud50v2", "hotspot_s05"),
        ("runbook", "clustered_s05"),
        ("runbook", "sliding_s05"),
    ]
    for ds_key, tr in crud_specs:
        wait_quiet_host()
        for eng, efs, mw in [("usearch-rust", [64, 96], None), ("hnsw-rs", [64], 10)]:
            res_list = run_rust_bench_list(
                "qbam", eng, SIFT100K_ROOTS[ds_key], efs, trace=tr, ef_c=200, threads=1, cores="0", max_windows=mw
            )
            for res in res_list:
                res["suite"] = "R1-A3"
                res["dataset"] = ds_key
                res["config"] = f"ef{res['ef']}"
                r1_a3_rows.append(res)
                log(f"[R1-A3] {ds_key}/{tr} {res['engine']} ef{res['ef']}: cpu={res['total_cpu_s']:.2f}s rec={res['mean_recall']:.4f} att={res['att_0965_pct']}% status={res['status']}")
            write_csv(r1_a3_rows, RESULTS_DIR / "r1_rust_a3_crud.csv")

    # R1-A1+D: 10-window Trap 4 NUMA/HT/MT apples-to-apples with exp6 A1+D
    r1_a1d_rows: List[Dict[str, Any]] = []
    regimes = [
        ("iso_numa0_1w", 1, ["0"], False),
        ("ht_sibling_2w", 1, ["0", "64"], False),
        ("same_numa_4w", 1, ["0", "1", "2", "3"], False),
        ("cross_numa_8w", 1, ["0", "1", "2", "3", "32", "33", "34", "35"], False),
        ("mt_same_numa_4t", 4, ["0-3"], True),
        ("mt_unpinned_8t", 8, ["0-127"], True),
    ]
    for eng, ef_use in [("usearch-rust", 128), ("hnsw-rs", 96)]:
        for reg_name, thr, core_specs, batch_q in regimes:
            wait_quiet_host()
            if len(core_specs) == 1:
                res = run_rust_bench(
                    "qbam", eng, SIFT100K_ROOTS["drift"], trace="abrupt_s05", ef_c=200, ef=ef_use, threads=thr, cores=core_specs[0], batch_query=batch_q, max_windows=10
                )
                res["suite"] = "R1-A1D"
                res["regime"] = reg_name
                res["worker_id"] = 0
                r1_a1d_rows.append(res)
            else:
                cmds = [
                    [
                        str(RUST_BIN_BENCH), "--mode", "qbam", "--engine", eng,
                        "--data-dir", str(SIFT100K_ROOTS["drift"]), "--trace", "abrupt_s05",
                        "--ef-construction", "200", "--ef", str(ef_use),
                        "--threads", str(thr), "--cores", c, "--max-windows", "10",
                    ]
                    for c in core_specs
                ]
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(cmds)) as ex:
                    futs = [(idx, ex.submit(_spawn_rust_worker, cmd)) for idx, cmd in enumerate(cmds)]
                    for idx, fu in futs:
                        r = fu.result()
                        r["suite"] = "R1-A1D"
                        r["regime"] = reg_name
                        r["worker_id"] = idx
                        r1_a1d_rows.append(r)
            last = [x for x in r1_a1d_rows if x["regime"] == reg_name and x["engine"] == eng.replace("-", "_")]
            mean_c = statistics.mean(x["total_cpu_s"] for x in last)
            mean_p99 = statistics.mean((x["p99_us"] or 0.0) for x in last)
            log(f"[R1-A1D] {eng} {reg_name}: mean_cpu={mean_c:.2f}s, mean_p99={mean_p99:.1f}us")
            write_csv(r1_a1d_rows, RESULTS_DIR / "r1_rust_a1d_numa.csv")


# ---------------------------------------------------------------------------
# Python Engine Subprocess Worker for BigANN (single-build multi-knob support!)
# ---------------------------------------------------------------------------
PY_BIGANN_WORKER = r"""
import concurrent.futures, json, resource, struct, sys, tempfile, time
from pathlib import Path
import numpy as np

args = json.loads(sys.argv[1])
src_dir = Path(args["src_dir"])
wl_dir = Path(args["wl_dir"])
system = args["system"]
params = args["params"]
knobs = args["knobs"]
threads = int(args["threads"])
batch_q = bool(args["batch_query"])

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = str(threads)

def cpu_ms():
    return time.process_time() * 1000.0

def peak_rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(float(line.split(":")[1].strip().split()[0]) / 1024.0, 2)
    except Exception:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)

def read_u8_mm(p):
    n, d = struct.unpack("<II", p.open("rb").read(8))
    return np.memmap(p, dtype=np.uint8, mode="r", offset=8, shape=(n, d))

base_mm = read_u8_mm(src_dir / "base_11m.bin")
ins_mm = read_u8_mm(src_dir / "insert_vectors.bin")
q_mm = read_u8_mm(src_dir / "query.bin")

with (wl_dir / "init.bin").open("rb") as f:
    magic = f.read(8); assert magic == b"BGNTINIT"
    n_init, dim, _ = struct.unpack("<QII", f.read(16))
    init_dt = np.dtype([("id", "<u8"), ("src", "u1"), ("row", "<u4")], align=False)
    init = np.fromfile(f, dtype=init_dt, count=n_init)

with (wl_dir / "events.bin").open("rb") as f:
    magic = f.read(8); assert magic == b"BGNTEVNT"
    n_ev, n_win, nq_pw, k = struct.unpack("<IIII", f.read(16))
    ev_dt = np.dtype([("window", "<u4"), ("type", "u1"), ("id", "<u8"), ("src", "u1"), ("row", "<u4")], align=False)
    ev = np.fromfile(f, dtype=ev_dt, count=n_ev)

with (wl_dir / "gt.bin").open("rb") as f:
    magic = f.read(8); assert magic == b"BGNTGT10"
    gw_n, gw_nq, gw_k, _ = struct.unpack("<IIII", f.read(16))
    q_rows = np.fromfile(f, dtype="<u4", count=gw_n * gw_nq)
    gt_ids = np.fromfile(f, dtype="<u8", count=gw_n * gw_nq * gw_k).reshape(gw_n, gw_nq, gw_k)

def fetch_vecs(src_arr, row_arr):
    if len(src_arr) == 0:
        return np.empty((0, dim), dtype=np.float32)
    if np.all(src_arr == 0):
        return np.ascontiguousarray(base_mm[row_arr], dtype=np.float32)
    if np.all(src_arr == 1):
        return np.ascontiguousarray(ins_mm[row_arr], dtype=np.float32)
    out = np.empty((len(src_arr), dim), dtype=np.float32)
    m0 = src_arr == 0
    if np.any(m0): out[m0] = base_mm[row_arr[m0]]
    if np.any(~m0): out[~m0] = ins_mm[row_arr[~m0]]
    return np.ascontiguousarray(out, dtype=np.float32)

tb_w0 = time.perf_counter(); t_b0 = cpu_ms()
disk_tmp = None
n_ins_tot = int(np.sum(ev["type"] == 0))
max_cap = len(init) + n_ins_tot + 50_000
init_ids = init["id"].astype(np.int64)

if system == "faiss_ivf":
    import faiss
    faiss.omp_set_num_threads(threads)
    nlist = int(params.get("nlist", 1024))
    quant = faiss.IndexFlatL2(dim)
    idx_obj = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
    tr_n = min(150_000, len(init))
    tr_idx = np.linspace(0, len(init) - 1, tr_n, dtype=np.int64)
    idx_obj.train(fetch_vecs(init["src"][tr_idx], init["row"][tr_idx]))
    for s in range(0, len(init), 1_000_000):
        e = min(len(init), s + 1_000_000)
        idx_obj.add_with_ids(fetch_vecs(init["src"][s:e], init["row"][s:e]), init_ids[s:e])
elif system == "hnswlib":
    import hnswlib
    idx_obj = hnswlib.Index(space="l2", dim=dim)
    idx_obj.init_index(max_elements=max_cap, ef_construction=int(params.get("ef_construction", 100)), M=int(params.get("M", 16)), allow_replace_deleted=True)
    idx_obj.set_num_threads(threads)
    for s in range(0, len(init), 1_000_000):
        e = min(len(init), s + 1_000_000)
        idx_obj.add_items(fetch_vecs(init["src"][s:e], init["row"][s:e]), init_ids[s:e], num_threads=threads)
elif system == "diskann":
    import diskannpy
    R = int(params.get("R", 32)); Lb = int(params.get("Lbuild", 100))
    disk_tmp = tempfile.TemporaryDirectory(prefix="pvldb_1m_da_")
    init_vecs = fetch_vecs(init["src"], init["row"])
    tags0 = (init_ids + 1).astype(np.uint32)
    diskannpy.build_memory_index(init_vecs, "l2", disk_tmp.name, complexity=Lb, graph_degree=R, num_threads=threads, tags=tags0)
    idx_obj = diskannpy.DynamicMemoryIndex.from_file(disk_tmp.name, max_vectors=max_cap, complexity=Lb, graph_degree=R, num_threads=threads, initial_search_complexity=int(knobs[0]["search_L"]), search_threads=threads, concurrent_consolidation=False)
    del init_vecs
else:
    raise ValueError(system)

build_cpu_s = (cpu_ms() - t_b0) / 1000.0
build_wall_s = time.perf_counter() - tb_w0

nk = len(knobs)
m_cpu_ms = 0.0; m_wall_s = 0.0
q_cpu_ms = [0.0] * nk; q_wall_s = [0.0] * nk
recalls = [[] for _ in range(nk)]
p50s = [[] for _ in range(nk)]; p95s = [[] for _ in range(nk)]; p99s = [[] for _ in range(nk)]
has_del = False

try:
    for w in range(n_win):
        w_ev = ev[ev["window"] == w]
        er_ev = w_ev[w_ev["type"] == 2]; in_ev = w_ev[w_ev["type"] == 0]
        tm_w0 = time.perf_counter(); tm0 = cpu_ms()
        if system == "faiss_ivf":
            import faiss; faiss.omp_set_num_threads(threads)
            if len(er_ev): idx_obj.remove_ids(er_ev["id"].astype(np.int64))
            if len(in_ev): idx_obj.add_with_ids(fetch_vecs(in_ev["src"], in_ev["row"]), in_ev["id"].astype(np.int64))
        elif system == "hnswlib":
            idx_obj.set_num_threads(threads)
            for tid in er_ev["id"].tolist():
                idx_obj.mark_deleted(int(tid)); has_del = True
            if len(in_ev):
                idx_obj.add_items(fetch_vecs(in_ev["src"], in_ev["row"]), in_ev["id"].astype(np.int64), num_threads=threads, replace_deleted=has_del)
        elif system == "diskann":
            for tid in er_ev["id"].tolist(): idx_obj.mark_deleted(np.uint32(int(tid) + 1))
            if len(in_ev):
                iv = fetch_vecs(in_ev["src"], in_ev["row"])
                for tid, v in zip(in_ev["id"].tolist(), iv): idx_obj.insert(v, np.uint32(int(tid) + 1))
            ce = int(params.get("consolidate_every", 0))
            if ce > 0 and (w + 1) % ce == 0: idx_obj.consolidate_delete()
        m_cpu_ms += cpu_ms() - tm0
        m_wall_s += time.perf_counter() - tm_w0

        qs = q_rows[w * nq_pw:(w + 1) * nq_pw]
        qvecs = np.ascontiguousarray(q_mm[qs], dtype=np.float32)
        gt_w = gt_ids[w]

        for ki, kb in enumerate(knobs):
            tq_w0 = time.perf_counter(); tq0 = cpu_ms()
            lat = []; pred = np.full((nq_pw, k), -1, dtype=np.int64)
            if system == "faiss_ivf": idx_obj.nprobe = int(kb["nprobe"])
            elif system == "hnswlib": idx_obj.set_ef(int(kb["ef"]))
            sL = int(kb.get("search_L", 64))

            if batch_q and threads > 1:
                if system == "faiss_ivf":
                    import faiss; faiss.omp_set_num_threads(1)
                    def _q_f(qi):
                        t0 = time.perf_counter_ns(); _, r = idx_obj.search(qvecs[qi:qi+1], k); return (time.perf_counter_ns() - t0)/1000.0, r[0]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
                        for qi, (us, row) in enumerate(ex.map(_q_f, range(nq_pw))): lat.append(us); pred[qi] = row
                    faiss.omp_set_num_threads(threads)
                elif system == "hnswlib":
                    def _q_h(qi):
                        t0 = time.perf_counter_ns(); r, _ = idx_obj.knn_query(qvecs[qi:qi+1], k=k, num_threads=1); return (time.perf_counter_ns() - t0)/1000.0, r[0]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
                        for qi, (us, row) in enumerate(ex.map(_q_h, range(nq_pw))): lat.append(us); pred[qi] = row
                elif system == "diskann":
                    t0 = time.perf_counter_ns(); res = idx_obj.batch_search(qvecs, k, sL, threads)
                    pred = np.asarray(res.identifiers, dtype=np.int64) - 1
                    lat = [((time.perf_counter_ns() - t0) / 1000.0) / nq_pw]
            else:
                for i in range(nq_pw):
                    t0 = time.perf_counter_ns()
                    if system == "faiss_ivf": _, row = idx_obj.search(qvecs[i:i+1], k); pred[i] = row[0]
                    elif system == "hnswlib": row, _ = idx_obj.knn_query(qvecs[i:i+1], k=k, num_threads=threads); pred[i] = row[0]
                    elif system == "diskann": res = idx_obj.search(qvecs[i], k, sL); pred[i] = np.asarray(res.identifiers, dtype=np.int64) - 1
                    lat.append((time.perf_counter_ns() - t0) / 1000.0)

            q_cpu_ms[ki] += cpu_ms() - tq0
            q_wall_s[ki] += time.perf_counter() - tq_w0
            hits = sum(len(set(pred[i].tolist()) & set(gt_w[i].tolist())) for i in range(nq_pw))
            recalls[ki].append(hits / float(nq_pw * k))
            lat.sort()
            p50s[ki].append(lat[int(round((len(lat) - 1) * 0.50))])
            p95s[ki].append(lat[int(round((len(lat) - 1) * 0.95))])
            p99s[ki].append(lat[int(round((len(lat) - 1) * 0.99))])
finally:
    if disk_tmp is not None:
        disk_tmp.cleanup()

rss = peak_rss_mb()
out = []
for ki, kb in enumerate(knobs):
    q_s = q_cpu_ms[ki] / 1000.0; m_s = m_cpu_ms / 1000.0; tot_s = q_s + m_s
    dyn_w = m_wall_s + q_wall_s[ki]; wall_s = build_wall_s + dyn_w
    out.append({
        "engine": system,
        "config": kb["tag"],
        "mode": "bigann",
        "trace": wl_dir.name,
        "threads": threads,
        "batch_query": batch_q,
        "n_windows": n_win,
        "status": "ok",
        "build_cpu_s": round(build_cpu_s, 4),
        "build_wall_s": round(build_wall_s, 4),
        "query_cpu_s": round(q_s, 4),
        "maint_cpu_s": round(m_s, 4),
        "total_cpu_s": round(tot_s, 4),
        "dyn_wall_s": round(dyn_w, 4),
        "wall_s": round(wall_s, 4),
        "wall_cpu_ratio": round(wall_s / max(tot_s + build_cpu_s, 1e-6), 4),
        "query_pct": round(100.0 * q_s / max(tot_s, 1e-9), 2),
        "maint_pct": round(100.0 * m_s / max(tot_s, 1e-9), 2),
        "mean_recall": round(float(np.mean(recalls[ki])), 5),
        "min_recall": round(float(np.min(recalls[ki])), 5),
        "att_0965_pct": round(100.0 * sum(1 for r in recalls[ki] if r >= 0.965) / n_win, 1),
        "att_0950_pct": round(100.0 * sum(1 for r in recalls[ki] if r >= 0.950) / n_win, 1),
        "p50_us": round(float(np.mean(p50s[ki])), 2),
        "p95_us": round(float(np.mean(p95s[ki])), 2),
        "p99_us": round(float(np.mean(p99s[ki])), 2),
        "peak_rss_mb": rss,
    })
print(json.dumps(out))
"""


def run_py_bigann_list(wl_name: str, system: str, params: Dict[str, Any], knobs: List[Dict[str, Any]], threads: int, cores: str, batch_query: bool = False) -> List[Dict[str, Any]]:
    payload = json.dumps({
        "src_dir": str(BIGANN_SRC),
        "wl_dir": str(BIGANN_WL_ROOT / wl_name),
        "system": system,
        "params": params,
        "knobs": knobs,
        "threads": threads,
        "batch_query": batch_query,
    })
    cmd = ["taskset", "-c", cores, str(PY_ENGINES), "-c", PY_BIGANN_WORKER, payload]
    pr = subprocess.run(cmd, capture_output=True, text=True, check=True)
    res_list = json.loads(pr.stdout.strip().splitlines()[-1])
    for r in res_list:
        r["cores"] = cores
    return res_list


# ---------------------------------------------------------------------------
# Suite S1: Gate-4 Clean Dual Xeon Core-0 1M Audit
# ---------------------------------------------------------------------------
def run_suite_s1() -> None:
    log("=== Suite S1: Gate-4 Clean Dual Xeon Core-0 BigANN-1M (Drift & CRUD-20%) ===")
    wait_quiet_host()
    rows: List[Dict[str, Any]] = []
    for wl in ("bigann_1m_drift", "bigann_1m_crud20"):
        wait_quiet_host()
        # 1. FAISS-IVF (np16, np32, np64) single-build
        for r in run_py_bigann_list(
            wl, "faiss_ivf", {"nlist": 1024},
            [{"tag": "np16", "nprobe": 16}, {"tag": "np32", "nprobe": 32}, {"tag": "np64", "nprobe": 64}],
            threads=1, cores="0",
        ):
            r["suite"] = "S1-1M-Core0"; rows.append(r)
            log(f"[S1] {wl} {r['engine']} {r['config']}: cpu={r['total_cpu_s']:.2f}s rec={r['mean_recall']:.4f} att={r['att_0965_pct']}% p99={r['p99_us']}us w/c={r['wall_cpu_ratio']:.3f}")
        # 2. hnswlib (ef64, ef128, ef_c=100) single-build
        for r in run_py_bigann_list(
            wl, "hnswlib", {"M": 16, "ef_construction": 100},
            [{"tag": "ef64", "ef": 64}, {"tag": "ef128", "ef": 128}],
            threads=1, cores="0",
        ):
            r["suite"] = "S1-1M-Core0"; rows.append(r)
            log(f"[S1] {wl} {r['engine']} {r['config']}: cpu={r['total_cpu_s']:.2f}s rec={r['mean_recall']:.4f} att={r['att_0965_pct']}% p99={r['p99_us']}us w/c={r['wall_cpu_ratio']:.3f}")
        # 3. FreshDiskANN (L64 / cons1 on CRUD)
        d_tag = "L64_cons1" if "crud" in wl else "L64"
        for r in run_py_bigann_list(
            wl, "diskann", {"R": 32, "Lbuild": 100, "consolidate_every": 1 if "crud" in wl else 0},
            [{"tag": d_tag, "search_L": 64}],
            threads=1, cores="0",
        ):
            r["suite"] = "S1-1M-Core0"; rows.append(r)
            log(f"[S1] {wl} {r['engine']} {r['config']}: cpu={r['total_cpu_s']:.2f}s rec={r['mean_recall']:.4f} att={r['att_0965_pct']}% p99={r['p99_us']}us w/c={r['wall_cpu_ratio']:.3f}")
        # 4. USearch-Rust (ef64, ef128, ef_c=100) single-build
        for r in run_rust_bench_list(
            "bigann", "usearch-rust", BIGANN_WL_ROOT / wl, [64, 128], src_dir=BIGANN_SRC, ef_c=100, threads=1, cores="0"
        ):
            r["suite"] = "S1-1M-Core0"; r["config"] = f"ef{r['ef']}"; rows.append(r)
            log(f"[S1] {wl} {r['engine']} {r['config']}: cpu={r['total_cpu_s']:.2f}s rec={r['mean_recall']:.4f} att={r['att_0965_pct']}% p99={r['p99_us']}us w/c={r['wall_cpu_ratio']:.3f}")
        # 5. hnsw_rs on Drift
        if "drift" in wl:
            for r in run_rust_bench_list(
                "bigann", "hnsw-rs", BIGANN_WL_ROOT / wl, [64, 128], src_dir=BIGANN_SRC, ef_c=100, threads=1, cores="0"
            ):
                r["suite"] = "S1-1M-Core0"; r["config"] = f"ef{r['ef']}"; rows.append(r)
                log(f"[S1] {wl} {r['engine']} {r['config']}: cpu={r['total_cpu_s']:.2f}s rec={r['mean_recall']:.4f} att={r['att_0965_pct']}% p99={r['p99_us']}us w/c={r['wall_cpu_ratio']:.3f}")
        write_csv(rows, RESULTS_DIR / "s1_xeon_1m_gate4_clean.csv")


# ---------------------------------------------------------------------------
# Suite S2: 1M -> 2M -> 5M -> 10M Scale + 10M NUMA Trap 4 + 100x MILP Law
# ---------------------------------------------------------------------------
def solve_milp_exact_bound(obj: Dict[str, Any], budget: int) -> float:
    R, T, K = obj["R"], obj["T"], obj["K"]
    J = T // K
    Q, D, C = np.asarray(obj["Q"], dtype=float), np.asarray(obj["D"], dtype=float), np.asarray(obj["compaction_ops_per_shard"], dtype=float)
    D_cum_no_comp = np.zeros((T, R), dtype=float)
    for t in range(T):
        D_cum_no_comp[t] = (D_cum_no_comp[t - 1] if t > 0 else 0.0) + D[t]

    def seg_cost(r: int, j1: int, j2: int) -> float:
        tau = j1 * K
        base_d = D_cum_no_comp[tau - 1, r] if tau > 0 else 0.0
        sc = 0.0
        for t in range(tau, K * j2):
            sc += Q[t, r] * max(0.0, D_cum_no_comp[t, r] - base_d)
        maint = C[r] + (base_d if j1 > 0 else 0.0)
        return float(sc + (maint if j1 > 0 else 0.0))

    segs = []
    for r in range(R):
        for j2 in range(1, J + 1):
            segs.append((r, 0, j2, seg_cost(r, 0, j2)))
        for j1 in range(1, J):
            for j2 in range(j1 + 1, J + 1):
                segs.append((r, j1, j2, seg_cost(r, j1, j2)))
    n_var = len(segs)
    c_vec = np.array([s[3] for s in segs], dtype=float)
    rows_a, lb, ub = [], [], []
    for r in range(R):
        row = np.zeros(n_var)
        for i, (sr, j1, _, _) in enumerate(segs):
            if sr == r and j1 == 0: row[i] = 1.0
        rows_a.append(row); lb.append(1.0); ub.append(1.0)
        for j in range(1, J):
            rf = np.zeros(n_var)
            for i, (sr, j1, j2, _) in enumerate(segs):
                if sr == r:
                    if j2 == j: rf[i] += 1.0
                    if j1 == j: rf[i] -= 1.0
            rows_a.append(rf); lb.append(0.0); ub.append(0.0)
    for j in range(1, J):
        rs = np.zeros(n_var)
        for i, (_, j1, _, _) in enumerate(segs):
            if j1 == j: rs[i] = 1.0
        rows_a.append(rs); lb.append(0.0); ub.append(1.0)
    rb = np.zeros(n_var)
    for i, (_, j1, _, _) in enumerate(segs):
        if j1 > 0: rb[i] = 1.0
    rows_a.append(rb); lb.append(0.0); ub.append(float(budget))
    res = milp(c=c_vec, integrality=np.ones(n_var), bounds=Bounds(0, 1), constraints=LinearConstraint(np.vstack(rows_a), lb, ub))
    return float(res.fun)


def run_suite_s2() -> None:
    log("=== Suite S2: 1M -> 2M -> 5M -> 10M Scale + 10M NUMA Trap 4 + 100x MILP ===")
    wait_quiet_host()

    s2_rows: List[Dict[str, Any]] = []
    scale_cfgs = [
        ("1m", 1024, 32),
        ("2m", 2048, 48),
        ("5m", 4096, 64),
        ("10m", 4096, 64),
    ]
    hnswlib_5m_crud_wall = 0.0
    hnsw_rs_5m_wall = 0.0
    iso_10m_cache: Dict[str, Dict[str, Any]] = {}

    for stag, nlist, np_target in scale_cfgs:
        for wl_type in ("drift", "crud20"):
            wl = f"bigann_{stag}_{wl_type}"
            specs: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = [
                ("faiss_ivf", {"nlist": nlist}, [{"tag": f"np{np_target}", "nprobe": np_target}]),
                ("usearch_rust", {"ef_construction": 100}, [{"tag": "ef96", "ef": 96}]),
            ]
            if not (stag == "10m" and wl_type == "crud20" and hnswlib_5m_crud_wall > 300.0):
                specs.append(("hnswlib", {"M": 16, "ef_construction": 100}, [{"tag": "ef96", "ef": 96}]))
            if wl_type == "drift" and not (stag == "10m" and hnsw_rs_5m_wall > 600.0):
                specs.append(("hnsw_rs", {"ef_construction": 100}, [{"tag": "ef96", "ef": 96}]))

            for sys_name, par, knobs in specs:
                wait_quiet_host()
                if sys_name in ("usearch_rust", "hnsw_rs"):
                    res = run_rust_bench(
                        "bigann", sys_name.replace("_", "-"), BIGANN_WL_ROOT / wl, src_dir=BIGANN_SRC, ef_c=100, ef=96, threads=8, cores="0-7", batch_query=True
                    )
                    res["config"] = "ef96"
                else:
                    res = run_py_bigann_list(wl, sys_name, par, knobs, threads=8, cores="0-7", batch_query=True)[0]
                res["suite"] = "S2-Scale"
                res["scale"] = stag.upper()
                res["wl_type"] = wl_type
                s2_rows.append(res)
                if stag == "5m" and wl_type == "crud20" and sys_name == "hnswlib":
                    hnswlib_5m_crud_wall = float(res["wall_s"])
                if stag == "5m" and wl_type == "drift" and sys_name == "hnsw_rs":
                    hnsw_rs_5m_wall = float(res["wall_s"])
                if stag == "10m" and wl_type == "drift":
                    c_iso = dict(res)
                    c_iso["suite"] = "S2-NUMA10M"
                    c_iso["regime"] = "iso_numa0_8t"
                    iso_10m_cache[sys_name] = c_iso
                log(f"[S2-Scale] {wl} {sys_name} {res['config']}: build_w={res['build_wall_s']:.2f}s dyn_w={res['dyn_wall_s']:.2f}s rec={res['mean_recall']:.4f} att={res['att_0965_pct']}% p99={res['p99_us']}us rss={res['peak_rss_mb']}MB")
                write_csv(s2_rows, RESULTS_DIR / "s2_scale_1m_to_10m.csv")

    # S2-NUMA10M: Trap 4 at 10M Scale across available 10M engines (reusing iso_numa0_8t)
    numa_rows: List[Dict[str, Any]] = []
    numa_regimes = [
        ("ht_sibling_8t", 8, "0-3,64-67"),
        ("cross_numa_8t", 8, "0-3,32-35"),
        ("unpinned_16t", 16, "0-127"),
    ]
    wl_10m = "bigann_10m_drift"
    eng_10m = [
        ("faiss_ivf", {"nlist": 4096}, [{"tag": "np64", "nprobe": 64}]),
        ("usearch_rust", {"ef_construction": 100}, [{"tag": "ef96", "ef": 96}]),
        ("hnswlib", {"M": 16, "ef_construction": 100}, [{"tag": "ef96", "ef": 96}]),
    ]
    if "hnsw_rs" in iso_10m_cache:
        eng_10m.append(("hnsw_rs", {"ef_construction": 100}, [{"tag": "ef96", "ef": 96}]))

    for sys_name, par, knobs in eng_10m:
        if sys_name in iso_10m_cache:
            numa_rows.append(iso_10m_cache[sys_name])
            write_csv(numa_rows, RESULTS_DIR / "s2_numa_10m_trap4.csv")
        for reg_name, thr, cores in numa_regimes:
            wait_quiet_host()
            if sys_name in ("usearch_rust", "hnsw_rs"):
                res = run_rust_bench(
                    "bigann", sys_name.replace("_", "-"), BIGANN_WL_ROOT / wl_10m, src_dir=BIGANN_SRC, ef_c=100, ef=96, threads=thr, cores=cores, batch_query=True
                )
                res["config"] = "ef96"
            else:
                res = run_py_bigann_list(wl_10m, sys_name, par, knobs, threads=thr, cores=cores, batch_query=True)[0]
            res["suite"] = "S2-NUMA10M"
            res["regime"] = reg_name
            numa_rows.append(res)
            log(f"[S2-NUMA10M] {sys_name} {reg_name}: dyn_w={res['dyn_wall_s']:.2f}s cpu={res['total_cpu_s']:.2f}s p99={res['p99_us']}us rec={res['mean_recall']:.4f}")
            write_csv(numa_rows, RESULTS_DIR / "s2_numa_10m_trap4.csv")

    # S2-Headroom: Combine Rust scale-replay + exact Theorem-1 MILP for 1M -> 10M
    raw_replay = json.loads((RESULTS_DIR / "rust_scale_replay_raw.json").read_text(encoding="utf-8"))
    tr_dir = BIGANN_WL_ROOT / "partition_traces"
    milp_cache: Dict[str, Tuple[float, float]] = {}
    for fp in sorted(tr_dir.glob("*.json")):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        w_star = solve_milp_exact_bound(obj, 5)
        w_noop = next(r["total_ops"] for r in raw_replay["rows"] if r["file"] == fp.name and r["policy"] == "noop")
        milp_cache[fp.name] = (w_noop, w_star)

    head_rows = []
    for r in raw_replay["rows"]:
        w_noop, w_star = milp_cache[r["file"]]
        gcr = 100.0 * (w_noop - r["total_ops"]) / max(w_noop - w_star, 1e-9)
        head_rows.append({
            **r,
            "noop_ops": w_noop,
            "milp_star_ops": round(w_star, 2),
            "gcr_pct": round(gcr, 2),
        })
    write_csv(head_rows, RESULTS_DIR / "s2_headroom_100k_to_10m.csv")
    log(f"[S2-Headroom] Computed MILP GCR across {len(milp_cache)} 1M-10M traces ({len(head_rows)} rows).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["SMOKE", "R1", "S1", "S2", "ALL"], default="ALL")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.suite in ("SMOKE", "ALL"):
        run_smoke_and_speedup()
    if args.suite in ("R1", "ALL"):
        run_suite_r1()
    if args.suite in ("S1", "ALL"):
        run_suite_s1()
    if args.suite in ("S2", "ALL"):
        run_suite_s2()
    log("ALL Stage 3 Rust & 100x Scale experiments complete.")


if __name__ == "__main__":
    main()
