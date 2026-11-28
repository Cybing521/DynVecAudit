#!/usr/bin/env python3
"""window_harness.py -- unified window-flow harness for the v3 external baselines.

Replays ONE Q-BAM trace (data/sift100k_u10/traces/<trace>/trace.bin, the frozen event order from
qbam/docs/engine_contract.md §4) against an external ANN system, window by window, and writes one
JSONL record per window plus a run header.  The schema is aligned with the engine's
type="window" rows (engine_contract §7) so EXTERNAL and Q-BAM runs can be paired per window:

  {"type":"window","system":"quake","trace":"uniform_s00","window":0,
   "n_insert":10000,"n_queries":500,"k":10,"threads":1,
   "query_cpu_ms":246.0,        # CPU spent answering that window's queries (single-thread)
   "maint_cpu_ms":60.5,         # CPU spent on that window's CRUD/maintenance work
   "p50_us":697.2,"p95_us":4011.8,"p99_us":6199.2,   # per-query WALL latency
   "mean_recall":0.9684,        # vs the trace's exact gt.bin rows for that window
   "loadavg":12.9,"pin_cpu":9,"params":{...},"window_status":"ok"}

  header: {"type":"run","system":...,"trace":...,"params":{...},"n_windows":40,"build_cpu_ms":...}

Backends
  * quake   -- in-process Python bindings.  add() CPU counts as maintenance; maintenance() is called
               every window (Quake's own split/merge policy) and its CPU + n_splits/n_deletes are
               recorded.  Per-query search calls give the latency distribution.
  * diskann -- out-of-process bin/diskann_driver (C++): builds the FreshDiskANN in-memory dynamic
               index from init, then per window inserts the window's vectors and searches the
               window's queries, all timed with clock_gettime inside the process.  For an insert-only
               stream the insert cost IS the maintenance cost (no delete/consolidate is triggered).

Timing honesty: the harness records the real loadavg at start/end.  Per the v3 protocol §6 tiers,
a run with loadavg >= 2 is class C (correctness only, not citable timing); use taskset+nice and a
quiet machine for class A/B runs.  CPU is CLOCK_PROCESS_CPUTIME_ID-equivalent; wall latency comes
from perf_counter/steady_clock around each individual call.

Usage
  python3 qbam/external/harness/window_harness.py --system quake --trace uniform_s00 \
      --data-root data/sift100k_u10 --out results/harness/quake_uniform_s00.jsonl
  python3 qbam/external/harness/window_harness.py --system diskann --trace hotspot_s00 ...
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ENGINES_DIR = Path(__file__).resolve().parent
EXT = Path(os.environ.get("QBAM_EXTERNAL_ROOT", str(ENGINES_DIR)))
sys.path.insert(0, str(ENGINES_DIR))
if (EXT / "tools").exists():
    sys.path.insert(0, str(EXT / "tools"))
import qbam_io as qio  # noqa: E402


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)


def load_common(args):
    root = Path(args.data_root)
    trdir = root / "traces" / args.trace
    pool, ph = qio.read_pool(root / "pool.bin")
    init, ih = qio.read_init(root / "init.bin")
    ev, th = qio.read_trace(trdir / "trace.bin")
    gt_ids, _, gh = qio.read_gt(trdir / "gt.bin")
    n_windows = th.fields["n_windows"]
    q_off, q_idx = {}, 0
    for w, _crud, qe in qio.window_events(ev, n_windows):
        q_off[w] = q_idx
        q_idx += len(qe)
    return dict(root=root, trdir=trdir, pool=pool, init=init, ev=ev, gt_ids=gt_ids,
                n_windows=n_windows, q_off=q_off, k=gh.fields["k"], dim=ph.fields["dim"],
                trace_sha256=None)


BUILTIN_BACKENDS = {"quake": "QuakeBackend", "diskann": "DiskannBackend"}


def load_backend(name, args, data):
    """Return a backend instance.

    Built-ins (quake, diskann) live in this file.  Any other name is a plugin at
    harness/backends/<name>.py exposing make_backend(args, data) plus the same
    attribute/method contract: name, n_threads, params, build_cpu_ms,
    build_wall_ms, run_window(data, w, crud, qe).
    """
    if name in BUILTIN_BACKENDS:
        return globals()[BUILTIN_BACKENDS[name]](args, data)
    import importlib.util
    path = Path(__file__).resolve().parent / "backends" / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"unknown --system {name!r}: no built-in and no {path}")
    spec = importlib.util.spec_from_file_location(f"qbam_backend_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make_backend"):
        raise SystemExit(f"{path} must define make_backend(args, data)")
    return mod.make_backend(args, data)


class QuakeBackend:
    name = "quake"

    def __init__(self, args, data):
        import torch
        import quake

        self.torch, self.quake = torch, quake
        torch.set_num_threads(1)
        self.args = args
        vec = torch.from_numpy(data["pool"][data["init"]["vec_ref"]].astype(np.float32, copy=True))
        ids = torch.from_numpy(data["init"]["id"].astype(np.int64))
        bp = quake.IndexBuildParams()
        bp.nlist = args.nlist
        bp.metric = "l2"
        bp.num_workers = 1
        bp.num_merge_workers = 1
        bp.use_numa = False
        bp.use_gpu = False
        self.index = quake.QuakeIndex()
        c0, t0 = time.process_time(), time.perf_counter()
        self.index.build(vec, ids, bp)
        self.build_cpu_ms = (time.process_time() - c0) * 1e3
        self.build_wall_ms = (time.perf_counter() - t0) * 1e3
        # Quake always runs the coordinator/main thread plus num_workers scan threads; with
        # num_workers=1 that is 2 online threads (both inherit the caller's taskset mask when
        # QUAKE_DISABLE_AFFINITY=1 is set -- see tools/patch_quake_affinity.py).
        self.n_threads = 2
        self.params = {"nlist": args.nlist, "metric": "l2", "num_workers": 1, "online_threads": 2, "k": args.k,
                       "search_mode": "recall_target" if args.recall_target > 0 else "nprobe",
                       "nprobe": args.nprobe, "recall_target": args.recall_target,
                       "maintenance": "called_per_window"}

    def run_window(self, data, w, crud, qe):
        torch = self.torch
        ins = crud[crud["type"] == qio.EV_INSERT]
        if len(ins) != len(crud):
            return {"window_status": "unsupported_non_insert_crud"}
        ins_vec = torch.from_numpy(data["pool"][ins["vec_ref"]].astype(np.float32, copy=True))
        ins_ids = torch.from_numpy(ins["id"].astype(np.int64))
        qvec = torch.from_numpy(data["pool"][qe["vec_ref"]].astype(np.float32, copy=True))
        nq = int(qvec.shape[0])
        sp = self.quake.SearchParams()
        sp.k = self.args.k
        sp.num_threads = 1
        sp.nprobe = int(self.args.nprobe)
        if self.args.recall_target > 0:
            sp.recall_target = float(self.args.recall_target)

        c0 = time.process_time()
        if ins_vec.shape[0] > 0:   # static (u=0) traces have no insert events
            self.index.add(ins_vec, ins_ids)
        add_cpu_ms = (time.process_time() - c0) * 1e3

        c0 = time.process_time()
        mt = self.index.maintenance()
        maint_cpu_ms = (time.process_time() - c0) * 1e3

        ret = np.empty((nq, self.args.k), dtype=np.int64)
        wall_us, cpu_us = [], []
        t_all = time.process_time()
        for i in range(nq):
            tq, cq = time.perf_counter(), time.process_time()
            res = self.index.search(qvec[i:i + 1], sp)
            wall_us.append((time.perf_counter() - tq) * 1e6)
            cpu_us.append((time.process_time() - cq) * 1e6)
            row = np.asarray(res.ids.cpu().numpy(), dtype=np.int64).reshape(-1)
            if row.size < self.args.k:
                row = np.pad(row, (0, self.args.k - row.size), constant_values=-1)
            ret[i] = row[: self.args.k]
        query_cpu_ms = (time.process_time() - t_all) * 1e3
        gt = data["gt_ids"][data["q_off"][w]: data["q_off"][w] + nq]
        recall = qio.mean_recall_at_k(ret, gt, self.args.k)
        return {
            "n_insert": int(len(ins)), "n_queries": nq,
            "query_cpu_ms": query_cpu_ms, "maint_cpu_ms": add_cpu_ms + maint_cpu_ms,
            "insert_cpu_ms": add_cpu_ms, "explicit_maintenance_cpu_ms": maint_cpu_ms,
            "n_splits": int(getattr(mt, "n_splits", -1)), "n_deletes": int(getattr(mt, "n_deletes", -1)),
            "p50_us": pct(wall_us, 50), "p95_us": pct(wall_us, 95), "p99_us": pct(wall_us, 99),
            "query_cpu_us_per_query_mean": float(np.mean(cpu_us)),
            "mean_recall": recall, "index_n": int(self.index.ntotal()),
            "window_status": "ok",
        }


class DiskannBackend:
    name = "diskann"

    def __init__(self, args, data):
        self.args = args
        # unique prep dir per output run: static and dynamic runs of the same trace must not share
        # the same converted files (a concurrent prep rewrites plan.jsonl/ins_*.bin and corrupts a
        # running driver -> "FATAL: bad .bin header"; observed 2026-11-17).
        self.work = EXT / "data" / f"prep_{Path(args.out).stem}"
        self.work.mkdir(parents=True, exist_ok=True)
        self.log = Path(args.out).with_suffix(".driver.log")
        cmd = [sys.executable, EXT / "tools" / "prep_smoke_data.py",
               "--data-root", args.data_root, "--trace", args.trace, "--out-dir", str(self.work),
               "--windows", "all", "--search-L", str(args.search_L)]
        with open(self.log, "ab") as lf:
            subprocess.run([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, check=True)
        self.meta = json.loads((self.work / "meta.json").read_text())
        self.build_cpu_ms = None
        self.build_wall_ms = None
        self.driver_out = Path(args.out).with_suffix(".driver.jsonl")
        if self.driver_out.exists():
            self.driver_out.unlink()
        self.n_threads = 1   # online search/insert path is single-threaded (build phase uses 4 threads offline)
        self.params = {"R": args.R, "Lbuild": args.Lbuild, "alpha": args.alpha, "k": args.k,
                       "search_L": args.search_L, "build_threads": args.build_threads,
                       "insert_threads": 1, "maintenance": "insert_is_maintenance (insert-only stream)"}
        pin = ["taskset", "-c", str(args.pin_cpu), "nice", "-n", "19"]
        env = dict(os.environ, OMP_NUM_THREADS="1")
        cmd = pin + [str(EXT / "bin" / (args.diskann_bin)), "--base", str(self.work / "base.bin"),
                     "--base_tags", str(self.work / "base_tags.u32"),
                     "--plan", str(self.work / "plan.jsonl"), "--out", str(self.driver_out),
                     "--dim", str(self.meta["dim"]), "--R", str(args.R), "--Lbuild", str(args.Lbuild),
                     "--alpha", str(args.alpha), "--search_L", str(args.search_L), "--K", str(args.k),
                     "--build_threads", str(args.build_threads), "--run_id", f"harness_{args.trace}"]
        self.t_run_start = time.time()
        with open(self.log, "ab") as lf:
            subprocess.run([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, check=True, env=env)
        self.driver_recs = [json.loads(x) for x in self.driver_out.read_text().splitlines() if x.strip()]
        runrec = next((r for r in self.driver_recs if r.get("type") == "run"), {})
        self.build_cpu_ms = runrec.get("build_cpu_ms")
        self.build_wall_ms = runrec.get("build_wall_ms")
        self.win = {r["window"]: r for r in self.driver_recs if r.get("type") == "window"}

    def run_window(self, data, w, crud, qe):
        rec = self.win.get(w)
        if rec is None:
            return {"window_status": "missing_driver_window"}
        gt = data["gt_ids"][data["q_off"][w]: data["q_off"][w] + int(rec["n_queries"])]
        res_path = self.work / f"res_w{w:03d}.jsonl"
        ret = np.full((gt.shape[0], self.args.k), -1, dtype=np.int64)
        for line in res_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ret[r["q"], :] = r["ids"][: self.args.k]
        recall = qio.mean_recall_at_k(ret, gt, self.args.k)
        return {
            "n_insert": rec["n_insert"], "n_queries": rec["n_queries"],
            "query_cpu_ms": rec["query_cpu_ms"], "maint_cpu_ms": rec["insert_cpu_ms"],
            "insert_cpu_ms": rec["insert_cpu_ms"], "insert_wall_ms": rec["insert_wall_ms"],
            "explicit_maintenance_cpu_ms": 0.0, "consolidate_triggered": False,
            "p50_us": rec["p50_wall_us"], "p95_us": rec["p95_wall_us"], "p99_us": rec["p99_wall_us"],
            "mean_recall": recall, "index_n": rec.get("index_n"),
            "failed_inserts": rec.get("failed_inserts"), "window_status": "ok",
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True,
                    help="built-in: quake|diskann; otherwise harness/backends/<name>.py with make_backend(args, data)")
    ap.add_argument("--data-root", default="data/sift100k_u10_q2500_v2")
    ap.add_argument("--trace", default="uniform_s00")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="all", help="all or comma list")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pin-cpu", type=int, default=9)
    ap.add_argument("--force", action="store_true")
    # quake
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--recall-target", type=float, default=-1.0)
    # diskann
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--Lbuild", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--search-L", type=int, default=64)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--diskann-bin", default="diskann_driver")
    ap.add_argument("--bl-args", default="{}",
                    help="JSON dict of backend-specific knobs, e.g. {\"ef\":64,\"rebuild_every\":8}")
    args = ap.parse_args()

    args.bl = json.loads(args.bl_args) if args.bl_args else {}
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = load_common(args)
    windows = list(range(data["n_windows"])) if args.windows == "all" else \
        [int(x) for x in args.windows.split(",")]

    lv0 = os.getloadavg()[0]
    t_start = time.time()
    backend = load_backend(args.system, args, data)   # built-in quake/diskann or harness/backends/<name>.py
    header = {"type": "run", "system": backend.name, "trace": args.trace,
              "data_root": str(data["root"]), "n_windows": len(windows), "k": args.k,
              "threads": backend.n_threads, "pin_cpu": args.pin_cpu, "params": backend.params,
              "build_cpu_ms": backend.build_cpu_ms, "build_wall_ms": backend.build_wall_ms,
              "loadavg_start": round(lv0, 2), "dim": int(data["dim"]),
              "harness": "qbam/external/harness/window_harness.py",
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lines = [header]
    for w, crud, qe in qio.window_events(data["ev"], data["n_windows"]):
        if w not in windows:
            continue
        rec = backend.run_window(data, w, crud, qe)
        rec.update({"type": "window", "system": backend.name, "trace": args.trace, "window": w,
                    "k": args.k, "threads": backend.n_threads, "pin_cpu": args.pin_cpu,
                    "params": backend.params, "loadavg": round(os.getloadavg()[0], 2)})
        lines.append(rec)
        print(f"[{backend.name}] w{w:02d} ins={rec.get('n_insert')} q={rec.get('n_queries')} "
              f"recall={rec.get('mean_recall'):.4f} qcpu={rec.get('query_cpu_ms'):.1f}ms "
              f"maint={rec.get('maint_cpu_ms'):.1f}ms p95={rec.get('p95_us'):.0f}us", flush=True)
    header["loadavg_end"] = round(os.getloadavg()[0], 2)
    header["wall_s"] = round(time.time() - t_start, 3)
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    header["peak_rss_mb"] = round(rss_raw / (1024.0 * 1024.0) if sys.platform == "darwin" else rss_raw / 1024.0, 2)
    lines[0] = header
    with open(out, "w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({"out": str(out), "n_windows": len(windows), "loadavg_start": header["loadavg_start"],
                      "loadavg_end": header["loadavg_end"], "wall_s": header["wall_s"],
                      "mean_recall_all": float(np.mean([r["mean_recall"] for r in lines[1:]])),
                      "sum_query_cpu_ms": float(np.sum([r["query_cpu_ms"] for r in lines[1:]])),
                      "sum_maint_cpu_ms": float(np.sum([r["maint_cpu_ms"] for r in lines[1:]]))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
