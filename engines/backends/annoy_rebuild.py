#!/usr/bin/env python3
"""annoy_rebuild -- E3 external baseline: Annoy static index with periodic FULL rebuild.

Architecture under test
-----------------------
Annoy's forest is immutable after build(), so the only maintenance policy it can implement is
"buffer inserts, rebuild the whole index from scratch every N windows".  In E3 this is the
*periodic-rebuild* extreme of the update-architecture axis: zero per-insert cost, but every
rebuild window pays for building a new forest over the entire accumulated set
(init + all inserts so far), and inserts are invisible to search until the next rebuild.

Contract (harness/backends/README.md, E3 protocol v1.9)
------------------------------------------------------
  build at t0            -> build_cpu_ms / build_wall_ms (excluded from the totals)
  rebuild inside window  -> maint_cpu_ms of THAT window, rebuild_triggered=True
                           (explicit_maintenance_cpu_ms = rebuild part, insert_cpu_ms = buffering part)
  window queries         -> query_cpu_ms (process time) + per-query perf_counter wall
  recall                 -> harness/GT only; this file never post-filters and never reads GT

Knobs (args.bl, all echoed in self.params)
  n_trees        number of trees per build, default 10
  search_k       nodes to inspect per query, default 2000
  rebuild_every  rebuild every N windows, default 4 (0 = never rebuild: pure static baseline)

Notes / limitations
  * annoy is single-threaded (no thread knob); OMP_NUM_THREADS is forced to 1 for hygiene.
  * Static index: EV_UPDATE / EV_ERASE cannot be applied.  Per contract rule 5 they are NOT
    applied silently -- the window is flagged in window_status.  The q2500 main grid is
    insert-only, so those branches never fire there.
  * Rebuild cost grows linearly with the accumulated set, so the last windows of a 40-window
    trace are the expensive ones (this is the point of the baseline, not an artifact).
"""
from __future__ import annotations

import os

# contract rule 2: annoy itself is single-threaded, keep the process single-threaded too
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np

import qbam_io as qio  # the harness puts <external>/tools on sys.path before loading backends

from annoy import AnnoyIndex


class AnnoyRebuildBackend:
    name = "annoy_rebuild"

    def __init__(self, args, data):
        self.args = args
        self.bl = dict(getattr(args, "bl", {}) or {})
        self.k = int(data["k"])
        self.dim = int(data["dim"])
        self.n_trees = int(self.bl.get("n_trees", 10))
        self.search_k = int(self.bl.get("search_k", 2000))
        self.rebuild_every = int(self.bl.get("rebuild_every", 4))
        self.n_threads = 1
        self.params = {
            "n_trees": self.n_trees, "search_k": self.search_k,
            "rebuild_every": self.rebuild_every, "metric": "euclidean", "k": self.k,
            "index_type": "static_forest",
            "maintenance": "periodic_full_rebuild (buffer inserts between rebuilds)",
        }

        init = data["init"]
        self.vecs = np.ascontiguousarray(data["pool"][init["vec_ref"]], dtype=np.float32)
        self.ids = np.ascontiguousarray(init["id"], dtype=np.int64)
        assert self.ids.min() >= 0 and self.ids.max() < 2 ** 31, "annoy item ids must fit int32"

        self.n_seen = 0        # windows replayed (rebuild bookkeeping)
        self.n_rebuilds = 0
        c0, t0 = time.process_time(), time.perf_counter()
        self.index, self.index_n = self._build(self.vecs, self.ids)
        self.build_cpu_ms = (time.process_time() - c0) * 1e3
        self.build_wall_ms = (time.perf_counter() - t0) * 1e3

    # ---- maintenance -------------------------------------------------------------
    def _build(self, vecs: np.ndarray, ids: np.ndarray):
        idx = AnnoyIndex(self.dim, "euclidean")
        for i, v in zip(ids.tolist(), vecs):
            idx.add_item(int(i), v)
        idx.build(self.n_trees)
        return idx, int(ids.shape[0])

    def rebuild(self) -> None:
        """Full rebuild from init + every vector inserted so far (the only possible policy)."""
        self.index, self.index_n = self._build(self.vecs, self.ids)
        self.n_rebuilds += 1

    # ---- search ------------------------------------------------------------------
    def search_one(self, vec: np.ndarray) -> np.ndarray:
        sk = min(self.search_k, self.index_n)   # search_k > n_items is meaningless
        got = self.index.get_nns_by_vector(
            np.ascontiguousarray(vec, dtype=np.float32),
            self.k, search_k=sk, include_distances=False)
        row = np.asarray(got, dtype=np.int64)[: self.k]
        if row.size < self.k:
            row = np.pad(row, (0, self.k - row.size), constant_values=-1)
        return row

    # ---- harness hook ------------------------------------------------------------
    def run_window(self, data, w, crud, qe):
        types = crud["type"]
        n_ins = int(np.sum(types == qio.EV_INSERT))
        n_upd = int(np.sum(types == qio.EV_UPDATE))
        n_ers = int(np.sum(types == qio.EV_ERASE))
        status = "ok"
        if n_upd:
            status = "unsupported_update_static_index"
        if n_ers:
            status = "unsupported_erase_static_index"

        c0 = time.process_time()
        if n_ins:
            sel = types == qio.EV_INSERT
            self.vecs = np.concatenate(
                [self.vecs, np.ascontiguousarray(data["pool"][crud["vec_ref"][sel]], dtype=np.float32)],
                axis=0)
            self.ids = np.concatenate([self.ids, np.ascontiguousarray(crud["id"][sel], dtype=np.int64)])
        buffer_ms = (time.process_time() - c0) * 1e3

        self.n_seen += 1
        rebuild_ms = 0.0
        triggered = False
        if self.rebuild_every > 0 and (self.n_seen % self.rebuild_every) == 0:
            c1 = time.process_time()
            self.rebuild()
            rebuild_ms = (time.process_time() - c1) * 1e3
            triggered = True
        maint_ms = buffer_ms + rebuild_ms

        qvec = data["pool"][qe["vec_ref"]]
        nq = int(qvec.shape[0])
        ret = np.full((nq, self.k), -1, dtype=np.int64)
        walls = []
        c2 = time.process_time()
        for i in range(nq):
            t0 = time.perf_counter()
            ret[i] = self.search_one(qvec[i])
            walls.append(time.perf_counter() - t0)
        query_ms = (time.process_time() - c2) * 1e3

        gt = data["gt_ids"][data["q_off"][w]: data["q_off"][w] + nq]
        sw = sorted(walls)

        def pct(p):
            if not sw:
                return None
            i = min(len(sw) - 1, int(round((len(sw) - 1) * p / 100.0)))
            return sw[i] * 1e6

        return {
            "n_insert": n_ins, "n_update": n_upd, "n_erase": n_ers, "n_queries": nq,
            "query_cpu_ms": query_ms,
            "maint_cpu_ms": maint_ms,
            "insert_cpu_ms": buffer_ms,
            "explicit_maintenance_cpu_ms": rebuild_ms,   # the rebuild is the explicit maintenance
            "rebuild_triggered": triggered, "n_rebuilds": self.n_rebuilds,
            "p50_us": pct(50), "p95_us": pct(95), "p99_us": pct(99),
            "mean_recall": qio.mean_recall_at_k(ret, gt, self.k),
            "index_n": int(self.index_n),
            "window_status": status,
        }


def make_backend(args, data):
    return AnnoyRebuildBackend(args, data)
