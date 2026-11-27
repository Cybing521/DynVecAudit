#!/usr/bin/env python3
"""hnswlib_mut -- E3 external baseline: ONE mutable HNSW graph, updated in place.

Architecture under test
-----------------------
This is the "single mutable graph" family that IP-DiskANN (SIGMOD 2024) argues for SSD-resident
search: a single graph, no immutable segments, no merge step; every insert rewires the one graph
in place.  In memory that architecture is exactly hnswlib's dynamic index, so this backend
isolates the *update architecture* variable of E3 (one mutable graph vs. segment/merge designs)
without any disk component.

Contract (harness/backends/README.md, E3 protocol v1.9)
------------------------------------------------------
  init build      -> build_cpu_ms / build_wall_ms (excluded from the totals)
  window CRUD     -> maint_cpu_ms (in-place graph update IS the maintenance work)
  window queries  -> query_cpu_ms (process time) + per-query perf_counter wall
  recall          -> harness/GT only; this file never post-filters and never reads GT

Knobs (args.bl, all echoed in self.params)
  ef                search breadth, default 64
  M                 graph degree, default 16
  ef_construction   build breadth, default 200

Insert-only vs CRUD
-------------------
The q2500 main grid is insert-only (u=10%, 0 update / 0 erase events) and keeps the previous code
path, params and fields verbatim.  The crud30 / crud50 grids carry EV_UPDATE=1 and EV_ERASE=2:

  * init / insert labels ARE the trace ids (same as the insert-only path).
  * update: mark_deleted(current_label) then add_items(new_vec, fresh_label, replace_deleted=True).
    replace_deleted=True makes hnswlib reuse the slot that mark_deleted just freed, so an update
    is slot-neutral and never grows max_elements; add_items() runs the C++ updatePoint() graph
    repair on the reused slot.  The fresh label comes from a monotonic allocator whose base sits
    above every trace id; id -> current label is tracked only for ids that were updated, and a
    compact label -> trace-id table translates search hits back to trace ids.
  * erase: mark_deleted(current_label).
  * every mark/add call is charged to maint_cpu_ms (split into insert/update/erase sub-timers).

Why the label allocator is monotonic (no free list)
---------------------------------------------------
hnswlib leaves a deleted label inside its label_lookup_ map.  If a label were re-issued for a
different slot while its old deleted slot was still parked in the replace-deleted pool, a later
reuse of that old slot would erase the live label from label_lookup_ (the C++ replace path erases
the label stored on the slot).  Monotonic labels (never re-issued) avoid that failure mode; the
label space is uint64 and bounded by max(trace id) + #update events, so exhaustion is not a risk.

Capacity
--------
max_elements is sized from the trace (init + all insert events + 5% slack + 1024).  Updates are
slot-neutral and erases only free slots, so the physical slot count never exceeds init + inserts;
resize_index() still guards any longer stream.
"""
from __future__ import annotations

import os

# contract rule 2: single-threaded worker -- set before hnswlib is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np

import qbam_io as qio  # the harness puts <external>/tools on sys.path before loading backends

import hnswlib

_UINT64_MAX = np.uint64(18446744073709551615)  # hnswlib's "no result" sentinel


class HnswlibMutBackend:
    name = "hnswlib_mut"

    def __init__(self, args, data):
        self.args = args
        self.bl = dict(getattr(args, "bl", {}) or {})
        self.k = int(data["k"])
        self.dim = int(data["dim"])
        self.ef = int(self.bl.get("ef", 64))
        self.M = int(self.bl.get("M", 16))
        self.ef_construction = int(self.bl.get("ef_construction", 200))
        self.n_threads = 1

        ev = data["ev"]
        types = ev["type"]
        self._n_update_ev = int(np.sum(types == qio.EV_UPDATE))
        self._n_erase_ev = int(np.sum(types == qio.EV_ERASE))
        self._churn_trace = bool(self._n_update_ev + self._n_erase_ev)

        self.params = {
            "ef": self.ef, "M": self.M, "ef_construction": self.ef_construction,
            "metric": "l2", "k": self.k, "num_threads": 1,
            "allow_replace_deleted": True,
            "architecture": "single_mutable_graph_in_place",
            "maintenance": "insert_is_maintenance (no explicit merge/consolidate)",
        }

        init = data["init"]
        self.n_init = int(init.shape[0])
        n_ins_ev = int(np.sum(types == qio.EV_INSERT))
        # every insert event owns a new label on this grid; updates reuse a deleted slot and erases
        # only free slots, so the physical slot count never exceeds init + inserts.  Keep headroom.
        self.max_elements = int(self.n_init + n_ins_ev + 1024 + 0.05 * n_ins_ev)
        self.n_slots = self.n_init
        self._n_live = self.n_init

        ids0 = np.ascontiguousarray(init["id"], dtype=np.uint64)
        self._n_synth = 0            # >0 only after the first update (churn traces); 0 => fast path
        if self._churn_trace:
            # fresh-label allocator + id->label map + label->trace-id translation table
            elem_ids = ev["id"][types != qio.EV_QUERY].astype(np.int64)
            max_trace_id = int(elem_ids.max()) if elem_ids.size else -1
            if self.n_init:
                max_trace_id = max(max_trace_id, int(init["id"].max()))
            self._label_base = max_trace_id + 1
            self._label_base_u = np.uint64(self._label_base)
            self._next_label = self._label_base
            self._label_of = {}
            self._synth_ids = np.empty(self._n_update_ev, dtype=np.int64)
            self._n_synth = 0
            self.params["delete_support"] = (
                "mark_deleted(label) on erase; update = mark_deleted(old label) + add_items(new "
                "vector, fresh label, replace_deleted=True), reusing the freed slot")
            self.params["id_map"] = (
                "init/insert labels = trace ids; update labels from a monotonic allocator >= "
                f"{self._label_base} (never re-issued)")
            self.params["maintenance"] = "insert/update/erase are the in-place maintenance work"

        self.index = hnswlib.Index(space="l2", dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, M=self.M,
                              ef_construction=self.ef_construction, allow_replace_deleted=True)
        self.index.set_ef(self.ef)
        self.index.set_num_threads(1)

        vecs = np.ascontiguousarray(data["pool"][init["vec_ref"]], dtype=np.float32)
        c0, t0 = time.process_time(), time.perf_counter()
        self.index.add_items(vecs, ids0, num_threads=1, replace_deleted=True)
        self.build_cpu_ms = (time.process_time() - c0) * 1e3
        self.build_wall_ms = (time.perf_counter() - t0) * 1e3

    # ---- maintenance -------------------------------------------------------------
    def _ensure_capacity(self, extra: int) -> None:
        need = self.n_slots + extra
        if need > self.max_elements:
            self.max_elements = int(need * 1.2) + 1024
            self.index.resize_index(self.max_elements)

    def insert(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        if vecs.shape[0] == 0:
            return
        self._ensure_capacity(int(vecs.shape[0]))
        self.index.add_items(np.ascontiguousarray(vecs, dtype=np.float32),
                             np.ascontiguousarray(ids, dtype=np.uint64),
                             num_threads=1, replace_deleted=True)
        self.n_slots += int(vecs.shape[0])

    def _apply_crud(self, data, crud):
        """Apply this window's non-query events in place.  Returns counts/status/per-op timers."""
        types = crud["type"]
        n_ins = int(np.sum(types == qio.EV_INSERT))
        n_upd = int(np.sum(types == qio.EV_UPDATE))
        n_ers = int(np.sum(types == qio.EV_ERASE))
        status = "ok"
        timers = {"insert": 0.0, "update": 0.0, "erase": 0.0}

        if n_ins:
            sel = types == qio.EV_INSERT
            c = time.process_time()
            self.insert(data["pool"][crud["vec_ref"][sel]], crud["id"][sel])
            timers["insert"] = (time.process_time() - c) * 1e3
            self._n_live += n_ins

        if n_upd:
            sel = types == qio.EV_UPDATE
            upd_ids = crud["id"][sel].astype(np.int64)
            upd_vecs = np.ascontiguousarray(data["pool"][crud["vec_ref"][sel]], dtype=np.float32)
            c = time.process_time()
            lo = self._n_synth
            new_labs = np.arange(self._next_label, self._next_label + n_upd, dtype=np.uint64)
            self._next_label += n_upd
            ok_mask = np.ones(n_upd, dtype=bool)
            for i, tid in enumerate(upd_ids.tolist()):
                cur = self._label_of.get(tid, tid)
                try:
                    self.index.mark_deleted(int(cur))
                except RuntimeError as exc:  # label never inserted / already deleted (invalid trace)
                    ok_mask[i] = False
                    if status == "ok":
                        status = f"update_label_missing: {exc}"
            keep = np.nonzero(ok_mask)[0]
            if keep.size:
                self.index.add_items(upd_vecs[keep], new_labs[keep],
                                     num_threads=1, replace_deleted=True)
            # label -> trace-id table; failed positions never get inserted so their value is inert
            self._synth_ids[lo:lo + n_upd] = -1
            self._synth_ids[lo + keep] = upd_ids[keep]
            self._n_synth = lo + n_upd
            for i in keep.tolist():
                self._label_of[int(upd_ids[i])] = int(new_labs[i])
            timers["update"] = (time.process_time() - c) * 1e3

        if n_ers:
            sel = types == qio.EV_ERASE
            ers_ids = crud["id"][sel].astype(np.int64)
            c = time.process_time()
            for tid in ers_ids.tolist():
                cur = self._label_of.pop(tid, tid)
                try:
                    self.index.mark_deleted(int(cur))
                except RuntimeError as exc:
                    if status == "ok":
                        status = f"erase_label_missing: {exc}"
            timers["erase"] = (time.process_time() - c) * 1e3
            self._n_live -= n_ers

        return n_ins, n_upd, n_ers, status, timers

    # ---- search ------------------------------------------------------------------
    def search_one(self, vec: np.ndarray) -> np.ndarray:
        labels, _ = self.index.knn_query(
            np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1),
            k=self.k, num_threads=1)
        raw = np.asarray(labels).reshape(-1)
        if self._n_synth:
            # translate the update labels (>= label_base) back to trace ids; init/insert labels are
            # already trace ids, so this only touches the synthesised range.
            raw = raw[: self.k]
            out = raw.astype(np.int64)
            mask = (raw >= self._label_base_u) & (raw != _UINT64_MAX)
            if bool(mask.any()):
                out[mask] = self._synth_ids[(raw[mask] - self._label_base_u).astype(np.int64)]
            if out.size < self.k:
                out = np.pad(out, (0, self.k - out.size), constant_values=-1)
            return out
        row = raw.astype(np.int64)[: self.k]
        if row.size < self.k:
            row = np.pad(row, (0, self.k - row.size), constant_values=-1)
        return row

    # ---- harness hook ------------------------------------------------------------
    def run_window(self, data, w, crud, qe):
        c0 = time.process_time()
        n_ins, n_upd, n_ers, status, timers = self._apply_crud(data, crud)
        maint_ms = (time.process_time() - c0) * 1e3

        self.index.set_ef(self.ef)
        qvec = data["pool"][qe["vec_ref"]]
        nq = int(qvec.shape[0])
        ret = np.full((nq, self.k), -1, dtype=np.int64)
        walls = []
        c1 = time.process_time()
        for i in range(nq):
            t0 = time.perf_counter()
            ret[i] = self.search_one(qvec[i])
            walls.append(time.perf_counter() - t0)
        query_ms = (time.process_time() - c1) * 1e3

        gt = data["gt_ids"][data["q_off"][w]: data["q_off"][w] + nq]
        sw = sorted(walls)

        def pct(p):
            if not sw:
                return None
            i = min(len(sw) - 1, int(round((len(sw) - 1) * p / 100.0)))
            return sw[i] * 1e6

        rec = {
            "n_insert": n_ins, "n_update": n_upd, "n_erase": n_ers, "n_queries": nq,
            "query_cpu_ms": query_ms,
            "maint_cpu_ms": maint_ms,
            # on an insert-only window all maintenance is the in-place insert (v1.9 semantics)
            "insert_cpu_ms": maint_ms if not self._churn_trace else timers["insert"],
            "explicit_maintenance_cpu_ms": 0.0,   # no separate merge/consolidate step
            "rebuild_triggered": False,
            "p50_us": pct(50), "p95_us": pct(95), "p99_us": pct(99),
            "mean_recall": qio.mean_recall_at_k(ret, gt, self.k),
            "index_n": int(self._n_live),          # live (non-deleted) elements, comparable to ntotal()
            "window_status": status,
        }
        if self._churn_trace:
            rec["update_cpu_ms"] = timers["update"]
            rec["erase_cpu_ms"] = timers["erase"]
            rec["index_slots"] = int(self.index.get_current_count())
        return rec


def make_backend(args, data):
    return HnswlibMutBackend(args, data)
