#!/usr/bin/env python3
"""usearch_mut -- DynVecAudit 6th open-source vector engine backend (USearch mutable HNSW).

Implements the exact window_harness.py backend contract (protocol v1.9):
  - init build      -> build_cpu_ms / build_wall_ms (excluded from replay totals)
  - window CRUD     -> maint_cpu_ms (insert / update / erase via single-threaded USearch Index)
  - window queries  -> query_cpu_ms (process CPU) + per-query perf_counter wall (p50/p95/p99_us)
  - recall          -> computed via qbam_io.mean_recall_at_k against trace exact GT

Knobs (args.bl, echoed in self.params):
  ef / expansion_search : search beam width (default 64)
  M / connectivity      : HNSW degree M (default 16)
  ef_construction / expansion_add : build expansion (default 200)
  dtype                 : vector storage dtype ("f32" default, or "f16"/"i8")
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np
import qbam_io as qio
from usearch.index import Index


class UsearchMutBackend:
    name = "usearch_mut"

    def __init__(self, args, data):
        self.args = args
        self.bl = dict(getattr(args, "bl", {}) or {})
        self.k = int(data["k"])
        self.dim = int(data["dim"])
        self.ef = int(self.bl.get("ef", self.bl.get("expansion_search", 64)))
        self.M = int(self.bl.get("M", self.bl.get("connectivity", 16)))
        self.ef_construction = int(self.bl.get("ef_construction", self.bl.get("expansion_add", 200)))
        self.dtype = str(self.bl.get("dtype", "f32"))
        self.n_threads = 1

        ev = data["ev"]
        types = ev["type"]
        self._n_update_ev = int(np.sum(types == qio.EV_UPDATE))
        self._n_erase_ev = int(np.sum(types == qio.EV_ERASE))
        self._churn_trace = bool(self._n_update_ev + self._n_erase_ev)

        self.params = {
            "ef": self.ef,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "dtype": self.dtype,
            "metric": "l2sq",
            "k": self.k,
            "num_threads": 1,
            "architecture": "single_mutable_hnsw_usearch",
            "maintenance": "in_place_add_remove",
        }

        init = data["init"]
        self.n_init = int(init.shape[0])
        self._n_live = self.n_init
        self._n_synth = 0

        if self._churn_trace:
            elem_ids = ev["id"][types != qio.EV_QUERY].astype(np.int64)
            max_trace_id = int(elem_ids.max()) if elem_ids.size else -1
            if self.n_init:
                max_trace_id = max(max_trace_id, int(init["id"].max()))
            self._label_base = max_trace_id + 1
            self._label_base_u = np.uint64(self._label_base)
            self._next_label = self._label_base
            self._label_of = {}
            self._synth_ids = np.empty(self._n_update_ev, dtype=np.int64)
            self.params["delete_support"] = (
                "remove(label) on erase; update = remove(old_label) + add(fresh_label, new_vec)"
            )

        self.index = Index(
            ndim=self.dim,
            metric="l2sq",
            dtype=self.dtype,
            connectivity=self.M,
            expansion_add=self.ef_construction,
            expansion_search=self.ef,
        )

        ids0 = np.ascontiguousarray(init["id"], dtype=np.uint64)
        vecs0 = np.ascontiguousarray(data["pool"][init["vec_ref"]], dtype=np.float32)
        c0, t0 = time.process_time(), time.perf_counter()
        if ids0.size:
            self.index.add(ids0, vecs0, threads=1)
        self.build_cpu_ms = (time.process_time() - c0) * 1e3
        self.build_wall_ms = (time.perf_counter() - t0) * 1e3

    def _apply_crud(self, data, crud):
        types = crud["type"]
        n_ins = int(np.sum(types == qio.EV_INSERT))
        n_upd = int(np.sum(types == qio.EV_UPDATE))
        n_ers = int(np.sum(types == qio.EV_ERASE))
        status = "ok"
        timers = {"insert": 0.0, "update": 0.0, "erase": 0.0}

        if n_ins:
            sel = types == qio.EV_INSERT
            ins_ids = np.ascontiguousarray(crud["id"][sel], dtype=np.uint64)
            ins_vecs = np.ascontiguousarray(data["pool"][crud["vec_ref"][sel]], dtype=np.float32)
            c = time.process_time()
            self.index.add(ins_ids, ins_vecs, threads=1)
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
            old_labels = []
            for i, tid in enumerate(upd_ids.tolist()):
                cur = int(self._label_of.get(tid, tid))
                old_labels.append((i, cur))
            try:
                if old_labels:
                    self.index.remove(np.asarray([cur for _, cur in old_labels], dtype=np.uint64))
            except Exception as exc:
                for i, cur in old_labels:
                    try:
                        self.index.remove(cur)
                    except Exception:
                        ok_mask[i] = False
                        if status == "ok":
                            status = f"update_label_missing: {exc}"
            keep = np.nonzero(ok_mask)[0]
            if keep.size:
                self.index.add(new_labs[keep], upd_vecs[keep], threads=1)
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
            cur_list = [int(self._label_of.pop(tid, tid)) for tid in ers_ids.tolist()]
            if cur_list:
                try:
                    self.index.remove(np.asarray(cur_list, dtype=np.uint64))
                except Exception as exc:
                    for cur in cur_list:
                        try:
                            self.index.remove(cur)
                        except Exception:
                            if status == "ok":
                                status = f"erase_label_missing: {exc}"
            timers["erase"] = (time.process_time() - c) * 1e3
            self._n_live -= n_ers

        return n_ins, n_upd, n_ers, status, timers

    def _search_one(self, vec: np.ndarray) -> np.ndarray:
        matches = self.index.search(
            np.ascontiguousarray(vec, dtype=np.float32),
            self.k,
            threads=1,
        )
        raw = np.asarray(matches.keys, dtype=np.uint64).reshape(-1)[: self.k]
        if self._n_synth:
            out = raw.astype(np.int64)
            mask = raw >= self._label_base_u
            if bool(mask.any()):
                out[mask] = self._synth_ids[(raw[mask] - self._label_base_u).astype(np.int64)]
            if out.size < self.k:
                out = np.pad(out, (0, self.k - out.size), constant_values=-1)
            return out
        row = raw.astype(np.int64)
        if row.size < self.k:
            row = np.pad(row, (0, self.k - row.size), constant_values=-1)
        return row

    def run_window(self, data, w, crud, qe):
        c0 = time.process_time()
        n_ins, n_upd, n_ers, status, timers = self._apply_crud(data, crud)
        maint_ms = (time.process_time() - c0) * 1e3

        self.index.expansion_search = self.ef
        qvec = data["pool"][qe["vec_ref"]]
        nq = int(qvec.shape[0])
        ret = np.full((nq, self.k), -1, dtype=np.int64)
        walls = []
        c1 = time.process_time()
        for i in range(nq):
            t0 = time.perf_counter()
            ret[i] = self._search_one(qvec[i])
            walls.append(time.perf_counter() - t0)
        query_ms = (time.process_time() - c1) * 1e3

        gt = data["gt_ids"][data["q_off"][w]: data["q_off"][w] + nq]
        sw = sorted(walls)

        def pct(p):
            if not sw:
                return None
            idx = min(len(sw) - 1, int(round((len(sw) - 1) * p / 100.0)))
            return sw[idx] * 1e6

        rec = {
            "n_insert": n_ins,
            "n_update": n_upd,
            "n_erase": n_ers,
            "n_queries": nq,
            "query_cpu_ms": query_ms,
            "maint_cpu_ms": maint_ms,
            "insert_cpu_ms": maint_ms if not self._churn_trace else timers["insert"],
            "explicit_maintenance_cpu_ms": 0.0,
            "rebuild_triggered": False,
            "p50_us": pct(50),
            "p95_us": pct(95),
            "p99_us": pct(99),
            "mean_recall": qio.mean_recall_at_k(ret, gt, self.k),
            "index_n": int(self._n_live),
            "window_status": status,
        }
        if self._churn_trace:
            rec["update_cpu_ms"] = timers["update"]
            rec["erase_cpu_ms"] = timers["erase"]
        return rec


def make_backend(args, data):
    return UsearchMutBackend(args, data)
