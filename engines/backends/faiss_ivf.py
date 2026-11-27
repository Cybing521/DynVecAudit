#!/usr/bin/env python3
"""faiss_ivf -- FAISS IndexIVFFlat (wrapped in IndexIDMap2) external baseline for the E3 grid.

Architecture family: inverted-file / coarse-quantizer index (IVF-Flat; Jegou, Douze & Schmid,
"Product Quantization for Nearest Neighbor Search", TPAMI 2011, and the FAISS reference
implementation).  IVF-Flat is the industry-default *inverted-list* baseline: k-means coarse
quantizer + exact residual vectors stored in per-list arrays.  Insert appends to one list (no
graph repair, no split/merge policy); delete physically drops the ids from the lists.

Contract: harness/backends/README.md (E3 protocol v1.9).
  * every knob comes from --bl-args and is echoed in self.params
  * time.process_time() for CPU, time.perf_counter() per query for wall, faiss.omp_set_num_threads(1)
  * initial train+add -> build_cpu_ms (excluded); per-window add()/remove()/rebuild -> maint_cpu_ms
  * ids are the trace ids, not IVF row numbers (insert-only path: IndexIDMap2 keeps the external
    id map; churn path: IndexIVFFlat stores the trace ids natively in its InvertedLists)

Knobs (--bl-args):
  nlist          number of coarse centroids (default 256)
  nprobe         lists scanned per query (default 16)
  train_sample   #initial vectors used to fit the coarse quantizer (default 65536; <=0 = all)
  retrain_every  0 = never (default); >0 -> every N windows retrain the coarse quantizer and
                 re-add the whole live set (charged to maint_cpu_ms, rebuild_triggered=True)
  sample_seed    RNG seed for the training subsample (default 20261118).  Note: FAISS k-means
                 init uses its own ClusteringParameters.seed (1234 by default), so a given
                 training sample is reproducible but the seed knob does not reach k-means.

Insert-only vs CRUD
-------------------
The q2500 main grid is insert-only (verified: 400k EV_INSERT + 100k EV_QUERY, 0 update / 0 erase)
and keeps the previous code path and previous params/fields verbatim.  The crud30 / crud50 grids
carry EV_UPDATE=1 and EV_ERASE=2; for those the standard IVF equivalent is used, and all of it is
charged to maint_cpu_ms:
  * erase  -> remove_ids(IDSelectorBatch(ids))            (physical removal from the lists)
  * update -> remove_ids(id) then add_with_ids(new_vec, id)
  * per window ALL update ids + erase ids go into ONE IDSelectorBatch, because the IDMap2 remove
    path is an O(ntotal) scan of the inverted lists -- one call per window instead of per event.
  * window_status stays "ok" only if every removal was applied; a count mismatch is reported.

IndexIDMap2 note: on the churn path the IDMap wrapper is NOT used.  faiss 1.16's
IndexIDMap2::remove_ids is unsound on top of IndexIVF (the IVF keeps explicit ids and does not
renumber survivors, the wrapper rebuilds a compacted id_map), which collapses recall after the
first delete and then aborts at IndexIDMap.cpp:292; IndexIVFFlat's native add_with_ids/remove_ids
give the identical external semantics (search returns the trace ids that were added).
"""
from __future__ import annotations

import time

import numpy as np

import qbam_io as qio  # the harness puts <external>/tools on sys.path before loading backends

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "faiss_ivf backend requires faiss: venv/bin/pip install faiss-cpu"
    ) from exc

# single-threaded by contract (README rule 2).  omp is a process-global, set at import time so the
# build (k-means) and the online search path both see it.
faiss.omp_set_num_threads(1)


class FaissIVFBackend:
    name = "faiss_ivf"

    def __init__(self, args, data):
        self.args = args
        self.bl = dict(getattr(args, "bl", {}) or {})
        self.k = int(data["k"])
        self.dim = int(data["dim"])
        self.nlist = int(self.bl.get("nlist", 256))
        self.nprobe = int(self.bl.get("nprobe", 16))
        self.train_sample = int(self.bl.get("train_sample", 65536))
        self.retrain_every = int(self.bl.get("retrain_every", 0))
        self.sample_seed = int(self.bl.get("sample_seed", 20261118))
        self.n_threads = 1

        ev = data["ev"]
        types = ev["type"]
        self._n_update_ev = int(np.sum(types == qio.EV_UPDATE))
        self._n_erase_ev = int(np.sum(types == qio.EV_ERASE))
        self._churn_trace = bool(self._n_update_ev + self._n_erase_ev)

        self.params = {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "train_sample": self.train_sample,
            "retrain_every": self.retrain_every,
            "sample_seed": self.sample_seed,
            "index": "IndexIVFFlat(IndexFlatL2 quantizer, METRIC_L2) + IndexIDMap2",
            "metric": "L2",
            "threads": 1,
            "k": self.k,
            "maintenance": "add() per window; optional retrain_every full rebuild",
            "delete_support": "none (insert-only grid; non-insert CRUD reported unsupported)",
        }
        if self._churn_trace:
            # churn-only knobs: an insert-only run keeps the v1.9 params verbatim
            self.params["index"] = (
                "IndexIVFFlat(IndexFlatL2 quantizer, METRIC_L2), native int64 trace ids in the "
                "InvertedLists (no IndexIDMap layer)")
            self.params["delete_support"] = (
                "remove_ids(IDSelectorBatch, one batch per window) for erase and for the remove "
                "half of update"
            )
            self.params["update_semantics"] = (
                "remove_ids(update ids) -> add_with_ids(new vec, same trace id); erase = remove_ids"
            )

        init = data["init"]
        vec_ref0 = init["vec_ref"].astype(np.int64)
        ids0 = init["id"].astype(np.int64)
        X0 = np.ascontiguousarray(data["pool"][vec_ref0], dtype=np.float32)

        if self._churn_trace:
            # exact live set (id -> pool ref) for the optional full rebuild; updates replace, erases pop
            self._live_ref = {int(i): int(r) for i, r in zip(ids0.tolist(), vec_ref0.tolist())}
        else:
            # live-set bookkeeping (refs into data["pool"], no vector copies) -- only needed for the
            # optional periodic-rebuild mode; keeping it always is ~4 bytes x nvec and harmless.
            self._live_refs = [vec_ref0.copy()]
            self._live_ids = [ids0.copy()]

        c0, t0 = time.process_time(), time.perf_counter()
        self.quantizer = faiss.IndexFlatL2(self.dim)
        self._new_index()
        Xtr = self._training_sample(X0)
        self.index.train(Xtr)
        self.index.add_with_ids(X0, ids0)
        self.ivf.nprobe = min(self.nprobe, self.nlist)
        self.build_cpu_ms = (time.process_time() - c0) * 1e3
        self.build_wall_ms = (time.perf_counter() - t0) * 1e3
        self.n_rebuilds = 0

    # ---------------- internals ----------------
    def _new_index(self):
        """Allocate a fresh IVF-Flat that keeps trace ids as the stored (external) ids.

        Insert-only traces keep the v1.9 layout, IndexIVFFlat wrapped in IndexIDMap2.
        Churn traces use the IVF's *native* id support instead: faiss 1.16's
        IndexIDMap2::remove_ids is unsound on top of IndexIVF -- IndexIVF keeps the explicit
        int64 ids in its InvertedLists and does not renumber the survivors after remove_ids(),
        while the IDMap wrapper rebuilds a compacted id_map; the surviving labels then exceed
        id_map.size() (search maps them to -1) and the next remove_ids() reads the map out of
        bounds (observed: recall collapse then FAISS_ASSERT abort at IndexIDMap.cpp:292).
        IndexIVFFlat supports add_with_ids()/remove_ids() natively over the stored ids, so search
        still returns exactly the trace ids that were added -- same external semantics, no map.
        """
        self.ivf = faiss.IndexIVFFlat(self.quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        if self._churn_trace:
            self.index = self.ivf
        else:
            # IndexIDMap2: the IVF list entries stay int64 row ids of the *wrapped* index, and the
            # map translates them back to the trace ids we pass to add_with_ids()/return from search().
            self.index = faiss.IndexIDMap2(self.ivf)

    def _training_sample(self, X):
        n = X.shape[0]
        if self.train_sample <= 0 or n <= self.train_sample:
            return np.ascontiguousarray(X, dtype=np.float32)
        rs = np.random.RandomState(self.sample_seed)
        pick = rs.choice(n, self.train_sample, replace=False)
        pick.sort()  # keep the sample contiguous-ish; k-means does not care about order
        return np.ascontiguousarray(X[pick], dtype=np.float32)

    def _rebuild(self, data):
        """Retrain the coarse quantizer on the live set and re-add every live vector."""
        if self._churn_trace:
            ids = np.fromiter(self._live_ref.keys(), dtype=np.int64, count=len(self._live_ref))
            refs = np.fromiter(self._live_ref.values(), dtype=np.int64, count=len(self._live_ref))
        else:
            refs = np.concatenate(self._live_refs) if len(self._live_refs) > 1 else self._live_refs[0]
            ids = np.concatenate(self._live_ids) if len(self._live_ids) > 1 else self._live_ids[0]
        X = np.ascontiguousarray(data["pool"][refs], dtype=np.float32)
        self._new_index()
        self.index.train(self._training_sample(X))
        self.index.add_with_ids(X, ids)
        self.ivf.nprobe = min(self.nprobe, self.nlist)
        self.n_rebuilds += 1

    # ---------------- hooks ----------------
    def insert(self, vecs, ids, vec_refs=None):
        if vecs.shape[0] == 0:
            return
        self.index.add_with_ids(vecs, ids)
        if vec_refs is None:
            return
        if self._churn_trace:
            for tid, ref in zip(ids.tolist(), np.asarray(vec_refs).tolist()):
                self._live_ref[int(tid)] = int(ref)
        else:
            self._live_refs.append(np.asarray(vec_refs, dtype=np.int64).copy())
            self._live_ids.append(np.asarray(ids, dtype=np.int64).copy())

    def search_one(self, vec):
        _, lab = self.index.search(vec.reshape(1, self.dim), self.k)
        return lab[0].astype(np.int64)

    def _query_stats(self, data, w, qe):
        qvec = data["pool"][qe["vec_ref"]].astype(np.float32, copy=False)
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

        return {
            "n_queries": nq,
            "query_cpu_ms": query_ms,
            "p50_us": pct(50),
            "p95_us": pct(95),
            "p99_us": pct(99),
            "mean_recall": qio.mean_recall_at_k(ret, gt, self.k),
        }

    def run_window(self, data, w, crud, qe):
        if self._churn_trace:
            return self._run_window_churn(data, w, crud, qe)
        return self._run_window_insert_only(data, w, crud, qe)

    def _run_window_insert_only(self, data, w, crud, qe):
        """v1.9 path, verbatim semantics/fields (the q2500 main grid never takes any other path)."""
        ins = crud
        if len(ins):
            vecs = np.ascontiguousarray(data["pool"][ins["vec_ref"]], dtype=np.float32)
            ids = ins["id"].astype(np.int64)
        else:
            vecs = np.zeros((0, self.dim), np.float32)
            ids = np.zeros(0, np.int64)

        c0 = time.process_time()
        self.insert(vecs, ids, ins["vec_ref"])
        t1 = time.process_time()
        insert_ms = (t1 - c0) * 1e3
        rebuild = bool(self.retrain_every > 0 and (w + 1) % self.retrain_every == 0)
        if rebuild:
            self._rebuild(data)
        t2 = time.process_time()
        rebuild_ms = (t2 - t1) * 1e3
        maint_ms = (t2 - c0) * 1e3

        q = self._query_stats(data, w, qe)
        return {
            "n_insert": int(vecs.shape[0]),
            "n_queries": q["n_queries"],
            "query_cpu_ms": q["query_cpu_ms"],
            "maint_cpu_ms": maint_ms,
            "insert_cpu_ms": insert_ms,
            "explicit_maintenance_cpu_ms": rebuild_ms,
            "rebuild_cpu_ms": rebuild_ms,
            "rebuild_triggered": rebuild,
            "p50_us": q["p50_us"],
            "p95_us": q["p95_us"],
            "p99_us": q["p99_us"],
            "mean_recall": q["mean_recall"],
            "index_n": int(self.index.ntotal),
            "window_status": "ok",
        }

    def _run_window_churn(self, data, w, crud, qe):
        types = crud["type"]
        n_ins = int(np.sum(types == qio.EV_INSERT))
        n_upd = int(np.sum(types == qio.EV_UPDATE))
        n_ers = int(np.sum(types == qio.EV_ERASE))
        status = "ok"

        c0 = time.process_time()
        if n_ins:
            sel = types == qio.EV_INSERT
            self.insert(np.ascontiguousarray(data["pool"][crud["vec_ref"][sel]], dtype=np.float32),
                        crud["id"][sel].astype(np.int64), crud["vec_ref"][sel])
        t1 = time.process_time()
        insert_ms = (t1 - c0) * 1e3

        remove_ms = 0.0
        update_ms = 0.0
        if n_upd + n_ers:
            upd_sel = types == qio.EV_UPDATE
            ers_sel = types == qio.EV_ERASE
            upd_ids = crud["id"][upd_sel].astype(np.int64)
            ers_ids = crud["id"][ers_sel].astype(np.int64)
            # one batched O(ntotal) remove pass for every update + erase target of this window
            rem_ids = np.concatenate([upd_ids, ers_ids])
            n_removed = int(self.index.remove_ids(faiss.IDSelectorBatch(rem_ids)))
            remove_ms = (time.process_time() - t1) * 1e3

            t2 = time.process_time()
            if n_upd:
                self.insert(np.ascontiguousarray(data["pool"][crud["vec_ref"][upd_sel]], dtype=np.float32),
                            upd_ids, crud["vec_ref"][upd_sel])
            if n_ers:
                for tid in ers_ids.tolist():
                    self._live_ref.pop(int(tid), None)
            update_ms = (time.process_time() - t2) * 1e3

            expected = n_upd + n_ers
            if n_removed != expected:
                status = f"remove_mismatch: removed={n_removed} expected={expected}"

        rebuild = bool(self.retrain_every > 0 and (w + 1) % self.retrain_every == 0)
        t3 = time.process_time()
        if rebuild:
            self._rebuild(data)
        t4 = time.process_time()
        rebuild_ms = (t4 - t3) * 1e3
        maint_ms = (t4 - c0) * 1e3

        q = self._query_stats(data, w, qe)
        return {
            "n_insert": n_ins,
            "n_update": n_upd,
            "n_erase": n_ers,
            "n_queries": q["n_queries"],
            "query_cpu_ms": q["query_cpu_ms"],
            "maint_cpu_ms": maint_ms,
            "insert_cpu_ms": insert_ms,
            "remove_cpu_ms": remove_ms,
            "update_cpu_ms": update_ms,
            "explicit_maintenance_cpu_ms": rebuild_ms,
            "rebuild_cpu_ms": rebuild_ms,
            "rebuild_triggered": rebuild,
            "p50_us": q["p50_us"],
            "p95_us": q["p95_us"],
            "p99_us": q["p99_us"],
            "mean_recall": q["mean_recall"],
            "index_n": int(self.index.ntotal),
            "window_status": status,
        }


def make_backend(args, data):
    return FaissIVFBackend(args, data)
